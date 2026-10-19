"""
BC_RNN + C-CaNCE + action-space contrast ("hybrid"), for robomimic.

Adds a CaMI-style action branch on top of BC_CaMI_CaNCE WITHOUT modifying it
(this file subclasses it and registers a new algo name, "bc_cami_cance_act").

--------------------------------------------------------------------------
Why
--------------------------------------------------------------------------
C-CaNCE's physics branch supervises the observation encoder with a single
bounded scalar (the gap phi_hat). Its loss is solved by ~epoch 300 (phi_hat
saturates at {0, phi_max}) and then stops shaping the representation, while
BC keeps overfitting. The action branch gives the policy latent a second,
richer and non-saturating target: identify the demonstrated FUTURE ACTIONS
from the current policy state. Both branches are training-only (force and
future actions are never inputs at inference).

--------------------------------------------------------------------------
Notation (anchor k, candidate l; one anchor = one sequence at t = 0)
--------------------------------------------------------------------------
    h_k      policy latent at the anchor (post-RNN per-step feature, t = 0)
    tau_k    demonstrated future action snippet (a_1 .. a_H), H = snippet_horizon
    z_k      = norm( psi(h_k) )                         state query   (grad -> policy)
    q_k      = norm( g(LSTM(tau_k)) )                   action query  (grad -> action enc.)
    y_k      = sg[ norm( g*(LSTM*(tau_k)) ) ]           target key    (EMA copy, no grad)
    S^(b)_kl = u_k^T y_l / beta_a,  u = z (b = state) or q (b = traj)

    l_k^(b)  = -log  exp(S_kk) / ( exp(S_kk) + sum_{l in N_k} exp(S_kl) )

ADDITIVE mode (joint_score = False; the default hybrid):
    L = L_BC + w_nce L_CaNCE + w_pen L_pen + w_mag L_mag + w_viol L_viol
            + lambda_s L_state + lambda_t L_traj
    The physics branch is unchanged (all B(T-1) pairs, same loss as BC_CaMI_CaNCE).

JOINT mode (joint_score = True; ablation): one softmax whose score is the sum
of the physics score and the action score, over the same candidates,
    s_kl = - lambda_hat_k^T phi_hat_l / beta  +  z_k^T y_l / beta_a
    L_joint = mean_k  -log  exp(s_kk) / ( exp(s_kk) + sum_{l in N_k} exp(s_kl) )
    L = L_BC + w_pen L_pen + w_mag L_mag + w_viol L_viol + w_joint L_joint + lambda_t L_traj
    A negative must now be distinguishable by EITHER the energy or the action
    similarity (it removes the "contact-looking successor" false negatives that
    floor the physics-only loss). Anchors are the B sequences at t = 0 (physics
    pair (0 -> 1)), so the 1000-pair harvest is not used by the contrastive term
    (L_pen / L_mag / L_viol still use all pairs). All anchors are used (free-space
    anchors still get signal from the action term). The untrained all-pairs
    physics NCE is still logged as NCE_Loss (diagnostic only).

Negatives (both branches): other sequences only (one anchor per sequence, so
every l != k) and, in "regime" mode, the opposite contact regime
(contact_definition: "window" = any contact label in t = 1..H, "anchor" = label at t = 0).

Gradient routing:  L_state -> policy + state_proj;  L_traj -> action_encoder only;
physics -> obs encoder via phi_hat, impulse_encoder via lambda_hat.
The target encoder follows the online one by EMA after each optimizer step.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc_cami_cance import BC_CaMI_CaNCE, build_mlp


@register_algo_factory_func("bc_cami_cance_act")
def algo_config_to_class(algo_config):
    return BC_CaMI_CaNCE_Act, {}


class ActionSnippetEncoder(nn.Module):
    """LSTM over the future action snippet [B, H, da] -> projection -> [B, d] (unnormalized)."""

    def __init__(self, action_dim, hidden_dim, num_layers, proj_hidden_dims, out_dim):
        super(ActionSnippetEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=action_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.proj = build_mlp(hidden_dim, list(proj_hidden_dims), out_dim)

    def forward(self, snippet):
        _, (h_n, _) = self.lstm(snippet)
        return self.proj(h_n[-1])


def _per_anchor_nce(scores, neg_mask):
    """scores [N, N] (row = anchor, col = candidate, diagonal = positive) -> per-anchor InfoNCE loss [N]."""
    pos = scores.diagonal()
    neg = scores.masked_fill(~neg_mask, float("-inf"))
    log_denom = torch.logsumexp(torch.cat([pos.unsqueeze(1), neg], dim=1), dim=1)
    return log_denom - pos


def _masked_mean(x, valid, ref):
    if bool(valid.any()):
        return x[valid].mean()
    return ref.sum() * 0.0


@torch.no_grad()
def _retrieval(scores, neg_mask, valid):
    pos = scores.diagonal()
    max_neg = scores.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
    win = (pos > max_neg).float()
    if bool(valid.any()):
        return win[valid].mean()
    return torch.zeros((), device=scores.device)


class BC_CaMI_CaNCE_Act(BC_CaMI_CaNCE):

    # ------------------------------------------------------------------ networks
    def _create_networks(self):
        super(BC_CaMI_CaNCE_Act, self)._create_networks()

        a = self.algo_config.cami.action
        self._act_enabled = bool(a.enabled)
        self._act_joint = bool(a.joint_score)
        if self._act_joint and not self._act_enabled:
            raise ValueError("algo.cami.action.joint_score=True requires algo.cami.action.enabled=True")

        self._act_H = int(a.snippet_horizon)
        self._act_beta = float(a.temperature)
        self._act_m = float(a.momentum)
        self._act_w_state = float(a.loss_weight_state)
        self._act_w_traj = float(a.loss_weight_traj)
        self._act_w_joint = float(a.loss_weight_joint)
        self._act_joint_phys_aux = float(a.joint_phys_aux_weight)
        self._policy_latent_dim = int(a.policy_latent_dim)

        if a.negative_mode not in ("marginal", "regime"):
            raise ValueError("algo.cami.action.negative_mode must be 'marginal' or 'regime'")
        if a.contact_definition not in ("window", "anchor"):
            raise ValueError("algo.cami.action.contact_definition must be 'window' or 'anchor'")
        self._act_neg_mode = a.negative_mode
        self._act_contact_def = a.contact_definition

        d = int(a.contrastive_dim)
        # Always built (even when disabled) so every optim_params entry has a net.
        self.nets["state_proj"] = build_mlp(
            input_dim=self._policy_latent_dim,
            hidden_dims=list(a.state_proj_hidden_dims),
            output_dim=d,
        )
        self.nets["action_encoder"] = ActionSnippetEncoder(
            action_dim=self.ac_dim,
            hidden_dim=int(a.snippet_hidden_dim),
            num_layers=int(a.snippet_num_layers),
            proj_hidden_dims=list(a.key_proj_hidden_dims),
            out_dim=d,
        )
        target = copy.deepcopy(self.nets["action_encoder"])
        for p in target.parameters():
            p.requires_grad_(False)
        self.nets["action_encoder_target"] = target  # EMA copy: no optimizer, saved in checkpoints

        self.nets = self.nets.float().to(self.device)
        self._last_policy_feats = None

    # Capture the post-RNN per-step latent that the base class already computes (and drops),
    # so the base class does not need to be edited.
    def _forward_policy_with_latent(self, *args, **kwargs):
        out = super(BC_CaMI_CaNCE_Act, self)._forward_policy_with_latent(*args, **kwargs)
        self._last_policy_feats = out[1]
        return out

    # ------------------------------------------------------------------ forward
    def _forward_training(self, batch):
        self._last_policy_feats = None
        predictions = super(BC_CaMI_CaNCE_Act, self)._forward_training(batch)
        if not self._act_enabled:
            return predictions

        feats = self._last_policy_feats
        self._last_policy_feats = None
        if feats is None:
            raise RuntimeError("policy latent was not captured; _forward_policy_with_latent was not called")
        B, T, D = feats.shape
        if D != self._policy_latent_dim:
            raise RuntimeError(
                "policy latent has dim {} but algo.cami.action.policy_latent_dim={}. "
                "Set policy_latent_dim={} in the config.".format(D, self._policy_latent_dim, D)
            )
        H = self._act_H
        if T < H + 1:
            raise RuntimeError(
                "action branch needs seq_length >= snippet_horizon + 1 = {} (got T={}).".format(H + 1, T)
            )

        h0 = feats[:, 0, :]                                  # [B, D]   anchor latent (t = 0)
        tau = batch["actions"][:, 1:H + 1, :]                # [B, H, da]  a_{t+1..t+H}
        z = F.normalize(self.nets["state_proj"](h0), dim=-1)
        q = F.normalize(self.nets["action_encoder"](tau), dim=-1)
        with torch.no_grad():
            y = F.normalize(self.nets["action_encoder_target"](tau), dim=-1)

        predictions["act_z"] = z
        predictions["act_q"] = q
        predictions["act_y"] = y

        if not hasattr(self, "_debug_printed_act"):
            self._debug_printed_act = True
            print("\n[BC_CaMI_CaNCE_Act DEBUG] action branch")
            print("  joint_score         :", self._act_joint)
            print("  policy latent       :", tuple(feats.shape))
            print("  snippet tau         :", tuple(tau.shape))
            print("  z / q / y           :", tuple(z.shape), tuple(q.shape), tuple(y.shape))
            print("  anchors per batch   :", B)
        return predictions

    # ------------------------------------------------------------------ negatives
    def _action_negative_mask(self, batch):
        """[B, B] bool: valid negatives l for anchor k (rows = anchors)."""
        z_ref = batch["actions"]
        B = z_ref.shape[0]
        eye = torch.eye(B, dtype=torch.bool, device=z_ref.device)
        mask = ~eye
        if self._act_neg_mode == "regime":
            cl = batch["contact_label"]
            if cl is None:
                raise KeyError("action negative_mode='regime' requires contact_label in the batch")
            H = self._act_H
            if self._act_contact_def == "window":
                c = cl[:, 1:H + 1].max(dim=1).values > 0.5
            else:
                c = cl[:, 0] > 0.5
            mask = mask & (c.unsqueeze(1) != c.unsqueeze(0))
        return mask

    # ------------------------------------------------------------------ losses
    def _compute_losses(self, predictions, batch):
        losses = super(BC_CaMI_CaNCE_Act, self)._compute_losses(predictions, batch)

        cami_enabled = self.algo_config.cami.enabled if "enabled" in self.algo_config.cami else False
        if not (self._act_enabled and cami_enabled):
            return losses

        lcp = self.algo_config.cami.lcp
        z, q, y = predictions["act_z"], predictions["act_q"], predictions["act_y"]
        beta_a = self._act_beta
        neg_mask = self._action_negative_mask(batch)
        valid = neg_mask.any(dim=1)
        n_neg = neg_mask.sum(dim=1).float()

        S_state = (z @ y.T) / beta_a
        S_traj = (q @ y.T) / beta_a

        loss_traj = _masked_mean(_per_anchor_nce(S_traj, neg_mask), valid, S_traj)
        losses["act_loss_traj"] = loss_traj
        losses["act_retr_traj"] = _retrieval(S_traj, neg_mask, valid)
        losses["act_valid_frac"] = valid.float().mean().detach()
        # dimensional-collapse monitors: healthy ~ 1/sqrt(d) (~0.09 for d=128); -> 0 means collapse
        losses["act_std_z"] = z.detach().std(dim=0).mean()
        losses["act_std_y"] = y.detach().std(dim=0).mean()
        losses["act_ceiling"] = (
            torch.log1p(n_neg[valid]).mean() if bool(valid.any()) else torch.zeros((), device=z.device)
        ).detach()

        # base terms without the contrastive physics loss; re-assembled below
        total = (
            losses["bc_action_loss"]
            + lcp.w_pen * losses["pen_loss"]
            + lcp.w_mag * losses["mag_loss"]
            + self._w_viol * losses["viol_loss"]
            + self._act_w_traj * loss_traj
        )

        if not self._act_joint:
            loss_state = _masked_mean(_per_anchor_nce(S_state, neg_mask), valid, S_state)
            losses["act_loss_state"] = loss_state
            losses["act_retr_state"] = _retrieval(S_state, neg_mask, valid)
            total = total + lcp.loss_weight * losses["nce_loss"] + self._act_w_state * loss_state
        else:
            # anchors k = (b, t=0): lambda_hat from force at t=0, phi_hat at t=1 (same pairing as the harvest)
            lam0 = predictions["lambda_hat_all"][:, 0, :]
            phi1 = predictions["phi_hat_all"][:, 1, :]
            S_phys = -(lam0 @ phi1.T) / lcp.beta
            S_joint = S_phys + S_state
            loss_joint = _masked_mean(_per_anchor_nce(S_joint, neg_mask), valid, S_joint)
            losses["joint_loss"] = loss_joint
            total = total + self._act_w_joint * loss_joint
            # standalone physics NCE keeps phi_hat / lambda_hat trained on their own. Without it the
            # joint softmax lets the (easier) action score carry the loss and w_viol drives phi_hat -> 0.
            if self._act_joint_phys_aux > 0.0:
                total = total + self._act_joint_phys_aux * losses["nce_loss"]

            with torch.no_grad():
                losses["joint_retr"] = _retrieval(S_joint, neg_mask, valid)
                losses["joint_retr_phys_only"] = _retrieval(S_phys, neg_mask, valid)
                losses["joint_retr_act_only"] = _retrieval(S_state, neg_mask, valid)
                zero = torch.zeros((), device=z.device)
                losses["joint_pos_phys"] = S_phys.diagonal()[valid].mean() if bool(valid.any()) else zero
                losses["joint_neg_phys"] = S_phys[neg_mask].mean() if bool(neg_mask.any()) else zero
                losses["joint_pos_act"] = S_state.diagonal()[valid].mean() if bool(valid.any()) else zero
                losses["joint_neg_act"] = S_state[neg_mask].mean() if bool(neg_mask.any()) else zero

        losses["action_loss"] = total
        return losses

    # ------------------------------------------------------------------ optimisation
    @torch.no_grad()
    def _momentum_update(self):
        m = self._act_m
        for p_t, p in zip(
            self.nets["action_encoder_target"].parameters(),
            self.nets["action_encoder"].parameters(),
        ):
            p_t.data.mul_(m).add_(p.data, alpha=1.0 - m)

    def _train_step(self, losses):
        required = ["policy", "gap_encoder", "impulse_encoder"]
        if self._act_enabled:
            required += ["state_proj", "action_encoder"]
        missing = [k for k in required if k not in self.optimizers]
        if len(missing) > 0:
            raise KeyError(
                "Missing optimizer entries for {}. Add them under algo.optim_params "
                "in your config.".format(missing)
            )

        info = OrderedDict()
        for name in required:
            self.optimizers[name].zero_grad()

        losses["action_loss"].backward()

        max_grad_norm = self.global_config.train.max_grad_norm
        for name in required:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.nets[name].parameters(),
                max_grad_norm if max_grad_norm is not None else 1e9,
            )
            info["{}_grad_norm".format(name)] = float(grad_norm)

        for name in required:
            self.optimizers[name].step()

        if self._act_enabled:
            self._momentum_update()
        return info

    # ------------------------------------------------------------------ logging
    def log_info(self, info):
        log = super(BC_CaMI_CaNCE_Act, self).log_info(info)
        losses = info["losses"]
        for key, val in losses.items():
            if key.startswith("act_") or key.startswith("joint_"):
                log[key] = val.item()
        for key in ["state_proj_grad_norm", "action_encoder_grad_norm"]:
            if key in info:
                log[key] = info[key]
        return log