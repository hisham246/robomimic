"""
ManiFlow Policy + CaMI (C-CaNCE physics branch + action-space contrast), for robomimic.

Subclasses ManiFlowPolicy (robomimic/algo/maniflow_policy.py) WITHOUT modifying it and
registers a new algo name, "maniflow_cami". Inference (get_action / _get_action_trajectory)
is inherited unchanged: force and contact labels are used only in the training loss.

--------------------------------------------------------------------------
Where CaMI attaches
--------------------------------------------------------------------------
ManiFlow encodes each observation frame with obs_encoder -> e_t in R^D and conditions the
DiT-X on the concatenation of the first To frames. CaMI supervises the same encoder:

    physics branch  e_j -> gap head -> phi_hat_j      (frames j = 0..K-1 of the window)
                    force_j -> impulse head -> lambda_hat_j
                    anchors (b, j), positive phi_hat_{j+1}, energy E_kl = lambda_hat_k^T phi_hat_l
                    -> InfoNCE over regime negatives (same loss as BC_CaMI_CaNCE)
    action branch   z = norm(psi(obs_cond))  vs  a key built from the future snippet
                    (see "Action-branch options" below)

The policy only sees frames 0..To-1. The physics branch encodes K >= To frames (default 6) in
the SAME encoder pass, so the first To features are reused for the policy and the extra frames
cost one encoder pass each. K controls the pair pool: N = B (K - 1) anchors per batch.

Total objective (all CaMI terms switchable):
    L = L_FM + L_CT
        + w_nce L_CaNCE + w_pen L_pen + w_mag L_mag + w_viol L_viol        (physics branch)
        + lambda_s L_state + lambda_t L_traj                                (action branch)

Gradient routing: physics -> obs encoder (via phi_hat) and impulse head (via lambda_hat);
L_state -> obs encoder + state_proj; L_traj -> action_encoder only. The contrastive terms never
see noisy actions or the flow time t.

--------------------------------------------------------------------------
Action-branch options (algo.cami.action.*; the defaults reproduce the v1 behaviour, so
checkpoints and configs from v1 runs load and behave unchanged)
--------------------------------------------------------------------------
key_inputs     "actions"        key encoder sees a_k                                   (v1)
               "actions_force"  key encoder sees [a_k, |f_k| scaled, contact_k] per step:
                                the key must encode how the upcoming interaction feels,
                                which the flow loss never sees. Force stays training-only.
loss           "ema"            L_state (z vs EMA key y) + L_traj (online key q vs y)   (v1)
               "symmetric"      CLIP-style  L_sym = 1/2 [NCE(z -> q) + NCE(q -> z)],
                                both encoders trained, no EMA target, no L_traj.
neg_weighting  "none"           every valid negative counts fully                      (v1)
               "force_profile"  Neil's soft weights: negative l of anchor k is scaled by
                                W_kl = 1 - exp(-D_kl / tau_c), D_kl = Huber distance between
                                the future force-magnitude profiles of k and l (no gradient).
                                Look-alike futures (likely false negatives) count less.
                                tau_c = neg_weight_tau * median(D over valid negatives) in the
                                batch, so it adapts to the force scale.
snippet_align  "next"           snippet = a_{t+1..t+H}, H = snippet_horizon             (v1)
               "executed"       snippet = the chunk ManiFlow executes, a_{t..t+Ta-1}
                                (Ta = action_horizon; snippet_horizon is ignored)

--------------------------------------------------------------------------
EMA
--------------------------------------------------------------------------
ManiFlow's EMA is used for inference and for the consistency target. Here it is rebuilt over
the policy sub-network only, so the CaMI heads (and the action-branch target encoder, which has
its own momentum update) are never averaged into it. Checkpoints written by this class load
back into this class; they are not interchangeable with plain ManiFlowPolicy checkpoints.

--------------------------------------------------------------------------
Time alignment
--------------------------------------------------------------------------
Obs arrive as [B, T_obs, ...] and force / contact_label / actions as [B, T_ds, ...].
offset = T_obs - T_ds (0 if dataset keys are frame-stacked like obs, To - 1 if not).
Obs frame j pairs with force index j - offset. The action chunk follows the parent's
convention (actions[:, :Tp], current step at index To - 1).
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from diffusers.training_utils import EMAModel

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.maniflow_policy import ManiFlowPolicy


@register_algo_factory_func("maniflow_cami_v2")
def algo_config_to_class(algo_config):
    if algo_config.transformer.enabled:
        return ManiFlowCaMIV2, {}
    raise RuntimeError("maniflow_cami requires algo.transformer.enabled = True")


# ---------------------------------------------------------------------------- helpers
def build_mlp(input_dim, hidden_dims, output_dim):
    layers, d = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)


class ActionSnippetEncoder(nn.Module):
    """LSTM over the future action snippet [B, H, da] -> projection -> [B, d] (unnormalized)."""

    def __init__(self, action_dim, hidden_dim, num_layers, proj_hidden_dims, out_dim):
        super(ActionSnippetEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=action_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.proj = build_mlp(hidden_dim, list(proj_hidden_dims), out_dim)

    def forward(self, snippet):
        _, (h_n, _) = self.lstm(snippet)
        return self.proj(h_n[-1])


def _per_anchor_nce(logits, neg_mask, log_w=None):
    """
    logits [N, N] (row = anchor, diagonal = positive) -> per-anchor InfoNCE loss [N].
    log_w [N, N] (optional): log of per-negative weights, i.e. the denominator uses
    sum_l W_kl exp(s_kl) instead of sum_l exp(s_kl).
    """
    pos = logits.diagonal()
    neg = logits if log_w is None else logits + log_w
    neg = neg.masked_fill(~neg_mask, float("-inf"))
    log_denom = torch.logsumexp(torch.cat([pos.unsqueeze(1), neg], dim=1), dim=1)
    return log_denom - pos


def _masked_mean(x, mask):
    if bool(mask.any()):
        return x[mask].mean()
    return x.sum() * 0.0


@torch.no_grad()
def _profile_weights(profile, neg_mask, tau_factor, eps=1e-6):
    """
    Neil's soft negative weights from future force profiles.
    profile [B, H] scaled force magnitudes -> (log W [B, B], mean W over valid negatives).
    D_kl = mean_h Huber(profile_k,h - profile_l,h);  W_kl = 1 - exp(-D_kl / tau_c),
    tau_c = tau_factor * median(D over valid negatives).
    """
    diff = profile.unsqueeze(1) - profile.unsqueeze(0)                    # [B, B, H]
    D = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=1.0).mean(-1)
    if bool(neg_mask.any()):
        tau = tau_factor * D[neg_mask].median().clamp_min(eps)
    else:
        tau = torch.ones((), device=profile.device)
    W = 1.0 - torch.exp(-D / tau)
    w_mean = W[neg_mask].mean() if bool(neg_mask.any()) else torch.zeros((), device=profile.device)
    return torch.log(W.clamp_min(eps)), w_mean


@torch.no_grad()
def _retrieval(logits, neg_mask, mask):
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device)
    pos = logits.diagonal()
    max_neg = logits.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
    return (pos > max_neg).float()[mask].mean()


# ---------------------------------------------------------------------------- algo
class ManiFlowCaMIV2(ManiFlowPolicy):

    # ------------------------------------------------------------------ networks
    def _create_networks(self):
        super(ManiFlowCaMIV2, self)._create_networks()

        cami = self.algo_config.cami
        lcp = cami.lcp
        act = cami.action
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        self._cami_on = bool(cami.enabled)
        self._phys_on = self._cami_on and bool(lcp.enabled)
        self._act_on = self._cami_on and bool(act.enabled)

        # physics branch settings
        self._K = int(lcp.num_frames)
        if self._phys_on and self._K < 2:
            raise ValueError("algo.cami.lcp.num_frames must be >= 2 (consecutive pairs)")
        if lcp.negative_mode not in ("marginal", "regime"):
            raise ValueError("algo.cami.lcp.negative_mode must be 'marginal' or 'regime'")
        if lcp.anchor_mode not in ("contact", "all"):
            raise ValueError("algo.cami.lcp.anchor_mode must be 'contact' or 'all'")
        if int(lcp.force_dim) != 1:
            raise ValueError("algo.cami.lcp.force_dim must be 1 (impulse head sees the scaled force magnitude)")
        self._force_dim = 1
        self._phi_max = float(lcp.phi_max) if lcp.phi_max is not None else 0.0

        # action branch settings
        Ta = self.algo_config.horizon.action_horizon
        for key, allowed in (("key_inputs", ("actions", "actions_force")),
                             ("loss", ("ema", "symmetric")),
                             ("neg_weighting", ("none", "force_profile")),
                             ("snippet_align", ("next", "executed"))):
            if getattr(act, key) not in allowed:
                raise ValueError("algo.cami.action.{} must be one of {}".format(key, allowed))
        self._key_force = (act.key_inputs == "actions_force")
        self._act_sym = (act.loss == "symmetric")
        self._neg_w = (act.neg_weighting == "force_profile")
        if act.snippet_align == "executed":
            self._snip = (To - 1, To - 1 + Ta)                 # a_t .. a_{t+Ta-1}: the executed chunk
        else:
            self._snip = (To, To + int(act.snippet_horizon))   # a_{t+1} .. a_{t+H}
        self._H = self._snip[1] - self._snip[0]
        if self._act_on and self._snip[1] > Tp:
            raise ValueError("action snippet ends at index {} but prediction_horizon is {}".format(
                self._snip[1], Tp))
        if act.negative_mode not in ("marginal", "regime"):
            raise ValueError("algo.cami.action.negative_mode must be 'marginal' or 'regime'")
        if act.contact_definition not in ("window", "anchor"):
            raise ValueError("algo.cami.action.contact_definition must be 'window' or 'anchor'")

        obs_dim = self.nets["policy"]["obs_encoder"].output_shape()[0]
        d = int(act.contrastive_dim)

        # Heads are always built (even when disabled) so every optim_params entry has a net.
        self.nets["gap_encoder"] = build_mlp(obs_dim, list(lcp.gap_hidden_dims), int(lcp.num_contacts))
        self.nets["impulse_encoder"] = build_mlp(self._force_dim, list(lcp.impulse_hidden_dims),
                                                 int(lcp.num_contacts))
        self.nets["state_proj"] = build_mlp(obs_dim * To, list(act.state_proj_hidden_dims), d)
        # key encoder input per step: action (+ scaled force magnitude + contact label)
        key_in_dim = self.ac_dim + (2 if self._key_force else 0)
        self.nets["action_encoder"] = ActionSnippetEncoder(
            action_dim=key_in_dim, hidden_dim=int(act.snippet_hidden_dim),
            num_layers=int(act.snippet_num_layers), proj_hidden_dims=list(act.key_proj_hidden_dims),
            out_dim=d)
        target = copy.deepcopy(self.nets["action_encoder"])
        for p in target.parameters():
            p.requires_grad_(False)
        self.nets["action_encoder_target"] = target  # EMA copy (loss="ema" only): no optimizer, saved in checkpoints
        self.nets = self.nets.float().to(self.device)

        # Rebuild ManiFlow's EMA over the policy only (shares modules with self.nets; the EMA
        # itself holds a deep copy). averaged_model["policy"][...] keeps the parent's indexing.
        self._ema_src = nn.ModuleDict({"policy": self.nets["policy"]})
        self.ema = None
        if self.algo_config.ema.enabled:
            self.ema = EMAModel(model=self._ema_src, power=self.algo_config.ema.power)

    def _cami_optimizer_names(self):
        names = ["policy"]
        if self._phys_on:
            names += ["gap_encoder", "impulse_encoder"]
        if self._act_on:
            names += ["state_proj", "action_encoder"]
        missing = [k for k in names if k not in self.optimizers]
        if missing:
            raise KeyError("Missing optimizer entries {} under algo.optim_params".format(missing))
        return names

    # ------------------------------------------------------------------ batch
    def _num_frames_needed(self, T_obs, T_ds):
        To = self.algo_config.horizon.observation_horizon
        offset = T_obs - T_ds
        if not self._phys_on:
            return To, offset
        if offset < 0:
            raise RuntimeError("obs window ({}) shorter than force window ({})".format(T_obs, T_ds))
        need = max(To, offset + self._K)
        if need > T_obs or self._K > T_ds:
            raise RuntimeError(
                "physics branch needs {} obs frames and {} force frames but the window has {} / {}. "
                "Lower algo.cami.lcp.num_frames.".format(need, self._K, T_obs, T_ds))
        return need, offset

    def process_batch_for_training(self, batch):
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        force = None
        if "force" in batch:
            force = batch["force"]
        elif "force" in batch["obs"]:
            force = batch["obs"]["force"]
        contact = None
        if "contact_label" in batch:
            contact = batch["contact_label"]
        elif "contact_label" in batch["obs"]:
            contact = batch["obs"]["contact_label"]

        obs = {k: batch["obs"][k] for k in batch["obs"] if k not in ("force", "contact_label")}
        T_obs = next(iter(obs.values())).shape[1]
        T_ds = force.shape[1] if force is not None else batch["actions"].shape[1]
        n_frames, offset = self._num_frames_needed(T_obs, T_ds)

        need_force = self._phys_on or (self._act_on and (self._key_force or self._neg_w))
        need_contact = (self._phys_on and self.algo_config.cami.lcp.negative_mode == "regime") or \
                       (self._phys_on and self.algo_config.cami.lcp.anchor_mode == "contact") or \
                       (self._act_on and self.algo_config.cami.action.negative_mode == "regime") or \
                       (self._act_on and self._key_force)
        if need_force and force is None:
            raise KeyError("CaMI needs 'force' in train.dataset_keys (physics branch / force-aware action key)")
        if need_contact and contact is None:
            raise KeyError("CaMI needs 'contact_label' in train.dataset_keys (regime negatives / contact anchors)")
        if contact is not None and contact.ndim == 3:
            contact = contact.squeeze(-1)
        if contact is not None and contact.shape[1] == 1 and T_ds > 1:
            # one label per sequence (discrete CaMI pipeline): broadcast, as BC_CaMI_CaNCE does
            print("[ManiFlowCaMI WARNING] contact_label has 1 timestep; broadcasting across the window")
            contact = contact.expand(-1, T_ds)

        input_batch = dict()
        input_batch["obs"] = {k: v[:, :n_frames] for k, v in obs.items()}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        input_batch["force"] = force
        input_batch["contact_label"] = contact
        # kept on the algo (not in the batch) so robomimic's batch post-processing only sees tensors
        self._frame_offset = offset

        if not self.action_check_done:
            a = input_batch["actions"]
            if not bool(torch.all((-1 <= a) & (a <= 1)).item()):
                raise ValueError("'actions' must be in range [-1,1] for ManiFlow! "
                                 "Check the action normalization in the config.")
            self.action_check_done = True

        if not hasattr(self, "_dbg_batch"):
            self._dbg_batch = True
            print("\n[ManiFlowCaMI DEBUG] process_batch_for_training")
            print("  obs T (raw / used)  :", T_obs, "/", n_frames, " offset:", offset)
            print("  actions (raw / used):", tuple(batch["actions"].shape), "/", tuple(input_batch["actions"].shape))
            print("  force               :", None if force is None else tuple(force.shape))
            print("  contact_label       :", None if contact is None else tuple(contact.shape))
            print("  physics / action    :", self._phys_on, "/", self._act_on)

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    # ------------------------------------------------------------------ ManiFlow losses (copied from parent)
    def _flow_and_consistency_losses(self, actions, obs_cond):
        mf = self.algo_config.maniflow
        B = actions.shape[0]
        n_flow = int(B * mf.flow_batch_ratio)
        n_ct = B - n_flow
        net = self.nets["policy"]["noise_pred_net"]
        relative = (mf.target_t_mode == "relative")

        x1 = actions[:n_flow]
        x0 = torch.randn_like(x1)
        t = self._sample_t(n_flow, mf.sample_t_mode_flow)
        xt = (1 - t) * x0 + t * x1
        dt = torch.zeros(n_flow, device=self.device)
        target_t = dt if relative else t.view(-1) + dt
        v_pred = net(xt, t.view(-1), target_t, global_cond=obs_cond[:n_flow])
        loss_flow = F.mse_loss(v_pred, x1 - x0)

        loss_ct = torch.zeros((), device=self.device)
        if n_ct > 0:
            x1 = actions[n_flow:]
            x0 = torch.randn_like(x1)
            t = self._sample_t(n_ct, mf.sample_t_mode_consistency)
            dt = torch.rand(n_ct, device=self.device)
            t_next = torch.clamp(t + dt.view(-1, 1, 1), max=1.0)
            xt = (1 - t) * x0 + t * x1
            x_t_next = (1 - t_next) * x0 + t_next * x1
            target_t_next = dt if relative else t_next.view(-1) + dt
            ema_net = self.ema.averaged_model["policy"]["noise_pred_net"]
            with torch.no_grad():
                v_avg = ema_net(x_t_next, t_next.view(-1), target_t_next,
                                global_cond=obs_cond[n_flow:].detach())
            x1_hat = x_t_next + (1 - t_next) * v_avg
            v_target = (x1_hat - xt) / (1 - t)
            target_t = dt if relative else t_next.view(-1)
            v_pred = net(xt, t.view(-1), target_t, global_cond=obs_cond[n_flow:])
            loss_ct = F.mse_loss(v_pred, v_target)
        return loss_flow, loss_ct

    # ------------------------------------------------------------------ physics branch
    def _impulse_input(self, force):
        """[B, T, D] raw wrench -> [B, T, 1] scaled, clipped force magnitude (as in BC_CaMI_CaNCE)."""
        lcp = self.algo_config.cami.lcp
        f = force.float()[..., :3].norm(dim=-1, keepdim=True) / lcp.force_scale
        return f.clamp(max=lcp.force_clip)

    def _physics_losses(self, feats, batch):
        lcp = self.algo_config.cami.lcp
        K, off = self._K, self._frame_offset
        e = feats[:, off:off + K]                                   # [B, K, D]
        B = e.shape[0]

        raw_phi = TensorUtils.time_distributed(e, self.nets["gap_encoder"])          # [B, K, p]
        phi = self._phi_max * torch.sigmoid(raw_phi) if self._phi_max > 0 else raw_phi
        f_in = self._impulse_input(batch["force"][:, :K])          # [B, K, 1]
        lam = F.softplus(TensorUtils.time_distributed(f_in, self.nets["impulse_encoder"]))  # [B, K, p]

        # pair pool: anchor (b, j), j = 0..K-2 ; lambda at j, positive phi at j + 1
        p = phi.shape[-1]
        lam_a = lam[:, :-1].reshape(-1, p)
        phi_p = phi[:, 1:].reshape(-1, p)
        fmag_a = f_in[:, :-1].reshape(-1)          # mag target = |reduced force| (1-d), as in the BC class
        seq_id = torch.arange(B, device=e.device).unsqueeze(1).expand(B, K - 1).reshape(-1)
        cl = batch["contact_label"]
        cl_a = cl[:, :K][:, :-1].reshape(-1) if cl is not None else None     # raw labels, as in the BC class
        contact_a = cl_a.bool() if cl is not None else None

        E = lam_a @ phi_p.T                                         # [N, N]
        logits = -E / lcp.beta
        neg_mask = seq_id.unsqueeze(1) != seq_id.unsqueeze(0)
        if lcp.negative_mode == "regime":
            neg_mask = neg_mask & (cl_a.unsqueeze(1) != cl_a.unsqueeze(0))
        has_neg = neg_mask.any(dim=1)
        if lcp.anchor_mode == "contact":
            anchor = has_neg & contact_a
        else:
            anchor = has_neg

        per_anchor = _per_anchor_nce(logits, neg_mask)
        out = OrderedDict()
        out["nce_loss"] = _masked_mean(per_anchor, anchor)
        out["pen_loss"] = torch.clamp(phi_p, max=0.0).pow(2).sum(-1).mean()
        out["mag_loss"] = F.mse_loss(lam_a.sum(-1), fmag_a)
        E_kk = E.diagonal()
        if lcp.anchor_mode == "contact":
            out["viol_loss"] = _masked_mean(E_kk, contact_a)
        else:
            out["viol_loss"] = E_kk.mean()

        with torch.no_grad():
            n_neg = neg_mask.sum(dim=1).float()
            zero = torch.zeros((), device=e.device)
            out["retrieval_acc"] = _retrieval(logits, neg_mask, anchor)
            out["avg_valid_negatives"] = n_neg[anchor].mean() if bool(anchor.any()) else zero
            out["collapse_ceiling"] = torch.log1p(n_neg[anchor]).mean() if bool(anchor.any()) else zero
            out["valid_anchor_fraction"] = anchor.float().mean()
            out["pos_logit_mean"] = logits.diagonal()[anchor].mean() if bool(anchor.any()) else zero
            out["neg_logit_mean"] = logits[neg_mask].mean() if bool(neg_mask.any()) else zero
            if contact_a is not None:
                # per-regime diagnostics over every anchor that has negatives, whatever anchor_mode is
                cm, fm = has_neg & contact_a, has_neg & ~contact_a
                out["contact_anchor_frac"] = _masked_mean(contact_a.float(), has_neg)
                out["nce_loss_contact"] = _masked_mean(per_anchor, cm)
                out["nce_loss_free"] = _masked_mean(per_anchor, fm)
                out["retrieval_contact"] = _retrieval(logits, neg_mask, cm)
                out["retrieval_free"] = _retrieval(logits, neg_mask, fm)
                out["phi_contact"] = _masked_mean(phi_p.mean(-1), cm)
                out["phi_free"] = _masked_mean(phi_p.mean(-1), fm)
                out["lambda_contact"] = _masked_mean(lam_a.sum(-1), cm)
                out["lambda_free"] = _masked_mean(lam_a.sum(-1), fm)

        if not hasattr(self, "_dbg_phys"):
            self._dbg_phys = True
            print("\n[ManiFlowCaMI DEBUG] physics branch")
            print("  frames K / offset   :", K, "/", off, "  feats:", tuple(feats.shape))
            print("  phi / lambda        :", tuple(phi.shape), tuple(lam.shape))
            print("  pairs per batch (N) :", E.shape[0])
            print("  negatives min/mean/max:", int(n_neg.min()), float(n_neg.mean()), int(n_neg.max()))
            print("  anchor fraction     :", float(anchor.float().mean()))
        return out

    # ------------------------------------------------------------------ action branch
    def _action_losses(self, obs_cond, batch):
        a = self.algo_config.cami.action
        To = self.algo_config.horizon.observation_horizon
        s0, s1 = self._snip
        actions = batch["actions"]
        B = actions.shape[0]

        # ---- snippet (and, for the contact-aware key, the matching force / contact profile)
        tau = actions[:, s0:s1]                                   # [B, H, da]
        f_prof = None
        if self._key_force or self._neg_w:
            # same scaled, clipped |f_xyz| as the impulse head; dataset keys share the actions' time axis
            f_prof = self._impulse_input(batch["force"][:, s0:s1])                # [B, H, 1]
        key_in = tau
        if self._key_force:
            c_prof = (batch["contact_label"][:, s0:s1] > 0.5).float().unsqueeze(-1)  # [B, H, 1]
            key_in = torch.cat([tau, f_prof, c_prof], dim=-1)                       # [B, H, da + 2]

        z = F.normalize(self.nets["state_proj"](obs_cond.flatten(start_dim=1)), dim=-1)
        q = F.normalize(self.nets["action_encoder"](key_in), dim=-1)

        # ---- negatives
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=actions.device)
        if a.negative_mode == "regime":
            cl = batch["contact_label"]
            if a.contact_definition == "window":
                c = cl[:, s0:s1].max(dim=1).values > 0.5
            else:
                c = cl[:, To - 1] > 0.5
            neg_mask = neg_mask & (c.unsqueeze(1) != c.unsqueeze(0))
        valid = neg_mask.any(dim=1)

        # ---- Neil's soft weights from future force profiles (symmetric in k, l; no gradient)
        log_w, w_mean = None, None
        if self._neg_w:
            log_w, w_mean = _profile_weights(f_prof.squeeze(-1), neg_mask, float(a.neg_weight_tau))

        out = OrderedDict()
        temp = a.temperature
        if self._act_sym:
            # CLIP-style: both directions, both encoders trained, no EMA target
            S = (z @ q.T) / temp
            l_zq = _masked_mean(_per_anchor_nce(S, neg_mask, log_w), valid)
            l_qz = _masked_mean(_per_anchor_nce(S.T, neg_mask.T,
                                                None if log_w is None else log_w.T), valid)
            out["act_loss_sym"] = 0.5 * (l_zq + l_qz)
            with torch.no_grad():
                out["act_retr_state"] = _retrieval(S, neg_mask, valid)      # z -> q
                out["act_retr_key"] = _retrieval(S.T, neg_mask.T, valid)    # q -> z
                out["act_std_q"] = q.std(dim=0).mean()
        else:
            with torch.no_grad():
                y = F.normalize(self.nets["action_encoder_target"](key_in), dim=-1)
            S_state = (z @ y.T) / temp
            S_traj = (q @ y.T) / temp
            out["act_loss_state"] = _masked_mean(_per_anchor_nce(S_state, neg_mask, log_w), valid)
            out["act_loss_traj"] = _masked_mean(_per_anchor_nce(S_traj, neg_mask, log_w), valid)
            with torch.no_grad():
                out["act_retr_state"] = _retrieval(S_state, neg_mask, valid)
                out["act_retr_traj"] = _retrieval(S_traj, neg_mask, valid)
                out["act_std_y"] = y.std(dim=0).mean()

        with torch.no_grad():
            n_neg = neg_mask.sum(dim=1).float()
            out["act_valid_frac"] = valid.float().mean()
            out["act_ceiling"] = torch.log1p(n_neg[valid]).mean() if bool(valid.any()) \
                else torch.zeros((), device=actions.device)
            out["act_std_z"] = z.std(dim=0).mean()
            if w_mean is not None:
                out["act_neg_w_mean"] = w_mean      # 1 = no down-weighting; lower = more look-alike negatives

        if not hasattr(self, "_dbg_act"):
            self._dbg_act = True
            print("\n[ManiFlowCaMI DEBUG] action branch")
            print("  key_inputs / loss   :", a.key_inputs, "/", a.loss)
            print("  neg_weighting       :", a.neg_weighting, "  snippet_align:", a.snippet_align,
                  "  indices [{}, {})".format(s0, s1))
            print("  obs_cond / key in   :", tuple(obs_cond.shape), tuple(key_in.shape))
            print("  anchors per batch   :", B, "  mean negatives:", float(n_neg.mean()))
        return out

    # ------------------------------------------------------------------ training
    def train_on_batch(self, batch, epoch, validate=False):
        To = self.algo_config.horizon.observation_horizon
        lcp = self.algo_config.cami.lcp
        act = self.algo_config.cami.action

        with TorchUtils.maybe_no_grad(no_grad=validate):
            # skip ManiFlowPolicy.train_on_batch (re-implemented below), keep PolicyAlgo's bookkeeping
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)

            inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            feats = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"],
                                                 inputs_as_kwargs=True)   # [B, F, D]
            assert feats.ndim == 3
            obs_cond = feats[:, :To].flatten(start_dim=1).unsqueeze(1)   # [B, 1, To * D], as in the parent

            loss_flow, loss_ct = self._flow_and_consistency_losses(batch["actions"], obs_cond)
            policy_loss = loss_flow + loss_ct
            total = policy_loss
            losses = OrderedDict(loss_flow=loss_flow, loss_ct=loss_ct, policy_loss=policy_loss)

            if self._phys_on:
                phys = self._physics_losses(feats, batch)
                losses.update(phys)
                total = total + lcp.loss_weight * phys["nce_loss"] + lcp.w_pen * phys["pen_loss"] \
                    + lcp.w_mag * phys["mag_loss"] + lcp.w_viol * phys["viol_loss"]
            if self._act_on:
                acts = self._action_losses(obs_cond, batch)
                losses.update(acts)
                if self._act_sym:
                    total = total + act.loss_weight_sym * acts["act_loss_sym"]
                else:
                    total = total + act.loss_weight_state * acts["act_loss_state"] \
                        + act.loss_weight_traj * acts["act_loss_traj"]

            losses["l2_loss"] = total   # parent's log_info reports this as "Loss"
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                info.update(self._step(total))
        return info

    def _step(self, total):
        names = self._cami_optimizer_names()
        for n in names:
            self.optimizers[n].zero_grad()
        total.backward()
        max_norm = self.global_config.train.max_grad_norm
        info = OrderedDict()
        for n in names:
            # clip_grad_norm_ returns the pre-clip norm (the parent logs robomimic's sum of squares)
            gn = torch.nn.utils.clip_grad_norm_(self.nets[n].parameters(),
                                                max_norm if max_norm is not None else 1e9)
            info["{}_grad_norm".format(n)] = float(gn)
        for n in names:
            self.optimizers[n].step()
        info["policy_grad_norms"] = info["policy_grad_norm"]

        if self.ema is not None:
            self.ema.step(self._ema_src)
        if self._act_on and not self._act_sym:
            m = float(self.algo_config.cami.action.momentum)
            with torch.no_grad():
                for p_t, p in zip(self.nets["action_encoder_target"].parameters(),
                                  self.nets["action_encoder"].parameters()):
                    p_t.data.mul_(m).add_(p.data, alpha=1.0 - m)
        return info

    # ------------------------------------------------------------------ logging
    _LOG_KEYS = [
        "nce_loss", "pen_loss", "mag_loss", "viol_loss",
        "retrieval_acc", "avg_valid_negatives", "collapse_ceiling", "valid_anchor_fraction",
        "pos_logit_mean", "neg_logit_mean", "contact_anchor_frac",
        "nce_loss_contact", "nce_loss_free", "retrieval_contact", "retrieval_free",
        "phi_contact", "phi_free", "lambda_contact", "lambda_free",
    ]
    _LOG_RENAME = {"nce_loss": "NCE_Loss", "pen_loss": "Pen_Loss", "mag_loss": "Mag_Loss",
                   "viol_loss": "Viol_Loss"}

    def log_info(self, info):
        log = super(ManiFlowCaMIV2, self).log_info(info)   # Loss (= total), Flow_Loss, Consistency_Loss, ...
        losses = info["losses"]
        log["Policy_Loss"] = losses["policy_loss"].item()
        for k in self._LOG_KEYS:
            if k in losses:
                log[self._LOG_RENAME.get(k, k)] = losses[k].item()
        for k, v in losses.items():
            if k.startswith("act_"):
                log[k] = v.item()
        for k, v in info.items():
            if k.endswith("_grad_norm"):
                log[k] = v
        return log