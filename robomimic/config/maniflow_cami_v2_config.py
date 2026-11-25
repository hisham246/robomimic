"""
Config for ManiFlow Policy + CaMI (see robomimic/algo/maniflow_cami.py).

Inherits every ManiFlow field from ManiFlowPolicyConfig and adds algo.cami.* plus optimizer
entries for the CaMI heads. ALGO_NAME must match register_algo_factory_func("maniflow_cami").

Defaults mirror the BC-RNN additive hybrid (bc_cami_cance_act) so the two are comparable.
"""

from copy import deepcopy

from robomimic.config.maniflow_policy_config import ManiFlowPolicyConfig


class ManiFlowCaMIV2Config(ManiFlowPolicyConfig):
    ALGO_NAME = "maniflow_cami_v2"

    def train_config(self):
        super(ManiFlowCaMIV2Config, self).train_config()
        # privileged training-only signals; never policy inputs
        self.train.dataset_keys = ("actions", "rewards", "dones", "contact_label", "force")

    def algo_config(self):
        super(ManiFlowCaMIV2Config, self).algo_config()

        # optimizers for the CaMI heads (same optimizer / schedule as the policy).
        # action_encoder_target has no optimizer: it follows action_encoder by momentum.
        for name in ("gap_encoder", "impulse_encoder", "state_proj", "action_encoder"):
            setattr(self.algo.optim_params, name, deepcopy(self.algo.optim_params.policy))

        c = self.algo.cami
        c.enabled = True                      # False -> plain ManiFlow (control run, same code path)

        # ---------------- physics branch (C-CaNCE, bounded) ----------------
        p = c.lcp
        p.enabled = True                      # False -> action branch only (ablation)
        p.num_frames = 6                      # K obs frames encoded for the pair pool: N = B (K - 1) anchors.
                                              # Policy still uses only the first observation_horizon frames.
        p.loss_weight = 0.01                  # w_nce
        p.beta = 0.1                          # softmax temperature
        p.w_pen = 100.0                       # non-penetration penalty (0 when phi_max > 0, kept for parity)
        p.w_mag = 1.0                         # anchors lambda_hat to the measured force magnitude
        p.w_viol = 0.1                        # mean E_kk on contact anchors (gauge fixing)
        p.phi_max = 1.0                       # phi_hat = phi_max * sigmoid(.) ; <= 0 -> unbounded
        p.force_scale = 10.0                  # |f_xyz| / force_scale, clipped at force_clip
        p.force_clip = 10.0
        p.force_dim = 1                       # must be 1: impulse head sees the scaled, clipped |f_xyz| (as in BC_CaMI_CaNCE)
        p.num_contacts = 1
        p.anchor_mode = "contact"             # "contact": NCE over contact anchors | "all"
        p.negative_mode = "regime"            # "regime": other sequence AND opposite regime | "marginal"
        p.gap_hidden_dims = (256, 256)
        p.impulse_hidden_dims = (128,)

        # ---------------- action branch (additive) ----------------
        a = c.action
        a.enabled = True                      # False -> physics branch only (ablation)
        a.loss_weight_state = 0.01            # lambda_s: obs_cond query (grad -> obs encoder + state_proj)
        a.loss_weight_traj = 1.0              # lambda_t: snippet query (grad -> action_encoder only)
        a.temperature = 0.07
        a.snippet_horizon = 10                # H: a_{t+1..t+H}; needs observation_horizon + H <= prediction_horizon
        a.contrastive_dim = 128
        a.state_proj_hidden_dims = (128,)     # input dim = observation_horizon * obs feature dim (automatic)
        a.snippet_hidden_dim = 256
        a.snippet_num_layers = 1
        a.key_proj_hidden_dims = (128,)
        a.momentum = 0.995
        a.negative_mode = "marginal"
        a.contact_definition = "window"       # "window": any contact in the snippet | "anchor": label at t

        # ---- v2 options. Defaults reproduce v1 exactly, so v1 checkpoints/configs load unchanged.
        a.key_inputs = "actions"              # "actions" | "actions_force": key sees [a_k, |f_k| scaled, contact_k]
        a.loss = "ema"                        # "ema": L_state + L_traj with EMA key (v1) | "symmetric": CLIP-style, no EMA
        a.loss_weight_sym = 0.01              # weight of L_sym when loss = "symmetric" (replaces loss_weight_state / _traj)
        a.neg_weighting = "none"              # "none" | "force_profile": Neil's W_kl = 1 - exp(-D_kl / tau_c)
        a.neg_weight_tau = 1.0                # tau_c = neg_weight_tau * batch median of D_kl (Huber, future force profiles)
        a.snippet_align = "next"              # "next": a_{t+1..t+H} (v1) | "executed": a_{t..t+Ta-1}, the executed chunk