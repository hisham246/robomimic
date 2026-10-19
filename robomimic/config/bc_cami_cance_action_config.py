"""
Config for BC + C-CaNCE + action-space contrast (see bc_cami_cance_act.py).

Inherits every C-CaNCE field from BCCaMICaNCEConfig and adds
algo.cami.action.* plus optimizer entries for the two new networks
(state_proj, action_encoder). The EMA target encoder has no optimizer.

ALGO_NAME must match register_algo_factory_func("bc_cami_cance_act").
"""

from copy import deepcopy

from robomimic.config.bc_cami_cance_config import BCCaMICaNCEConfig


class BCCaMICaNCEActConfig(BCCaMICaNCEConfig):
    ALGO_NAME = "bc_cami_cance_act"

    def algo_config(self):
        super(BCCaMICaNCEActConfig, self).algo_config()

        # optimizers for the new trainable networks (target encoder is EMA-only)
        self.algo.optim_params.state_proj = deepcopy(self.algo.optim_params.policy)
        self.algo.optim_params.action_encoder = deepcopy(self.algo.optim_params.policy)

        a = self.algo.cami.action
        a.enabled = True                  # False -> identical to plain C-CaNCE (control)
        a.joint_score = False             # False: additive hybrid | True: one softmax over (physics + action) score

        a.loss_weight_state = 0.01        # lambda_s: state-query loss (grad -> policy); used when joint_score=False
        a.loss_weight_traj = 1.0          # lambda_t: trajectory-query loss (grad -> action encoder only)
        a.loss_weight_joint = 0.01        # w_joint: used only when joint_score=True (replaces w_nce and lambda_s)

        a.joint_phys_aux_weight = 0.01    # joint mode only: weight of the standalone physics NCE (0 = pure joint; physics collapsed in the smoke test)

        a.temperature = 0.07             # beta_a: action-branch temperature
        a.snippet_horizon = 10            # H: future actions a_{t+1..t+H}; needs train.seq_length >= H + 1
        a.policy_latent_dim = 1024        # dim of the post-RNN per-step policy feature (checked at the first forward)
        a.contrastive_dim = 128           # shared embedding dim d
        a.state_proj_hidden_dims = (128,)     # psi: policy_latent_dim -> 128 -> d
        a.snippet_hidden_dim = 256            # action LSTM hidden size
        a.snippet_num_layers = 1
        a.key_proj_hidden_dims = (128,)       # g: 256 -> 128 -> d
        a.momentum = 0.995                # EMA coefficient m for the target encoder

        a.negative_mode = "regime"        # "regime": opposite contact regime only | "marginal": all other sequences
        a.contact_definition = "window"   # regime label: "window" = any contact in t=1..H | "anchor" = label at t=0