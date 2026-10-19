"""
Config for ManiFlow Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig

class ManiFlowPolicyConfig(BaseConfig):
    ALGO_NAME = "maniflow_policy"

    def train_config(self):
        """
        Setting up training parameters for Diffusion Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(ManiFlowPolicyConfig, self).train_config()
        
        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # set compatible data loading parameters
        self.train.seq_length = 16 # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2 # should match self.algo.horizon.observation_horizon
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5 # number of cosine cycles (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500 # number of warmup steps (used by "cosine" scheduler)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs (used by "linear" and "multistep" schedulers)
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6          # L2 regularization strength

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # denoiser (transformer)
        self.algo.transformer.enabled = True
        self.algo.transformer.n_layer = 3
        self.algo.transformer.n_head = 4
        self.algo.transformer.n_emb = 256
        self.algo.transformer.timestep_embed_dim = 256

        # ManiFlow
        self.algo.maniflow.flow_batch_ratio = 0.75
        self.algo.maniflow.consistency_batch_ratio = 0.25
        self.algo.maniflow.sample_t_mode_flow = "beta"
        self.algo.maniflow.sample_t_mode_consistency = "discrete"
        self.algo.maniflow.denoise_timesteps = 10
        self.algo.maniflow.num_inference_steps = 2   # try 1 or 2
        self.algo.maniflow.target_t_mode = "relative"

        
        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75