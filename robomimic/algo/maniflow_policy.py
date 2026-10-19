"""
Implementation of ManiFlow Policy by Yan et al., CoRL 2025. Project: https://maniflow-policy.github.io/
"""
from typing import Callable
from packaging.version import parse as parse_version
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from collections import OrderedDict, deque
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_nets as ObsNets
from diffusers.training_utils import EMAModel
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils


from robomimic.algo import register_algo_factory_func, PolicyAlgo
import robomimic.models.maniflow_policy_nets as MFNets


@register_algo_factory_func("maniflow_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.transformer.enabled:
        return ManiFlowPolicy, {}
    else:
        raise RuntimeError()

class ManiFlowPolicy(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = MFNets.ManiFlowTransformer(
            input_dim=self.ac_dim,
            output_dim=self.ac_dim,
            horizon=self.algo_config.horizon.prediction_horizon,
            cond_dim=obs_dim,                                        # per-frame feature size
            cond_len=self.algo_config.horizon.observation_horizon,   # one token per obs frame
            timestep_embed_dim=self.algo_config.transformer.timestep_embed_dim,
            target_t_embed_dim=self.algo_config.transformer.timestep_embed_dim,
            n_layer=self.algo_config.transformer.n_layer,
            n_head=self.algo_config.transformer.n_head,
            n_emb=self.algo_config.transformer.n_emb,
        )
        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        mf = self.algo_config.maniflow
        n_flow = int(B * mf.flow_batch_ratio)
        n_ct = B - n_flow

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(ManiFlowPolicy, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            # obs features are passed to the transformer as [B, To, D] tokens (no flattening)
            net = self.nets["policy"]["noise_pred_net"]
            relative = (mf.target_t_mode == "relative")

            # flow loss: learn instantaneous velocity (dt = 0) on the first n_flow samples
            a_f = actions[:n_flow]
            x1 = a_f
            x0 = torch.randn_like(x1)
            t = self._sample_t(n_flow, mf.sample_t_mode_flow)
            xt = (1 - t) * x0 + t * x1
            dt = torch.zeros(n_flow, device=self.device)
            target_t = dt if relative else t.view(-1) + dt
            v_pred = net(xt, t.view(-1), target_t, global_cond=obs_features[:n_flow])
            loss_flow = F.mse_loss(v_pred, x1 - x0)
            loss = loss_flow

            # consistency loss: match the EMA model's average velocity over [t_next, t_next + dt]
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
                                    global_cond=obs_features[n_flow:].detach())
                x1_hat = x_t_next + (1 - t_next) * v_avg
                v_target = (x1_hat - xt) / (1 - t)
                target_t = dt if relative else t_next.view(-1)
                v_pred = net(xt, t.view(-1), target_t, global_cond=obs_features[n_flow:])
                loss_ct = F.mse_loss(v_pred, v_target)
                loss = loss + loss_ct

            # logging
            losses = {
                "l2_loss": loss,
                "loss_flow": loss_flow,
                "loss_ct": loss_ct,
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(ManiFlowPolicy, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        log["Flow_Loss"] = info["losses"]["loss_flow"].item()
        log["Consistency_Loss"] = info["losses"]["loss_ct"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def _sample_t(self, n, mode):
        """
        Sample flow times t in [0, 1) with shape [n, 1, 1].
        """
        if mode == "beta":
            # shifted Beta(1, 1.5) scaled to [0, 0.999], as in ManiFlow
            dist = Beta(torch.tensor(1.0, device=self.device), torch.tensor(1.5, device=self.device))
            t = 0.999 * dist.sample((n, 1, 1))
        elif mode == "uniform":
            t = torch.rand((n, 1, 1), device=self.device)
        elif mode == "discrete":
            denoise_timesteps = self.algo_config.maniflow.denoise_timesteps
            t = torch.randint(0, denoise_timesteps, (n, 1, 1), device=self.device).float() / denoise_timesteps
        else:
            raise ValueError(f"Unsupported t sampling mode {mode}")
        return t

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        mf = self.algo_config.maniflow
        N = mf.num_inference_steps

        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # initialize action from Gaussian noise (t = 0) and integrate the flow to t = 1 with Euler steps
        naction = torch.randn((B, Tp, action_dim), device=self.device)
        dt = 1.0 / N
        with torch.no_grad():
            for i in range(N):
                t = torch.full((B,), i * dt, device=self.device)
                target_t = torch.full((B,), dt, device=self.device)
                if mf.target_t_mode == "absolute":
                    target_t = t + dt
                v = nets["policy"]["noise_pred_net"](
                    sample=naction,
                    timestep=t,
                    target_t=target_t,
                    global_cond=obs_features
                )
                naction = naction + v * dt

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
