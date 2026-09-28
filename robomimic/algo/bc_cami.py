"""
BC_RNN + CaMI for robomimic

This version is aligned to robomimic's BC_RNN training flow and also follows
the SaMI repo pattern of:
    - online encoder for query
    - target encoder for keys
    - EMA / Polyak update for the target encoder

Implemented losses:
    1) PDF-style CAMI:
        z_i     = psi(s_i)
        e_i^+   = phi*(tau_i)
        e_j^-   = phi*(tau_j), j in N_i
        N_i     = {j | k_j != k_i}

    2) SaMI-style trajectory momentum CAMI:
        q_i     = phi(tau_i)
        e_i^+   = phi*(tau_i)
        e_j^-   = phi*(tau_j), j in N_i

The second term is included so the online snippet encoder actually receives
gradient, similar to how SaMI trains its online encoder against a momentum
target encoder.
"""

from collections import OrderedDict
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC_RNN


@register_algo_factory_func("bc_cami")
def algo_config_to_class(algo_config):
    return BC_CaMI, {}


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class BC_CaMI(BC_RNN):
    """
    BC_RNN with Contact-aware Mutual Information regularization.

    Policy:
        standard BC_RNN policy over full observation sequences

    CAMI:
        - state encoder psi(s_i) for the PDF anchor query
        - online snippet encoder phi(tau_i)
        - momentum target snippet encoder phi*(tau_i)
        - contact-aware negatives only
    """

    def _create_networks(self):
    # Continuous CaMI uses the recorded force only during training.
    # Validate the configuration before constructing any networks so
    # incorrect parameters fail immediately instead of during training.
    continuous_contact_enabled = (
        self.algo_config.cami
        .get("continuous_contact", {})
        .get("enabled", False)
    )

    if continuous_contact_enabled:
        continuous_config = (
            self.algo_config.cami.continuous_contact
        )

        # Each of these parameters appears in a denominator,
        # threshold, or exponent. Zero, negative, NaN, or infinite
        # values would make the force comparison invalid.
        parameters_that_must_be_positive = (
            "force_scale",
            "huber_delta",
            "contact_temperature",
            "gamma",
        )

        for parameter_name in parameters_that_must_be_positive:
            parameter_value = float(
                continuous_config[parameter_name]
            )

            if (
                not math.isfinite(parameter_value)
                or parameter_value <= 0
            ):
                raise ValueError(
                    "continuous_contact.{} must be finite "
                    "and positive".format(parameter_name)
                )

        # A stacked observation does not have a simple one-to-one
        # alignment with the future force window. This implementation
        # therefore requires one observation frame per timestep.
        if self.global_config.train.frame_stack != 1:
            raise ValueError(
                "Force-weighted CaMI requires frame_stack=1 "
                "for aligned future windows"
            )

        # Sequence index zero is the current anchor timestep.
        # Indices 1 through H contain the future action and force
        # sequence used by the CaMI objective.
        required_sequence_length = (
            self.algo_config.cami.snippet_horizon + 1
        )

        if (
            self.global_config.train.seq_length
            < required_sequence_length
        ):
            raise ValueError(
                "seq_length must be at least "
                "snippet_horizon + 1"
            )

        # Force is loaded as an auxiliary dataset tensor.
        # It must therefore appear in train.dataset_keys.
        if (
            continuous_config.force_dataset_key
            not in self.global_config.train.dataset_keys
        ):
            raise ValueError(
                "Add {} to train.dataset_keys".format(
                    continuous_config.force_dataset_key
                )
            )

        force_observation_name = (
            continuous_config.force_dataset_key
            .split("/", 1)[-1]
        )

        # Force is privileged supervision for the loss.
        # It must not appear in the observation shapes used to
        # construct the policy network.
        if force_observation_name in self.obs_shapes:
            raise ValueError(
                "Privileged force must not be a policy "
                "observation"
            )

        super(BC_CaMI, self)._create_networks()

        contrastive_dim = self.algo_config.cami.contrastive_dim
        # state_hidden = list(
        #     getattr(
        #         self.algo_config.cami,
        #         "state_proj_layers",
        #         getattr(self.algo_config.cami, "query_proj_layers", []),
        #     )
        # )
        if "state_proj_layers" in self.algo_config.cami:
            state_hidden = list(self.algo_config.cami.state_proj_layers)
        elif "query_proj_layers" in self.algo_config.cami:
            state_hidden = list(self.algo_config.cami.query_proj_layers)
        else:
            raise RuntimeError(
                "CaMI config must define either 'state_proj_layers' or 'query_proj_layers'."
            )
        key_hidden = list(self.algo_config.cami.key_proj_layers)

        state_input_dim = self.algo_config.cami.policy_latent_dim

        self.nets["state_encoder"] = build_mlp(
            input_dim=state_input_dim,
            hidden_dims=state_hidden,
            output_dim=contrastive_dim,
        )

        # snippet input size from tau_i
        snippet_input_dim = self.ac_dim

        self.nets["snippet_encoder"] = nn.LSTM(
            input_size=snippet_input_dim,
            hidden_size=self.algo_config.cami.snippet_hidden_dim,
            num_layers=self.algo_config.cami.snippet_num_layers,
            batch_first=True,
        )

        self.nets["key_proj"] = build_mlp(
            input_dim=self.algo_config.cami.snippet_hidden_dim,
            hidden_dims=key_hidden,
            output_dim=contrastive_dim,
        )

        # momentum target branch
        self.nets["snippet_encoder_target"] = copy.deepcopy(self.nets["snippet_encoder"])
        self.nets["key_proj_target"] = copy.deepcopy(self.nets["key_proj"])

        for p in self.nets["snippet_encoder_target"].parameters():
            p.requires_grad = False
        for p in self.nets["key_proj_target"].parameters():
            p.requires_grad = False

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
    """
    Prepare a batch for BC-RNN and CaMI training.

    Binary CaMI reads a precomputed binary contact label.

    Continuous CaMI reads a synchronized force or wrench sequence
    from the HDF5 dataset. Force is used only to construct the
    contrastive loss and is not passed into the policy.
    """
    input_batch = {}

    # These are the normal inputs used by the existing BC-RNN
    # training pipeline.
    input_batch["goal_obs"] = batch.get(
        "goal_obs",
        None,
    )
    input_batch["actions"] = batch["actions"]

    continuous_contact_enabled = (
        self.algo_config.cami
        .get("continuous_contact", {})
        .get("enabled", False)
    )

    if continuous_contact_enabled:
        force_dataset_key = (
            self.algo_config
            .cami
            .continuous_contact
            .force_dataset_key
        )

        # SequenceDataset loads HDF5-relative dataset keys such as
        # "obs/force". Loading force this way makes it available to
        # the loss without adding it to the policy observations.
        if force_dataset_key not in batch:
            raise KeyError(
                "Missing stored wrench at batch[{!r}]. "
                "Add it to train.dataset_keys.".format(
                    force_dataset_key
                )
            )

        wrench = batch[force_dataset_key]

        # Expected layouts:
        #
        # [B,T,3] = Fx, Fy, Fz
        # [B,T,6] = Fx, Fy, Fz, Tx, Ty, Tz
        #
        # Only the first three translational-force channels are used.
        # Force and torque cannot be combined directly because they
        # have different physical units.
        if (
            wrench.ndim != 3
            or wrench.shape[-1] not in (3, 6)
        ):
            raise ValueError(
                "Stored wrench must have shape [B,T,3] "
                "or [B,T,6], with Fx,Fy,Fz first"
            )

        # The force, action, and observation at timestep t must all
        # refer to the same simulator state.
        if (
            wrench.shape[:2]
            != batch["actions"].shape[:2]
        ):
            raise ValueError(
                "Wrench and action sequences must be "
                "synchronized and have equal lengths"
            )

        expected_padding_mask_shape = (
            wrench.shape[:2] + (1,)
        )

        # Robomimic pads short sequences by repeating boundary
        # samples. The padding mask distinguishes those repeated
        # values from real force measurements.
        if (
            "pad_mask" not in batch
            or batch["pad_mask"].shape
            != expected_padding_mask_shape
        ):
            raise ValueError(
                "Continuous CaMI requires SequenceDataset "
                "get_pad_mask=True"
            )

        snippet_horizon = (
            self.algo_config.cami.snippet_horizon
        )

        # The batch must contain the current anchor at index zero
        # followed by H future timesteps.
        if wrench.shape[1] < snippet_horizon + 1:
            raise ValueError(
                "Batch must contain the anchor plus "
                "snippet_horizon future timesteps"
            )

        # The existing action-snippet encoder uses actions from
        # t+1 through t+H. Select force at those exact timesteps so
        # the physical signal and action snippet remain aligned.
        #
        # Resulting shape: [B,H,3]
        input_batch["force_sequence"] = (
            wrench[
                :,
                1:snippet_horizon + 1,
                :3,
            ].detach()
        )

        # Resulting shape: [B,H]
        #
        # True means the corresponding force sample came from the
        # demonstration. False means Robomimic generated it through
        # sequence padding.
        input_batch["force_valid"] = (
            batch["pad_mask"][
                :,
                1:snippet_horizon + 1,
                0,
            ].bool()
        )

        # self.obs_shapes contains only the observations declared
        # as policy inputs. Selecting from that collection prevents
        # the auxiliary force key from entering the policy encoder.
        input_batch["obs"] = {
            observation_name: batch["obs"][
                observation_name
            ]
            for observation_name in self.obs_shapes
        }

        return TensorUtils.to_float(
            TensorUtils.to_device(
                input_batch,
                self.device,
            )
        )

    # This is Hisham's original binary CaMI path.
    if "contact_label" in batch:
        contact_label = batch["contact_label"]
    elif "contact_label" in batch["obs"]:
        contact_label = batch["obs"]["contact_label"]
    else:
        contact_label = None

    if contact_label is None:
        raise KeyError(
            "Missing contact_label in batch"
        )

    # Convert [B,T], [B,T,1], or [B,1] into one label for
    # each sampled anchor.
    if contact_label.ndim > 1:
        contact_label = contact_label[:, 0]

    if contact_label.ndim > 1:
        contact_label = contact_label.squeeze(-1)

    input_batch["contact_label"] = (
        contact_label.float()
    )

    # Keep physical supervision out of policy inputs in binary
    # mode as well.
    excluded_policy_keys = {
        "force",
        "force_rawbias",
        "force_obsbias",
        "contact_label",
    }

    input_batch["obs"] = {
        observation_name: observation
        for observation_name, observation
        in batch["obs"].items()
        if observation_name
        not in excluded_policy_keys
    }

    # Print the binary-label distribution once. This helps detect
    # datasets containing only one class, which would provide no
    # opposite-contact negatives.
    if not hasattr(
        self,
        "_debug_printed_contact_stats",
    ):
        self._debug_printed_contact_stats = False

    if not self._debug_printed_contact_stats:
        print(
            "\n[BC_CaMI DEBUG] "
            "process_batch_for_training"
        )
        print(
            "  actions shape:",
            tuple(batch["actions"].shape),
        )
        print(
            "  contact_label shape:",
            tuple(
                input_batch[
                    "contact_label"
                ].shape
            ),
        )
        print(
            "  contact_label unique/counts:",
            torch.unique(
                input_batch["contact_label"],
                return_counts=True,
            ),
        )
        print(
            "  policy obs keys:",
            list(input_batch["obs"].keys()),
        )

        self._debug_printed_contact_stats = True

    return TensorUtils.to_float(
        TensorUtils.to_device(
            input_batch,
            self.device,
        )
    )

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Run BC_RNN train step, then update target encoders.
        """
        info = super(BC_CaMI, self).train_on_batch(
            batch=batch,
            epoch=epoch,
            validate=validate,
        )
        if not validate:
            self._update_target_networks()
        return info


    def _select_future_snippet(self, batch):
        """
        Future snippet tau_i from future expert actions at timesteps 1..H,
        padded to length H if needed.

        Returns:
            dict with key:
                "actions": Tensor of shape [B, H, A]
        """
        H = self.algo_config.cami.snippet_horizon

        seq = batch["actions"]  # [B, T, A]
        T = seq.shape[1]
        end = min(H + 1, T)

        # take future steps only: 1..H
        snippet = seq[:, 1:end]

        # pad to fixed horizon H
        if snippet.shape[1] < H:
            if snippet.shape[1] == 0:
                # degenerate case: no future step available
                snippet = seq[:, 0:1].repeat(1, H, 1)
            else:
                pad_count = H - snippet.shape[1]
                pad = snippet[:, -1:].repeat(1, pad_count, 1)
                snippet = torch.cat([snippet, pad], dim=1)

        return {"actions": snippet}

    def _make_snippet_tensor(self, snippet_dict):
        return snippet_dict["actions"]

    def _encode_anchor_state(self, anchor_latent):
        z = self.nets["state_encoder"](anchor_latent)

        normalize_embeddings = (
            self.algo_config.cami.normalize_embeddings
            if "normalize_embeddings" in self.algo_config.cami
            else False
        )
        if normalize_embeddings:
            z = F.normalize(z, dim=-1)

        return anchor_latent, z

    def _encode_snippet(self, obs_seq, use_target=False):
        """
        Encode tau_i with online or target trajectory encoder.
        """
        x = self._make_snippet_tensor(obs_seq)

        if use_target:
            with torch.no_grad():
                _, (h_n, _) = self.nets["snippet_encoder_target"](x)
                feat = h_n[-1]
                emb = self.nets["key_proj_target"](feat)
        else:
            _, (h_n, _) = self.nets["snippet_encoder"](x)
            feat = h_n[-1]
            emb = self.nets["key_proj"](feat)

        normalize_embeddings = (
            self.algo_config.cami.normalize_embeddings
            if "normalize_embeddings" in self.algo_config.cami
            else False
        )
        if normalize_embeddings:
            emb = F.normalize(emb, dim=-1)

        return feat, emb

    def _forward_training(self, batch):
        """
        Forward pass for BC_RNN + CAMI.
        """

        predictions = OrderedDict()
        # BC_RNN policy branch with exposed latent
        policy_actions, policy_feats = self._forward_policy_with_latent(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )
        predictions["actions"] = policy_actions

        anchor_latent = policy_feats[:, 0, :]   # [B, D]

        future_snippet = self._select_future_snippet(batch)

        if not hasattr(self, "_debug_printed_snippet_stats"):
            self._debug_printed_snippet_stats = False

        if not self._debug_printed_snippet_stats:
            print("\n[BC_CaMI DEBUG] _forward_training")
            print("  batch['actions'] shape          :", tuple(batch["actions"].shape))
            print("  future_snippet['actions'] shape :", tuple(future_snippet["actions"].shape))
            print("  first future action sample      :", future_snippet["actions"][0, 0].detach().cpu())
            self._debug_printed_snippet_stats = True
        

        if not hasattr(self, "_debug_printed_policy_feature_stats"):
            self._debug_printed_policy_feature_stats = False

        if not self._debug_printed_policy_feature_stats:
            print("\n[BC_CaMI DEBUG] policy latent")
            print("  policy class         :", type(self.nets["policy"]))
            print("  policy_actions shape :", tuple(policy_actions.shape))
            print("  policy_feats shape   :", tuple(policy_feats.shape))
            print("  anchor_latent shape  :", tuple(anchor_latent.shape))
            self._debug_printed_policy_feature_stats = True

        # PDF-style state query
        _, state_query = self._encode_anchor_state(anchor_latent)
        predictions["state_query_embedding"] = state_query

        # online snippet query, SaMI-style
        online_snippet_feat, online_snippet_query = self._encode_snippet(
            future_snippet,
            use_target=False,
        )
        predictions["online_snippet_feat"] = online_snippet_feat
        predictions["traj_query_embedding"] = online_snippet_query

        # target momentum key
        target_snippet_feat, target_key_embedding = self._encode_snippet(
            future_snippet,
            use_target=True,
        )
        predictions["target_snippet_feat"] = target_snippet_feat
        predictions["target_key_embedding"] = target_key_embedding

        return predictions
    
    def _forward_policy_with_latent(self, obs_dict, goal_dict=None):
        """
        Forward policy and return both action outputs and per-step latent features.
        """
        if not hasattr(self.nets["policy"], "forward_with_features"):
            raise AttributeError(
                "Policy network does not expose forward_with_features."
            )

        out = self.nets["policy"].forward_with_features(
            obs=obs_dict,
            goal=goal_dict,
        )

        if not isinstance(out, tuple) or len(out) != 2:
            raise RuntimeError(
                "policy.forward_with_features must return (actions, feats), "
                "but got type {}".format(type(out))
            )

        actions, feats = out

        if isinstance(actions, dict):
            if "action" in actions:
                actions = actions["action"]
            elif "actions" in actions:
                actions = actions["actions"]
            else:
                raise RuntimeError(
                    "Policy returned dict outputs but no 'action' or 'actions' key was found. "
                    f"Available keys: {list(actions.keys())}"
                )

        if not torch.is_tensor(actions):
            raise RuntimeError(
                "Expected policy actions to be a tensor, got {}".format(type(actions))
            )

        if not torch.is_tensor(feats):
            raise RuntimeError(
                "Expected policy feats to be a tensor, got {}".format(type(feats))
            )

        if feats.ndim != 3:
            raise RuntimeError(
                "Expected policy feats with shape [B, T, D], got shape {}".format(tuple(feats.shape))
            )

        return actions, feats

    def _compute_contact_inbatch_cami_loss(
    self,
    query_embedding,
    key_embedding,
    contact_label,
    force_sequence=None,
    force_valid=None,
):
    """
    Calculate contact-aware in-batch InfoNCE.

    Binary mode:
        A pair is a negative when its binary contact labels differ.

    Continuous mode:
        A pair receives a continuous negative weight based on the
        Huber distance between its future force-magnitude sequences.

    The paired item at index i remains the positive for anchor i.
    """
    embedding_temperature = (
        self.algo_config.cami.temperature
    )

    normalize_embeddings = (
        self.algo_config.cami.normalize_embeddings
        if "normalize_embeddings"
        in self.algo_config.cami
        else False
    )

    # Cosine-style similarity is obtained by normalizing before
    # taking the dot product.
    if normalize_embeddings:
        query_embedding = F.normalize(
            query_embedding,
            dim=-1,
        )
        key_embedding = F.normalize(
            key_embedding,
            dim=-1,
        )

    batch_size = query_embedding.shape[0]

    # s_ij = q_i^T k_j / beta
    #
    # Shape: [B,B]
    similarity_logits = (
        torch.matmul(
            query_embedding,
            key_embedding.T,
        )
        / embedding_temperature
    )

    # The diagonal represents the paired positive:
    # q_i matched with k_i.
    positive_logits = (
        similarity_logits.diag()
    )

    continuous_contact_enabled = (
        self.algo_config.cami
        .get("continuous_contact", {})
        .get("enabled", False)
    )

    negative_weights = None

    if continuous_contact_enabled:
        continuous_config = (
            self.algo_config
            .cami
            .continuous_contact
        )

        # force_sequence: [B,H,3]
        # force_valid:    [B,H]
        invalid_force_input = (
            force_sequence is None
            or force_valid is None
            or force_sequence.ndim != 3
            or force_sequence.shape[0]
            != batch_size
            or force_sequence.shape[2] != 3
            or force_valid.shape
            != force_sequence.shape[:2]
        )

        if invalid_force_input:
            raise ValueError(
                "Expected future force [B,H,3] "
                "and validity [B,H]"
            )

        # Force measurements determine which representation pairs
        # are treated as strong or weak negatives. Gradients must
        # update the learned embeddings, not the measurements.
        with torch.no_grad():
            valid_timesteps = (
                force_valid.detach().bool()
            )
            force = (
                force_sequence.detach().float()
            )

            # NaN or infinite measurements would propagate through
            # the Huber distance and corrupt the entire batch loss.
            if not torch.isfinite(
                force[valid_timesteps]
            ).all():
                raise ValueError(
                    "Recorded force contains "
                    "nonfinite values"
                )

            # Padding is temporarily replaced with zero. It is also
            # removed from the pairwise average below, so this zero
            # never becomes a real observation.
            force = force.masked_fill(
                ~valid_timesteps[..., None],
                0.0,
            )

            # Convert each three-dimensional force vector into a
            # scalar magnitude:
            #
            # c_i,h = ||F_i,h||_2 / force_scale
            #
            # Dividing by force_scale makes huber_delta operate on
            # normalized values rather than raw newtons.
            normalized_force_magnitude = (
                torch.linalg.vector_norm(
                    force,
                    dim=-1,
                )
                / continuous_config.force_scale
            )

            # Reshape through broadcasting:
            #
            # left:  [B,1,H]
            # right: [1,B,H]
            #
            # The broadcasted tensors have shape [B,B,H], where
            # entry (i,j,h) compares examples i and j at offset h.
            left_force, right_force = (
                torch.broadcast_tensors(
                    normalized_force_magnitude[
                        :, None, :
                    ],
                    normalized_force_magnitude[
                        None, :, :
                    ],
                )
            )

            # Calculate the Huber distance for every pair and
            # future timestep:
            #
            # rho_delta(c_i,h - c_j,h)
            #
            # Huber is quadratic for small differences and linear
            # for large differences. Large force spikes therefore
            # have less influence than they would under squared
            # distance.
            huber_distance_per_timestep = (
                F.huber_loss(
                    left_force,
                    right_force,
                    reduction="none",
                    delta=(
                        continuous_config
                        .huber_delta
                    ),
                )
            )

            # A pairwise timestep is usable only when both sampled
            # sequences contain a real measurement at that offset.
            #
            # Shape: [B,B,H]
            jointly_valid_timesteps = (
                valid_timesteps[:, None, :]
                & valid_timesteps[None, :, :]
            )

            valid_timestep_count = (
                jointly_valid_timesteps.sum(
                    dim=-1
                )
            )

            # Average the Huber values over jointly valid future
            # timesteps:
            #
            # D_ij = (1 / |V_ij|)
            #        sum_h in V_ij rho_delta(c_i,h - c_j,h)
            #
            # clamp_min prevents division by zero. Pairs with no
            # common timestep are explicitly removed afterward.
            pairwise_force_distance = (
                huber_distance_per_timestep
                .masked_fill(
                    ~jointly_valid_timesteps,
                    0.0,
                )
                .sum(dim=-1)
                / valid_timestep_count.clamp_min(1)
            )

            if not torch.isfinite(
                pairwise_force_distance
            ).all():
                raise ValueError(
                    "Nonfinite contact distances. "
                    "Check force units and force_scale."
                )

            # Convert distance into a bounded negative weight:
            #
            # W_ij =
            # (1 - exp(-D_ij / tau_c))^gamma
            #
            # Similar force histories produce weights near zero.
            # Different force histories produce weights closer to one.
            #
            # expm1 computes exp(x)-1 accurately when x is close
            # to zero. Negating expm1(-x) gives 1-exp(-x).
            negative_weights = (
                -torch.expm1(
                    -pairwise_force_distance
                    / continuous_config
                        .contact_temperature
                )
            ).clamp(
                min=0.0,
                max=1.0,
            ).pow(
                continuous_config.gamma
            )

            # A pair with no overlapping real measurements cannot
            # provide evidence about contact similarity.
            negative_weights = (
                negative_weights.masked_fill(
                    valid_timestep_count == 0,
                    0.0,
                )
            )

            # Diagonal entries correspond to the paired positives.
            # They must never also appear as negatives.
            negative_weights.fill_diagonal_(0.0)

            # A strictly positive weight means the pair contributes
            # to the contrastive denominator.
            negative_mask = (
                negative_weights > 0
            )

    else:
        # Preserve the original binary CaMI behavior.
        contact_label = (
            contact_label.long().view(-1)
        )

        # An example is a valid negative when its contact label
        # differs from the anchor's contact label.
        negative_mask = (
            contact_label.unsqueeze(1)
            != contact_label.unsqueeze(0)
        )

    valid_negative_count = (
        negative_mask.sum(dim=1)
    )

    valid_anchor_mask = (
        valid_negative_count > 0
    )

    # Print the negative distribution once so the dataset and
    # weighting behavior can be inspected without flooding logs.
    if not hasattr(
        self,
        "_debug_printed_neg_stats",
    ):
        self._debug_printed_neg_stats = False

    if not self._debug_printed_neg_stats:
        print(
            "\n[BC_CaMI DEBUG] "
            "_compute_contact_inbatch_cami_loss"
        )

        if continuous_contact_enabled:
            print(
                "  contact comparison: "
                "Huber-weighted future force magnitude"
            )
        else:
            print(
                "  contact_label unique/counts:",
                torch.unique(
                    contact_label,
                    return_counts=True,
                ),
            )

        print(
            "  valid negatives min/max/mean:",
            valid_negative_count.min().item(),
            valid_negative_count.max().item(),
            valid_negative_count.float().mean().item(),
        )

        print(
            "  valid anchor fraction:",
            valid_anchor_mask.float().mean().item(),
        )

        self._debug_printed_neg_stats = True

    # A batch may contain no usable negatives. Return a differentiable
    # zero instead of producing NaN through an empty reduction.
    if valid_anchor_mask.sum() == 0:
        zero = similarity_logits.sum() * 0.0

        info = {
            "valid_anchor_count": zero.detach(),
            "valid_anchor_fraction": zero.detach(),
            "pos_logit_mean": zero.detach(),
            "neg_logit_mean": zero.detach(),
            "retrieval_acc": zero.detach(),
            "avg_valid_negatives": zero.detach(),
            "soft_scale_mean": zero.detach(),
            "negative_weight_mean": zero.detach(),
            "effective_negatives": zero.detach(),
        }

        return zero, info

    # Invalid negative locations become negative infinity, so their
    # exponentials contribute zero to the InfoNCE denominator.
    masked_negative_logits = (
        similarity_logits.masked_fill(
            ~negative_mask,
            float("-inf"),
        )
    )

    if continuous_contact_enabled:
        # The continuous denominator contains:
        #
        # W_ij * exp(s_ij)
        #
        # In log space:
        #
        # log(W_ij * exp(s_ij))
        # = s_ij + log(W_ij)
        #
        # Masked entries receive a temporary weight of one before
        # log. Their logits remain negative infinity from the mask.
        log_negative_weights = (
            negative_weights
            .masked_fill(
                ~negative_mask,
                1.0,
            )
            .log()
        )

        masked_negative_logits = (
            masked_negative_logits
            + log_negative_weights
        )

    # The first column contains the paired positive. Remaining
    # columns contain all eligible in-batch negatives.
    denominator_logits = torch.cat(
        [
            positive_logits.unsqueeze(1),
            masked_negative_logits,
        ],
        dim=1,
    )

    log_denominator = torch.logsumexp(
        denominator_logits,
        dim=1,
    )

    # L_i = -log(
    #     exp(s_ii)
    #     /
    #     (exp(s_ii) + sum_j W_ij exp(s_ij))
    # )
    per_anchor_loss = -(
        positive_logits - log_denominator
    )

    soft_scale_mean = torch.zeros(
        (),
        device=query_embedding.device,
    )

    soft_variant = (
        self.algo_config.cami.soft_variant
        if "soft_variant"
        in self.algo_config.cami
        else False
    )

    # Preserve the upstream soft variant for the binary baseline.
    # Continuous mode already supplies a continuous pair weight,
    # so applying the separate soft scaling would mix two different
    # weighting mechanisms.
    if (
        soft_variant
        and not continuous_contact_enabled
    ):
        scales = torch.ones(
            batch_size,
            device=query_embedding.device,
            dtype=query_embedding.dtype,
        )

        for anchor_index in range(batch_size):
            if valid_anchor_mask[anchor_index]:
                positive_key = key_embedding[
                    anchor_index
                ]

                negative_keys = key_embedding[
                    negative_mask[anchor_index]
                ]

                mean_embedding_distance = torch.norm(
                    positive_key.unsqueeze(0)
                    - negative_keys,
                    dim=1,
                ).mean()

                scales[anchor_index] = torch.clamp(
                    mean_embedding_distance,
                    min=1.0,
                )

        per_anchor_loss = (
            per_anchor_loss * scales
        )

        soft_scale_mean = scales[
            valid_anchor_mask
        ].mean()

    if continuous_contact_enabled:
        # In continuous mode, an anchor without an eligible
        # negative has denominator equal to its numerator.
        # Its loss is therefore exactly zero and it remains in the
        # full-batch average.
        loss = per_anchor_loss.mean()
    else:
        # Preserve the original reduction for the binary baseline.
        loss = per_anchor_loss[
            valid_anchor_mask
        ].mean()

    with torch.no_grad():
        valid_positive_logits = positive_logits[
            valid_anchor_mask
        ]

        positive_logit_mean = (
            valid_positive_logits.mean()
        )

        negative_logit_mean = (
            similarity_logits[
                negative_mask
            ].mean()
            if negative_mask.any()
            else torch.zeros(
                (),
                device=similarity_logits.device,
            )
        )

        maximum_negative_logits = (
            masked_negative_logits
            .max(dim=1)
            .values
        )

        # A retrieval is correct when the paired positive scores
        # higher than every eligible negative for that anchor.
        retrieval_accuracy = (
            positive_logits[valid_anchor_mask]
            > maximum_negative_logits[
                valid_anchor_mask
            ]
        ).float().mean()

        if continuous_contact_enabled:
            # Mean W_ij over all off-diagonal pair positions.
            negative_weight_mean = (
                negative_weights.sum()
                / max(
                    batch_size
                    * (batch_size - 1),
                    1,
                )
            )

            # Sum of continuous negative weights per anchor.
            # This is the effective number of full-strength negatives.
            effective_negatives = (
                negative_weights
                .sum(dim=1)
                .mean()
            )
        else:
            # Binary weights are either zero or one, so these
            # quantities reduce to the proportion and number of
            # valid binary negatives.
            negative_weight_mean = (
                negative_mask.float().sum()
                / max(
                    batch_size
                    * (batch_size - 1),
                    1,
                )
            )

            effective_negatives = (
                valid_negative_count
                .float()
                .mean()
            )

        info = {
            "valid_anchor_count": (
                valid_anchor_mask.float().sum()
            ),
            "valid_anchor_fraction": (
                valid_anchor_mask.float().mean()
            ),
            "pos_logit_mean": (
                positive_logit_mean
            ),
            "neg_logit_mean": (
                negative_logit_mean
            ),
            "retrieval_acc": (
                retrieval_accuracy
            ),
            "avg_valid_negatives": (
                valid_negative_count[
                    valid_anchor_mask
                ].float().mean()
            ),
            "soft_scale_mean": (
                soft_scale_mean.detach()
            ),
            "negative_weight_mean": (
                negative_weight_mean
            ),
            "effective_negatives": (
                effective_negatives
            ),
        }

    return loss, info

    # def _compute_contact_inbatch_cami_loss(self, query_embedding, key_embedding, contact_label):
    #     """
    #     Contact-aware InfoNCE:

    #         L_i = -log exp(q_i·k_i / beta)
    #                    ---------------------------------------------
    #                    exp(q_i·k_i / beta) + sum_{j in N_i} exp(q_i·k_j / beta)

    #     where N_i = {j | k_j != k_i}
    #     """
    #     beta = self.algo_config.cami.temperature

    #     normalize_embeddings = (
    #         self.algo_config.cami.normalize_embeddings
    #         if "normalize_embeddings" in self.algo_config.cami
    #         else False
    #     )
    #     if normalize_embeddings:
    #         query_embedding = F.normalize(query_embedding, dim=-1)
    #         key_embedding = F.normalize(key_embedding, dim=-1)

    #     contact_label = contact_label.long().view(-1)
    #     B = query_embedding.shape[0]

    #     logits = torch.matmul(query_embedding, key_embedding.T) / beta
    #     pos_logits = logits.diag()

    #     neg_mask = contact_label.unsqueeze(1) != contact_label.unsqueeze(0)
    #     valid_neg_count = neg_mask.sum(dim=1)
    #     valid_anchor_mask = valid_neg_count > 0

    #     if not hasattr(self, "_debug_printed_neg_stats"):
    #         self._debug_printed_neg_stats = False

    #     if not self._debug_printed_neg_stats:
    #         print("\n[BC_CaMI DEBUG] _compute_contact_inbatch_cami_loss")
    #         print("  contact_label unique/counts :", torch.unique(contact_label, return_counts=True))
    #         print("  valid_neg_count min/max/mean:",
    #             valid_neg_count.min().item(),
    #             valid_neg_count.max().item(),
    #             valid_neg_count.float().mean().item())
    #         print("  valid_anchor_fraction       :", valid_anchor_mask.float().mean().item())
    #         self._debug_printed_neg_stats = True

    #     if valid_anchor_mask.sum() == 0:
    #         zero = logits.sum() * 0.0
    #         info = {
    #             "valid_anchor_count": zero.detach(),
    #             "valid_anchor_fraction": zero.detach(),
    #             "pos_logit_mean": zero.detach(),
    #             "neg_logit_mean": zero.detach(),
    #             "retrieval_acc": zero.detach(),
    #             "avg_valid_negatives": zero.detach(),
    #             "soft_scale_mean": zero.detach(),
    #         }
    #         return zero, info

    #     neg_logits_masked = logits.masked_fill(~neg_mask, float("-inf"))
    #     denom_inputs = torch.cat([pos_logits.unsqueeze(1), neg_logits_masked], dim=1)
    #     log_denom = torch.logsumexp(denom_inputs, dim=1)
    #     per_anchor_loss = -(pos_logits - log_denom)

    #     soft_scale_mean = torch.zeros((), device=query_embedding.device)
    #     soft_variant = (
    #         self.algo_config.cami.soft_variant
    #         if "soft_variant" in self.algo_config.cami
    #         else False
    #     )

    #     if soft_variant:
    #         scales = torch.ones(B, device=query_embedding.device, dtype=query_embedding.dtype)
    #         for i in range(B):
    #             if valid_anchor_mask[i]:
    #                 e_pos_i = key_embedding[i]
    #                 e_neg_i = key_embedding[neg_mask[i]]
    #                 dist_mean = torch.norm(e_pos_i.unsqueeze(0) - e_neg_i, dim=1).mean()
    #                 scales[i] = torch.clamp(dist_mean, min=1.0)
    #         per_anchor_loss = per_anchor_loss * scales
    #         soft_scale_mean = scales[valid_anchor_mask].mean()

    #     loss = per_anchor_loss[valid_anchor_mask].mean()

    #     with torch.no_grad():
    #         valid_pos_logits = pos_logits[valid_anchor_mask]
    #         pos_logit_mean = valid_pos_logits.mean()
    #         neg_logit_mean = logits[neg_mask].mean() if neg_mask.any() else torch.zeros((), device=logits.device)
    #         max_neg_logits = neg_logits_masked.max(dim=1).values
    #         retrieval_acc = (
    #             (pos_logits[valid_anchor_mask] > max_neg_logits[valid_anchor_mask]).float().mean()
    #         )

    #         info = {
    #             "valid_anchor_count": valid_anchor_mask.float().sum(),
    #             "valid_anchor_fraction": valid_anchor_mask.float().mean(),
    #             "pos_logit_mean": pos_logit_mean,
    #             "neg_logit_mean": neg_logit_mean,
    #             "retrieval_acc": retrieval_acc,
    #             "avg_valid_negatives": valid_neg_count[valid_anchor_mask].float().mean(),
    #             "soft_scale_mean": soft_scale_mean.detach(),
    #         }

    #     return loss, info

    def _compute_losses(self, predictions, batch):
        """
        Total loss:
            L_total = L_BC
                    + lambda_state * L_CaMI_state
                    + lambda_traj  * L_CaMI_traj
        """
        losses = OrderedDict()

        a_target = batch["actions"]
        actions = predictions["actions"]

        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        bc_action_loss = (
            self.algo_config.loss.l2_weight * losses["l2_loss"]
            + self.algo_config.loss.l1_weight * losses["l1_loss"]
            + self.algo_config.loss.cos_weight * losses["cos_loss"]
        )
        losses["bc_action_loss"] = bc_action_loss

        zero = torch.zeros((), device=actions.device, dtype=actions.dtype)

        state_cami_loss = zero
        traj_cami_loss = zero

        cami_enabled = (
            self.algo_config.cami.enabled
            if "enabled" in self.algo_config.cami
            else False
        )
        if cami_enabled:
            contact_label = batch["contact_label"].view(-1)
            target_key_embedding = predictions["target_key_embedding"]

            # PDF-style CAMI: psi(s_i) against phi*(tau_i)
            state_query_embedding = predictions["state_query_embedding"]
            state_cami_loss, state_info = self._compute_contact_inbatch_cami_loss(
                query_embedding=state_query_embedding,
                key_embedding=target_key_embedding,
                contact_label=contact_label,
            )
            losses["state_cami_loss"] = state_cami_loss

            # SaMI-style momentum CAMI: phi(tau_i) against phi*(tau_i)
            traj_query_embedding = predictions["traj_query_embedding"]
            traj_cami_loss, traj_info = self._compute_contact_inbatch_cami_loss(
                query_embedding=traj_query_embedding,
                key_embedding=target_key_embedding,
                contact_label=contact_label,
            )
            losses["traj_cami_loss"] = traj_cami_loss

            # log state loss diagnostics
            losses["state_valid_anchor_count"] = state_info["valid_anchor_count"]
            losses["state_valid_anchor_fraction"] = state_info["valid_anchor_fraction"]
            losses["state_pos_logit_mean"] = state_info["pos_logit_mean"]
            losses["state_neg_logit_mean"] = state_info["neg_logit_mean"]
            losses["state_retrieval_acc"] = state_info["retrieval_acc"]
            losses["state_avg_valid_negatives"] = state_info["avg_valid_negatives"]
            losses["state_soft_scale_mean"] = state_info["soft_scale_mean"]

            # log traj loss diagnostics
            losses["traj_valid_anchor_count"] = traj_info["valid_anchor_count"]
            losses["traj_valid_anchor_fraction"] = traj_info["valid_anchor_fraction"]
            losses["traj_pos_logit_mean"] = traj_info["pos_logit_mean"]
            losses["traj_neg_logit_mean"] = traj_info["neg_logit_mean"]
            losses["traj_retrieval_acc"] = traj_info["retrieval_acc"]
            losses["traj_avg_valid_negatives"] = traj_info["avg_valid_negatives"]
            losses["traj_soft_scale_mean"] = traj_info["soft_scale_mean"]

        else:
            losses["state_cami_loss"] = zero
            losses["traj_cami_loss"] = zero

        # lambda_state = getattr(self.algo_config.cami, "state_loss_weight", self.algo_config.cami.loss_weight)
        lambda_state = (
            self.algo_config.cami.state_loss_weight
            if "state_loss_weight" in self.algo_config.cami
            else self.algo_config.cami.loss_weight
        )
        lambda_traj = (
            self.algo_config.cami.traj_loss_weight
            if "traj_loss_weight" in self.algo_config.cami
            else 1.0
        )
        losses["action_loss"] = (
            bc_action_loss
            + lambda_state * state_cami_loss
            + lambda_traj * traj_cami_loss
        )

        return losses

    def _train_step(self, losses):
        """
        Backprop through:
            - policy
            - state_encoder
            - snippet_encoder
            - key_proj
        """
        required_optimizers = ["policy", "state_encoder", "snippet_encoder", "key_proj"]
        missing = [k for k in required_optimizers if k not in self.optimizers]
        if len(missing) > 0:
            raise KeyError(
                "Missing optimizer entries for {}. Add them under algo.optim_params in your config.".format(missing)
            )

        info = OrderedDict()

        for name in required_optimizers:
            self.optimizers[name].zero_grad()

        losses["action_loss"].backward()

        max_grad_norm = self.global_config.train.max_grad_norm
        for name in required_optimizers:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.nets[name].parameters(),
                max_grad_norm if max_grad_norm is not None else 1e9,
            )
            info[f"{name}_grad_norm"] = float(grad_norm)

        for name in required_optimizers:
            self.optimizers[name].step()

        return info

    @torch.no_grad()
    def _update_target_networks(self):
        """
        EMA / Polyak update of target trajectory encoder.
        """
        cami_enabled = (
            self.algo_config.cami.enabled
            if "enabled" in self.algo_config.cami
            else False
        )
        if not cami_enabled:
            return

        use_momentum_target = (
            self.algo_config.cami.use_momentum_target
            if "use_momentum_target" in self.algo_config.cami
            else True
        )
        if not use_momentum_target:
            return

        m = self.algo_config.cami.momentum if "momentum" in self.algo_config.cami else None
        if m is None:
            tau = self.algo_config.cami.target_tau if "target_tau" in self.algo_config.cami else 0.05
            m = 1.0 - tau

        for online_param, target_param in zip(
            self.nets["snippet_encoder"].parameters(),
            self.nets["snippet_encoder_target"].parameters(),
        ):
            target_param.data.mul_(m)
            target_param.data.add_((1.0 - m) * online_param.data)

        for online_param, target_param in zip(
            self.nets["key_proj"].parameters(),
            self.nets["key_proj_target"].parameters(),
        ):
            target_param.data.mul_(m)
            target_param.data.add_((1.0 - m) * online_param.data)

    def log_info(self, info):
        log = super(BC_CaMI, self).log_info(info)
        losses = info["losses"]

        log["Loss"] = losses["action_loss"].item()
        log["BC_Action_Loss"] = losses["bc_action_loss"].item()
        log["State_CaMI_Loss"] = losses["state_cami_loss"].item()
        log["Traj_CaMI_Loss"] = losses["traj_cami_loss"].item()

        if "l2_loss" in losses:
            log["L2_Loss"] = losses["l2_loss"].item()
        if "l1_loss" in losses:
            log["L1_Loss"] = losses["l1_loss"].item()
        if "cos_loss" in losses:
            log["Cosine_Loss"] = losses["cos_loss"].item()

        for key in [
            "state_valid_anchor_count",
            "state_valid_anchor_fraction",
            "state_pos_logit_mean",
            "state_neg_logit_mean",
            "state_retrieval_acc",
            "state_avg_valid_negatives",
            "state_soft_scale_mean",
            "traj_valid_anchor_count",
            "traj_valid_anchor_fraction",
            "traj_pos_logit_mean",
            "traj_neg_logit_mean",
            "traj_retrieval_acc",
            "traj_avg_valid_negatives",
            "traj_soft_scale_mean",
        ]:
            if key in losses:
                log[key] = losses[key].item()

        for key in [
            "policy_grad_norm",
            "state_encoder_grad_norm",
            "snippet_encoder_grad_norm",
            "key_proj_grad_norm",
        ]:
            if key in info:
                log[key] = info[key]

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Preserve BC_RNN rollout behavior.
        """
        assert not self.nets.training

        # if "force" not in obs_dict:
        #     if ("robot0_ee_force" in obs_dict) and ("robot0_ee_torque" in obs_dict):
        #         obs_dict = dict(obs_dict)
        #         obs_dict["force"] = torch.cat(
        #             [obs_dict["robot0_ee_force"], obs_dict["robot0_ee_torque"]],
        #             dim=-1,
        #         )

        expected_keys = list(self.obs_shapes.keys())
        filtered_obs_dict = {k: obs_dict[k] for k in expected_keys if k in obs_dict}
        missing_keys = [k for k in expected_keys if k not in filtered_obs_dict]
        if len(missing_keys) > 0:
            raise KeyError(f"Missing required rollout observation keys: {missing_keys}")

        return super(BC_CaMI, self).get_action(filtered_obs_dict, goal_dict=goal_dict)
