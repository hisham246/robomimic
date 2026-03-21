"""
Custom implementation of BC with CaMI (Contact-Aware Mutual Information).
"""
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import robomimic.utils.loss_utils as LossUtils

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC

@register_algo_factory_func("bc_cami")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC_CaMI algo class.
    """
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

class BC_CaMI(BC):
    """
    Behavioral Cloning with Contact-aware Mutual Information (CaMI) regularization.

    Version 1:
        - Keeps standard BC policy training behavior
        - Adds placeholders for future CaMI modules and losses
        - Adds momentum target update hook
    """

    def _make_snippet_tensor(self, obs_seq):
        """
        obs_seq contains:
        force           [B, H, 6]
        robot0_eef_pos  [B, H, 3]
        robot0_eef_quat [B, H, 4]
        Returns:
        x [B, H, 13]
        """
        return torch.cat(
            [
                obs_seq["force"],
                obs_seq["robot0_eef_pos"],
                obs_seq["robot0_eef_quat"],
            ],
            dim=-1,
        )

    def _encode_snippet(self, obs_seq, use_target=False):
        x = self._make_snippet_tensor(obs_seq)  # [B, H, 13]

        if use_target:
            _, (h_n, c_n) = self.nets["snippet_encoder_target"](x)
            feat = h_n[-1]   # [B, hidden_dim]
            key = self.nets["key_proj_target"](feat)
        else:
            _, (h_n, c_n) = self.nets["snippet_encoder"](x)
            feat = h_n[-1]
            key = self.nets["key_proj"](feat)

        if getattr(self.algo_config.cami, "normalize_embeddings", False):
            key = torch.nn.functional.normalize(key, dim=-1)

        return feat, key

    def _create_networks(self):
        """
        Create networks and place them into @self.nets.

        For the first scaffold:
            - keep standard BC policy
            - add placeholder query / snippet modules
            - add target snippet encoder for future momentum updates
        """
        super(BC_CaMI, self)._create_networks()

        anchor_dim = self.algo_config.actor_layer_dims[-1]
        query_hidden = list(self.algo_config.cami.query_proj_layers)
        key_hidden = list(self.algo_config.cami.key_proj_layers)
        contrastive_dim = self.algo_config.cami.contrastive_dim

        self.nets["query_proj"] = build_mlp(
            input_dim=anchor_dim,
            hidden_dims=query_hidden,
            output_dim=contrastive_dim,
        )

        snippet_input_dim = 13  # force + eef_pos + eef_quat

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

        self.nets["snippet_encoder_target"] = copy.deepcopy(self.nets["snippet_encoder"])
        self.nets["key_proj_target"] = copy.deepcopy(self.nets["key_proj"])

        for p in self.nets["snippet_encoder_target"].parameters():
            p.requires_grad = False
        for p in self.nets["key_proj_target"].parameters():
            p.requires_grad = False

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Process dataloader batch.

        Current behavior:
            - anchor obs = timestep 0 (same as BC)
            - action target = timestep 0
            - additionally build a positive future snippet from the same sequence
            using timesteps 1..H for low-dim CaMI snippet inputs
        """
        input_batch = dict()

        # BC-style current anchor at t = 0
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, 0, :]

        H = self.algo_config.cami.snippet_horizon
        snippet_keys = ["force", "robot0_eef_pos", "robot0_eef_quat"]

        # Build positive future snippet from same trajectory: timesteps 1..H
        if all(k in batch["obs"] for k in snippet_keys):
            pos_future_obs = {}

            for k in snippet_keys:
                seq = batch["obs"][k]   # expected [B, T, D]
                T = seq.shape[1]
                end = min(H + 1, T)

                # future slice: t = 1 .. H
                snippet = seq[:, 1:end, :]


                # pad with last available future frame if too short
                if snippet.shape[1] < H:
                    if snippet.shape[1] == 0:
                        # fallback if sequence length is unexpectedly 1
                        snippet = seq[:, 0:1, :].repeat(1, H, 1)
                    else:
                        pad_count = H - snippet.shape[1]
                        pad = snippet[:, -1:, :].repeat(1, pad_count, 1)
                        snippet = torch.cat([snippet, pad], dim=1)

                pos_future_obs[k] = snippet
            
            # diff = (pos_future_obs["force"][:, 0, :] - pos_future_obs["force"][:, -1, :]).abs().mean()
            # print("mean force diff across snippet:", diff.item())

            input_batch["pos_future_obs"] = pos_future_obs

        # # Optional contact label if already available
        # if "contact_label" in batch:
        #     input_batch["contact_label"] = batch["contact_label"]

        if "contact_label" in batch:
            cl = batch["contact_label"]
            # if sequence-shaped, use anchor timestep t=0
            if cl.ndim > 1:
                cl = cl[:, 0]
            input_batch["contact_label"] = cl

        # Move to device / float at the end
        input_batch = TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

        return input_batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Same structure as BC, but with a hook for momentum target updates after
        optimization when not validating.
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
                self._update_target_networks()

        return info

    def _forward_training(self, batch):
        predictions = OrderedDict()

        actions, anchor_feat = self.nets["policy"].forward_with_features(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        query_embedding = self.nets["query_proj"](anchor_feat)
        if getattr(self.algo_config.cami, "normalize_embeddings", False):
            query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)

        predictions["actions"] = actions
        predictions["anchor_feat"] = anchor_feat
        predictions["query_embedding"] = query_embedding

        if "pos_future_obs" in batch and all(
            k in batch["pos_future_obs"] for k in ["force", "robot0_eef_pos", "robot0_eef_quat"]
        ):
            pos_snippet_feat, pos_key_embedding = self._encode_snippet(
                batch["pos_future_obs"], use_target=False
            )
            predictions["pos_snippet_feat"] = pos_snippet_feat
            predictions["pos_key_embedding"] = pos_key_embedding

        if not hasattr(self, "_printed_cami_debug"):
            print("[BC_CaMI DEBUG] anchor_feat shape:", tuple(anchor_feat.shape))
            print("[BC_CaMI DEBUG] query_embedding shape:", tuple(query_embedding.shape))
            print("[BC_CaMI DEBUG] actions shape:", tuple(actions.shape))
            print("[BC_CaMI DEBUG] target actions shape:", tuple(batch["actions"].shape))

            if "pos_future_obs" in batch and all(
                k in batch["pos_future_obs"] for k in ["force", "robot0_eef_pos", "robot0_eef_quat"]
            ):
                print("[BC_CaMI DEBUG] pos_future force shape:", tuple(batch["pos_future_obs"]["force"].shape))
                print("[BC_CaMI DEBUG] pos_future eef_pos shape:", tuple(batch["pos_future_obs"]["robot0_eef_pos"].shape))
                print("[BC_CaMI DEBUG] pos_future eef_quat shape:", tuple(batch["pos_future_obs"]["robot0_eef_quat"].shape))
                print("[BC_CaMI DEBUG] pos_snippet_feat shape:", tuple(predictions["pos_snippet_feat"].shape))
                print("[BC_CaMI DEBUG] pos_key_embedding shape:", tuple(predictions["pos_key_embedding"].shape))

            self._printed_cami_debug = True

        return predictions
    
    def _compute_cami_contact_inbatch_loss(self, query_embedding, key_embedding, contact_label):
        """
        Contact-aware in-batch InfoNCE.

        For anchor i:
            positive = key_embedding[i]
            negatives = key_embedding[j] such that contact_label[j] != contact_label[i]

        Same-contact samples are ignored as negatives.
        Anchors with no valid opposite-contact negatives are skipped from the loss.

        Args:
            query_embedding: (B, D)
            key_embedding:   (B, D)
            contact_label:   (B,) tensor with binary or categorical labels

        Returns:
            cami_loss: scalar tensor
            cami_info: dict of debug stats
        """
        temperature = self.algo_config.cami.temperature

        if getattr(self.algo_config.cami, "normalize_embeddings", False):
            query_embedding = F.normalize(query_embedding, dim=-1)
            key_embedding = F.normalize(key_embedding, dim=-1)

        contact_label = contact_label.long().view(-1)
        B = query_embedding.shape[0]

        # similarity matrix: (B, B)
        logits = torch.matmul(query_embedding, key_embedding.T) / temperature

        # positive logits are diagonal entries
        pos_logits = logits.diag()  # (B,)

        # valid negatives for anchor i are samples j with opposite contact label
        neg_mask = contact_label.unsqueeze(1) != contact_label.unsqueeze(0)  # (B, B)

        # number of valid negatives per anchor
        valid_neg_count = neg_mask.sum(dim=1)  # (B,)
        valid_anchor_mask = valid_neg_count > 0  # (B,)

        # if no anchors in this batch have valid opposite-contact negatives, return zero safely
        if valid_anchor_mask.sum() == 0:
            zero = logits.sum() * 0.0
            cami_info = {
                "cami_valid_anchor_count": zero.detach(),
                "cami_valid_anchor_fraction": zero.detach(),
                "cami_pos_logit_mean": zero.detach(),
                "cami_neg_logit_mean": zero.detach(),
                "cami_retrieval_acc": zero.detach(),
                "cami_avg_valid_negatives": zero.detach(),
            }
            return zero, cami_info

        # masked negatives only
        neg_logits_masked = logits.masked_fill(~neg_mask, float("-inf"))  # (B, B)

        # denominator = exp(pos) + sum exp(valid negatives)
        denom_inputs = torch.cat([pos_logits.unsqueeze(1), neg_logits_masked], dim=1)  # (B, 1+B)
        log_denom = torch.logsumexp(denom_inputs, dim=1)  # (B,)

        per_anchor_loss = -(pos_logits - log_denom)  # (B,)
        cami_loss = per_anchor_loss[valid_anchor_mask].mean()

        with torch.no_grad():
            valid_pos_logits = pos_logits[valid_anchor_mask]
            pos_logit_mean = valid_pos_logits.mean()

            if neg_mask.any():
                neg_logit_mean = logits[neg_mask].mean()
            else:
                neg_logit_mean = torch.zeros((), device=logits.device)

            max_neg_logits = neg_logits_masked.max(dim=1).values
            retrieval_acc = (
                (pos_logits[valid_anchor_mask] > max_neg_logits[valid_anchor_mask]).float().mean()
            )

            cami_info = {
                "cami_valid_anchor_count": valid_anchor_mask.float().sum(),
                "cami_valid_anchor_fraction": valid_anchor_mask.float().mean(),
                "cami_pos_logit_mean": pos_logit_mean,
                "cami_neg_logit_mean": neg_logit_mean,
                "cami_retrieval_acc": retrieval_acc,
                "cami_avg_valid_negatives": valid_neg_count[valid_anchor_mask].float().mean(),
            }

        return cami_loss, cami_info
    
    def _compute_losses(self, predictions, batch):
        """
        Compute BC loss + contact-aware CaMI in-batch InfoNCE loss.
        """
        losses = OrderedDict()

        a_target = batch["actions"]
        actions = predictions["actions"]

        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)

        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        bc_action_loss = sum(action_losses)
        losses["bc_action_loss"] = bc_action_loss

        # Contact-aware CaMI in-batch InfoNCE
        if (
            getattr(self.algo_config.cami, "enabled", False)
            and "pos_key_embedding" in predictions
            and "contact_label" in batch
        ):
            query_embedding = predictions["query_embedding"]
            key_embedding = predictions["pos_key_embedding"]
            contact_label = batch["contact_label"].long().view(-1)

            cami_loss, cami_info = self._compute_cami_contact_inbatch_loss(
                query_embedding=query_embedding,
                key_embedding=key_embedding,
                contact_label=contact_label,
            )

            if not hasattr(self, "_printed_contact_debug"):
                print("[BC_CaMI DEBUG] contact_label shape:", tuple(contact_label.shape))
                print("[BC_CaMI DEBUG] unique contact labels:", torch.unique(contact_label))
                print("[BC_CaMI DEBUG] valid anchor fraction:", cami_info["cami_valid_anchor_fraction"].item())
                print("[BC_CaMI DEBUG] avg valid negatives:", cami_info["cami_avg_valid_negatives"].item())
                self._printed_contact_debug = True
            losses["cami_loss"] = cami_loss

            for k, v in cami_info.items():
                losses[k] = v

            losses["cami_logit_gap"] = losses["cami_pos_logit_mean"] - losses["cami_neg_logit_mean"]

            losses["action_loss"] = (
                bc_action_loss
                + self.algo_config.cami.loss_weight * cami_loss
            )
        else:
            zero = torch.zeros((), device=actions.device, dtype=actions.dtype)
            losses["cami_loss"] = zero
            losses["cami_logit_gap"] = zero
            losses["cami_valid_anchor_count"] = zero
            losses["cami_valid_anchor_fraction"] = zero
            losses["cami_pos_logit_mean"] = zero
            losses["cami_neg_logit_mean"] = zero
            losses["cami_retrieval_acc"] = zero
            losses["cami_avg_valid_negatives"] = zero
            losses["action_loss"] = bc_action_loss

        return losses

    def _train_step(self, losses):
        """
        Joint optimization step for BC policy + CaMI MVP modules.
        """
        info = OrderedDict()

        if not hasattr(self, "_printed_optimizer_debug"):
            print("self.nets keys:", list(self.nets.keys()))
            print("self.optimizers keys:", list(self.optimizers.keys()))
            self._printed_optimizer_debug = True

        trainable_names = ["policy", "query_proj", "snippet_encoder", "key_proj"]

        for name in trainable_names:
            self.optimizers[name].zero_grad()

        losses["action_loss"].backward()

        max_grad_norm = self.global_config.train.max_grad_norm

        for name in trainable_names:
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.nets[name].parameters(),
                    max_grad_norm,
                )
                info[f"{name}_grad_norm"] = float(grad_norm)
            else:
                total_norm_sq = 0.0
                for p in self.nets[name].parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm_sq += param_norm ** 2
                info[f"{name}_grad_norm"] = total_norm_sq ** 0.5

        for name in trainable_names:
            self.optimizers[name].step()

        return info
    
    @torch.no_grad()
    def _update_target_networks(self):
        """
        EMA update for target snippet encoder branch.

        Version 1:
            harmless placeholder update since modules are identities.
            Later this will update the actual future snippet encoder + key projection head.
        """
        if not getattr(self.algo_config.cami, "enabled", False):
            return
        if not getattr(self.algo_config.cami, "use_momentum_target", False):
            return

        tau = self.algo_config.cami.target_tau

        for online_param, target_param in zip(
            self.nets["snippet_encoder"].parameters(),
            self.nets["snippet_encoder_target"].parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)

        for online_param, target_param in zip(
            self.nets["key_proj"].parameters(),
            self.nets["key_proj_target"].parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)

    def log_info(self, info):
        log = super(BC_CaMI, self).log_info(info)
        losses = info["losses"]

        if "action_loss" in losses:
            log["Loss"] = losses["action_loss"].item()
        if "bc_action_loss" in losses:
            log["BC/Action_Loss"] = losses["bc_action_loss"].item()
        if "cami_loss" in losses:
            log["CaMI/Loss"] = losses["cami_loss"].item()

        if "l2_loss" in losses:
            log["BC/L2_Loss"] = losses["l2_loss"].item()
        if "l1_loss" in losses:
            log["BC/L1_Loss"] = losses["l1_loss"].item()
        if "cos_loss" in losses:
            log["BC/Cosine_Loss"] = losses["cos_loss"].item()

        if "cami_pos_logit_mean" in losses:
            log["CaMI/Pos_Logit_Mean"] = losses["cami_pos_logit_mean"].item()
        if "cami_neg_logit_mean" in losses:
            log["CaMI/Neg_Logit_Mean"] = losses["cami_neg_logit_mean"].item()
        if "cami_valid_anchor_count" in losses:
            log["CaMI/Valid_Anchor_Count"] = losses["cami_valid_anchor_count"].item()
        if "cami_valid_anchor_fraction" in losses:
            log["CaMI/Valid_Anchor_Fraction"] = losses["cami_valid_anchor_fraction"].item()
        if "cami_retrieval_acc" in losses:
            log["CaMI/Retrieval_Acc"] = losses["cami_retrieval_acc"].item()
        if "cami_avg_valid_negatives" in losses:
            log["CaMI/Avg_Valid_Negatives"] = losses["cami_avg_valid_negatives"].item()
        # if "query_norm_mean" in losses:
        #     log["CaMI/Query_Norm_Mean"] = losses["query_norm_mean"].item()
        # if "key_norm_mean" in losses:
        #     log["CaMI/Key_Norm_Mean"] = losses["key_norm_mean"].item()
        if "cami_logit_gap" in losses:
            log["CaMI/Logit_Gap"] = losses["cami_logit_gap"].item()

        if "policy_grad_norm" in info:
            log["GradNorm/Policy"] = info["policy_grad_norm"]
        if "query_proj_grad_norm" in info:
            log["GradNorm/Query_Proj"] = info["query_proj_grad_norm"]
        if "snippet_encoder_grad_norm" in info:
            log["GradNorm/Snippet_Encoder"] = info["snippet_encoder_grad_norm"]
        if "key_proj_grad_norm" in info:
            log["GradNorm/Key_Proj"] = info["key_proj_grad_norm"]

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        """
        assert not self.nets.training
        if "force" not in obs_dict:
            if ("robot0_ee_force" in obs_dict) and ("robot0_ee_torque" in obs_dict):
                obs_dict = dict(obs_dict)
                obs_dict["force"] = 1000.0 * torch.cat(
                    [obs_dict["robot0_ee_force"], obs_dict["robot0_ee_torque"]],
                    dim=-1,
                )

                if not hasattr(self, "_printed_force_fallback_debug"):
                    print("[FORCE FALLBACK DEBUG] robot0_ee_force shape:", tuple(obs_dict["robot0_ee_force"].shape))
                    print("[FORCE FALLBACK DEBUG] robot0_ee_torque shape:", tuple(obs_dict["robot0_ee_torque"].shape))
                    print("[FORCE FALLBACK DEBUG] constructed force shape:", tuple(obs_dict["force"].shape))
                    self._printed_force_fallback_debug = True
        
        if not hasattr(self, "_printed_obs_shape_debug"):
            print("[POLICY SHAPE DEBUG] expected obs_shapes:", self.obs_shapes)
            for k, v in obs_dict.items():
                if hasattr(v, "shape"):
                    print(f"[POLICY SHAPE DEBUG] received {k}: {tuple(v.shape)}")
            self._printed_obs_shape_debug = True

        return self.nets["policy"](obs_dict, goal_dict=goal_dict)
