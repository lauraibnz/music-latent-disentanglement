import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import copy

from mld.autoencoder import (
    build_latent_codec,
    decode_model_latent_to_audio,
    encode_audio_batch_to_model_latent,
    encode_audio_to_model_latent,
    normalize_latent_codec_name,
)


def _normalize_group_key(group_key):
    if isinstance(group_key, (list, tuple)):
        return tuple(str(key) for key in group_key)
    return str(group_key)


def _resolve_metadata_group(meta, group_key):
    if not isinstance(meta, dict):
        raise ValueError(f"Expected metadata dict, got {type(meta).__name__}")

    if isinstance(group_key, tuple):
        missing = [key for key in group_key if key not in meta]
        if missing:
            raise ValueError(
                f"Missing composite group keys {missing} in metadata. "
                f"Available keys: {list(meta.keys())}"
            )
        return tuple(meta[key] for key in group_key)

    if group_key not in meta:
        raise ValueError(
            f"Missing group key '{group_key}' in metadata. "
            f"Available keys: {list(meta.keys())}"
        )
    return meta[group_key]


@gin.configurable
class Base(nn.Module):
    """Base model for disentanglement - agnostic to decoder architecture."""
    
    def __init__(self,
                 net=None,
                 encoder=None,
                 encoder_time=None,
                 latent_dim=64,
                 drop_value=-4.0,
                 device="cpu",
                 use_raw_time_cond_if_no_encoder=True,
                 latent_codec="music2latent"):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.drop_value = float(drop_value)
        self.use_raw_time_cond_if_no_encoder = bool(use_raw_time_cond_if_no_encoder)
        self.latent_codec = normalize_latent_codec_name(latent_codec)
        if self.latent_codec == "auto":
            raise ValueError("Base.latent_codec cannot be 'auto'. Use an explicit codec name.")
        
        # Encoders (will be set via gin config)
        self.encoder = encoder
        self.encoder_time = encoder_time
        
        # Decoder network - could be UNet, Flow, etc.
        self.net = net
        
        self._target_device = torch.device(device)

    @property
    def device(self):
        """Get model device."""
        return next(self.parameters()).device

    def encode_structure(self, latent_time):
        """Encode structure conditioning from latents (keeps temporal info)."""
        if self.encoder_time is not None:
            return self.encoder_time(latent_time)
        if self.use_raw_time_cond_if_no_encoder:
            return latent_time
        return None

    def _sample_struct_aug_latent(self, batch, latents):
        """Resolve one structure-augmentation latent per sample.

        Supports both:
        - legacy pre-selected `latent_struct_aug` shaped [B, C, T]
        - bank-style `z_struct_aug` shaped [B, K, C, T]
        """
        latent_struct_aug = batch.get("latent_struct_aug")
        if latent_struct_aug is not None:
            if latent_struct_aug.ndim != 3:
                raise ValueError(
                    f"'latent_struct_aug' must have shape [B, C, T], got {tuple(latent_struct_aug.shape)}"
                )
            if latent_struct_aug.shape != latents.shape:
                raise ValueError(
                    f"Shape mismatch: latent_struct_aug {tuple(latent_struct_aug.shape)} vs latent {tuple(latents.shape)}"
                )
            return latent_struct_aug

        struct_aug_bank = batch.get("z_struct_aug")
        if struct_aug_bank is None:
            raise ValueError(
                "use_struct_aug=True requires either batch['z_struct_aug'] or batch['latent_struct_aug']. "
                "Rebuild dataset with --save_struct_aug_latent."
            )

        if struct_aug_bank.ndim == 3:
            if struct_aug_bank.shape != latents.shape:
                raise ValueError(
                    f"Shape mismatch: z_struct_aug {tuple(struct_aug_bank.shape)} vs latent {tuple(latents.shape)}"
                )
            return struct_aug_bank

        if struct_aug_bank.ndim != 4:
            raise ValueError(
                f"'z_struct_aug' must have shape [B, K, C, T] or [B, C, T], got {tuple(struct_aug_bank.shape)}"
            )

        if struct_aug_bank.shape[0] != latents.shape[0] or struct_aug_bank.shape[2:] != latents.shape[1:]:
            raise ValueError(
                f"Shape mismatch: z_struct_aug {tuple(struct_aug_bank.shape)} vs latent {tuple(latents.shape)}"
            )

        batch_size, bank_size = int(struct_aug_bank.shape[0]), int(struct_aug_bank.shape[1])
        if bank_size <= 0:
            raise ValueError(f"'z_struct_aug' bank must have at least one augmentation, got shape {tuple(struct_aug_bank.shape)}")

        choice_indices = torch.randint(
            low=0,
            high=bank_size,
            size=(batch_size,),
            device=struct_aug_bank.device,
        )
        batch_indices = torch.arange(batch_size, device=struct_aug_bank.device)
        return struct_aug_bank[batch_indices, choice_indices]

    def get_timbre_source(self, latent_cond, batch):
        """Resolve the configured timbre source from the current batch."""
        if self.timbre_input_key == "none":
            return None
        if self.timbre_input_key == "latent":
            return latent_cond
        if self.timbre_input_key == "latent_mean":
            return latent_cond.mean(dim=-1)
        if self.timbre_input_key == "latent_mean_std":
            latent_mean = latent_cond.mean(dim=-1)
            latent_std = latent_cond.std(dim=-1)
            return torch.cat([latent_mean, latent_std], dim=1)

        timbre_source = batch.get(self.timbre_input_key)
        if timbre_source is None:
            raise ValueError(
                f"Configured timbre_input_key='{self.timbre_input_key}' but batch does not contain it. "
                f"Available keys: {list(batch.keys())}"
            )
        return timbre_source

    def get_timbre_aug_bank(self, batch, timbre_source):
        """Resolve optional timbre-augmentation bank from the current batch."""
        if not getattr(self, "use_timbre_aug", False):
            return None
        aug_key = getattr(self, "timbre_aug_input_key", "z_timbre_aug")
        timbre_aug_bank = batch.get(aug_key)
        if timbre_aug_bank is None:
            if not getattr(self, "_warned_missing_timbre_aug", False):
                print(
                    f"Warning: use_timbre_aug=True but batch has no '{aug_key}'. "
                    "Falling back to the base timbre source."
                )
                self._warned_missing_timbre_aug = True
            return None
        if timbre_source is None:
            raise ValueError("use_timbre_aug=True requires a non-null timbre source.")
        if timbre_aug_bank.shape[0] != timbre_source.shape[0]:
            raise ValueError(
                f"Timbre aug batch mismatch: expected {timbre_source.shape[0]}, got {timbre_aug_bank.shape[0]}"
            )
        return timbre_aug_bank

    def _sample_timbre_aug_source(self, timbre_aug_bank, timbre_source):
        """Sample one timbre-augmentation example per batch item."""
        if timbre_aug_bank is None:
            return timbre_source
        if timbre_aug_bank.ndim == timbre_source.ndim:
            sampled = timbre_aug_bank
        elif timbre_aug_bank.ndim == (timbre_source.ndim + 1):
            if timbre_aug_bank.shape[2:] != timbre_source.shape[1:]:
                raise ValueError(
                    f"Timbre aug shape mismatch: bank {tuple(timbre_aug_bank.shape)} vs source {tuple(timbre_source.shape)}"
                )
            batch_size, bank_size = int(timbre_aug_bank.shape[0]), int(timbre_aug_bank.shape[1])
            if bank_size <= 0:
                raise ValueError(
                    f"Timbre aug bank must have at least one augmentation, got shape {tuple(timbre_aug_bank.shape)}"
                )
            choice_indices = torch.randint(
                low=0,
                high=bank_size,
                size=(batch_size,),
                device=timbre_aug_bank.device,
            )
            batch_indices = torch.arange(batch_size, device=timbre_aug_bank.device)
            sampled = timbre_aug_bank[batch_indices, choice_indices]
        else:
            raise ValueError(
                f"Unexpected timbre aug shape {tuple(timbre_aug_bank.shape)} for timbre source {tuple(timbre_source.shape)}"
            )

        prob = float(getattr(self, "timbre_aug_prob", 1.0))
        if prob >= 1.0:
            return sampled
        if prob <= 0.0:
            return timbre_source
        mask = (torch.rand(timbre_source.shape[0], device=timbre_source.device) < prob)
        mask = mask.view(-1, *([1] * (timbre_source.ndim - 1)))
        return torch.where(mask, sampled, timbre_source)
    
    def encode_timbre(self, timbre_source, return_aux=False):
        """Encode timbre conditioning from the configured timbre source."""
        aux = {}
        if timbre_source is None:
            if return_aux:
                return None, aux
            return None

        if self.encoder is not None:
            encoded = self.encoder(timbre_source)

            if isinstance(encoded, dict):
                if "z" not in encoded:
                    raise ValueError("Timbre encoder returned dict without required key 'z'.")
                cond = encoded["z"]
                aux = {k: v for k, v in encoded.items() if k != "z"}
            else:
                cond = encoded
        else:
            # No timbre encoder: use the provided timbre source directly.
            if timbre_source.ndim != 2:
                raise ValueError(
                    "encoder=None expects a precomputed timbre vector with shape [B, D]. "
                    f"Got {tuple(timbre_source.shape)}."
                )
            cond = timbre_source

        if cond.ndim != 2:
            raise ValueError(f"Timbre conditioning must be [B, D], got {tuple(cond.shape)}")

        if return_aux:
            return cond, aux
        return cond

    def forward(self, *args, **kwargs):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def training_step(self, batch, batch_idx=None):
        """Training step - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement training_step method")
    
    def validation_step(self, batch, batch_idx=None):
        """Validation step - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement validation_step method")
    
    def sample(self, *args, **kwargs):
        """Sample - to be implemented by subclasses.""" 
        raise NotImplementedError("Subclasses must implement sample method")

    def build_latent_codec(self, device=None):
        codec_device = device if device is not None else self.device
        return build_latent_codec(self.latent_codec, device=codec_device)

    def encode_audio_to_model_latent(self, waveform, codec=None, device=None):
        codec = self.build_latent_codec(device=device) if codec is None else codec
        target_device = device if device is not None else self.device
        return encode_audio_to_model_latent(
            codec,
            self.latent_codec,
            waveform,
            latent_dim=self.latent_dim,
            device=target_device,
        )

    def encode_audio_batch_to_model_latent(self, waveforms, codec=None, device=None):
        codec = self.build_latent_codec(device=device) if codec is None else codec
        target_device = device if device is not None else self.device
        return encode_audio_batch_to_model_latent(
            codec,
            self.latent_codec,
            waveforms,
            latent_dim=self.latent_dim,
            device=target_device,
        )

    def decode_model_latent_to_audio(self, latent_batch, codec=None):
        codec = self.build_latent_codec() if codec is None else codec
        return decode_model_latent_to_audio(
            codec,
            self.latent_codec,
            latent_batch,
            latent_dim=self.latent_dim,
        )
        
    def configure_optimizers(self, lr=1e-4):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _move_batch_to_device(self, batch):
        """Move batch tensors to device with consistent dtype."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    batch[k] = v.to(device=self.device, dtype=torch.float32)
                else:
                    # Preserve integer dtypes (e.g., labels)
                    batch[k] = v.to(device=self.device)
        return batch

    @gin.configurable
    def fit(self,
            dataloader=None,
            validloader=None,
            learning_rate=1e-4,
            batch_size=32,
            epochs=100,
            train_steps_per_epoch=None,
            val_steps_per_epoch=None,
            save_every=10,
            val_every=1,
            wandb_logging=True,
            experiment_dir=None,
            grad_clip_max_norm=None,
            ema_decay=0.0,
            resume_from_checkpoint=None,
            **kwargs):
        """
        Train the model with configurable parameters from gin.
        """
        import wandb
        from tqdm import tqdm

        opt = self.configure_optimizers(lr=learning_rate)
        
        # Optional EMA setup
        use_ema = (ema_decay is not None) and (float(ema_decay) > 0.0)
        ema_decay = float(ema_decay)
        ema_model = None
        if use_ema:
            ema_model = copy.deepcopy(self).eval().to(self.device)
            for p in ema_model.parameters():
                p.requires_grad_(False)

        def _ema_update(ema_m, online_m, decay):
            """Update EMA model parameters + buffers in-place."""
            with torch.no_grad():
                msd = online_m.state_dict()
                esd = ema_m.state_dict()
                for k in esd.keys():
                    v = esd[k]
                    if k in msd:
                        v_src = msd[k]
                        if v.dtype.is_floating_point:
                            v.copy_(decay * v + (1.0 - decay) * v_src)
                        else:
                            v.copy_(v_src)

        best_val = float("inf")
        # Global train step counter used by staged schedules in training_step.
        if not hasattr(self, "_global_step"):
            self._global_step = 0
        start_epoch = 1

        if resume_from_checkpoint is not None:
            checkpoint_path = os.fspath(resume_from_checkpoint)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict")
            if state_dict is None:
                raise KeyError(
                    f"Checkpoint {checkpoint_path} has no 'model_state_dict'; cannot resume training."
                )
            self.load_state_dict(state_dict)

            optimizer_state = ckpt.get("optimizer_state_dict")
            if optimizer_state is None:
                raise KeyError(
                    f"Checkpoint {checkpoint_path} has no 'optimizer_state_dict'; cannot resume optimizer state."
                )
            opt.load_state_dict(optimizer_state)

            ckpt_epoch = int(ckpt.get("epoch", 0))
            start_epoch = ckpt_epoch + 1

            if "global_step" in ckpt:
                self._global_step = int(ckpt["global_step"])
            else:
                if train_steps_per_epoch is not None:
                    steps_per_epoch = int(train_steps_per_epoch)
                else:
                    steps_per_epoch = len(dataloader)
                self._global_step = int(ckpt_epoch) * int(steps_per_epoch)
                print(
                    "Warning: checkpoint has no global_step; "
                    f"estimated global_step={self._global_step} from epoch and dataloader length."
                )

            saved_best = ckpt.get("best_val_loss", ckpt.get("val_loss"))
            if saved_best is not None:
                best_val = float(saved_best)

            if use_ema:
                ema_state = ckpt.get("ema_state_dict")
                if ema_state is not None:
                    ema_model.load_state_dict(ema_state)
                else:
                    print("Warning: checkpoint has no ema_state_dict; initialized EMA from current model.")

            print(
                f"Resumed training from {checkpoint_path} "
                f"(checkpoint_epoch={ckpt_epoch}, start_epoch={start_epoch}, "
                f"global_step={self._global_step})."
            )

        if start_epoch > int(epochs):
            print(
                f"Checkpoint is already at epoch {start_epoch - 1}, "
                f"which is >= requested epochs={epochs}. Nothing to train."
            )
            return {
                "final_train_loss": None,
                "best_val_loss": best_val,
            }

        for epoch in tqdm(range(start_epoch, epochs + 1), desc="Training"):
            # Train epoch
            self.train()
            tr_total_sum, tr_diff_sum, tr_triplet_sum, tr_pitch_sum = 0.0, 0.0, 0.0, 0.0
            tr_has_triplet = False
            tr_has_pitch = False
            tr_n = 0
            # Inform about step caps to avoid confusion with dataloader length
            if train_steps_per_epoch is not None or val_steps_per_epoch is not None:
                print(f"Limiting epoch: train_steps={train_steps_per_epoch}, val_steps={val_steps_per_epoch}")

            train_total = int(train_steps_per_epoch) if train_steps_per_epoch is not None else None
            for step, batch in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch {epoch}", total=train_total), start=1):
                # Move to device and ensure consistent dtype
                batch = self._move_batch_to_device(batch)
                
                # Training step - get full loss dict
                opt.zero_grad(set_to_none=True)
                self._global_step = int(self._global_step)
                loss_dict = self.training_step(batch, None)
                loss = loss_dict["total_loss"]
                
                # Backward pass
                loss.backward()
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip_max_norm))
                opt.step()
                
                # Update EMA after optimizer step
                if use_ema:
                    _ema_update(ema_model, self, ema_decay)
                
                # Accumulate losses
                tr_total_sum += float(loss_dict["total_loss"].detach())
                tr_diff_sum += float(loss_dict["diff_loss"].detach())
                if "timbre_triplet_loss" in loss_dict:
                    tr_triplet_sum += float(loss_dict["timbre_triplet_loss"].detach())
                    tr_has_triplet = True
                if "structure_pitch_loss" in loss_dict:
                    tr_pitch_sum += float(loss_dict["structure_pitch_loss"].detach())
                    tr_has_pitch = True
                tr_n += 1
                self._global_step += 1
                # Optional cap on training steps per epoch
                if train_steps_per_epoch is not None and step >= int(train_steps_per_epoch):
                    break
            
            tr_total_loss = tr_total_sum / max(1, tr_n)
            tr_diff_loss = tr_diff_sum / max(1, tr_n)
            tr_triplet_loss = tr_triplet_sum / max(1, tr_n) if tr_has_triplet else None
            tr_pitch_loss = tr_pitch_sum / max(1, tr_n) if tr_has_pitch else None

            val_total_loss, val_diff_loss, val_triplet_loss, val_pitch_loss = None, None, None, None
            if validloader is not None and (epoch % val_every == 0):
                self.eval()

                val_total_sum, val_diff_sum, val_triplet_sum, val_pitch_sum = 0.0, 0.0, 0.0, 0.0
                val_has_triplet = False
                val_has_pitch = False
                vn = 0
                with torch.no_grad():
                    val_total = int(val_steps_per_epoch) if val_steps_per_epoch is not None else None
                    for vstep, batch in enumerate(tqdm(validloader, leave=False, desc=f"Val {epoch}", total=val_total), start=1):
                        batch = self._move_batch_to_device(batch)
                        loss_dict = self.validation_step(batch, None)
                        val_total_sum += float(loss_dict["total_loss"])
                        val_diff_sum += float(loss_dict["diff_loss"])
                        if "timbre_triplet_loss" in loss_dict:
                            val_triplet_sum += float(loss_dict["timbre_triplet_loss"])
                            val_has_triplet = True
                        if "structure_pitch_loss" in loss_dict:
                            val_pitch_sum += float(loss_dict["structure_pitch_loss"])
                            val_has_pitch = True
                        vn += 1
                        # Optional cap on validation steps per epoch
                        if val_steps_per_epoch is not None and vstep >= int(val_steps_per_epoch):
                            break
                
                val_total_loss = val_total_sum / max(1, vn)
                val_diff_loss = val_diff_sum / max(1, vn)
                val_triplet_loss = val_triplet_sum / max(1, vn) if val_has_triplet else None
                val_pitch_loss = val_pitch_sum / max(1, vn) if val_has_pitch else None

            if wandb_logging:
                logd = {
                    "epoch": epoch,
                    "train/total_loss": tr_total_loss,
                    "train/diff_loss": tr_diff_loss
                }
                if tr_triplet_loss is not None:
                    logd["train/timbre_triplet_loss"] = tr_triplet_loss
                if tr_pitch_loss is not None:
                    logd["train/structure_pitch_loss"] = tr_pitch_loss
                if val_total_loss is not None:
                    logd["val/total_loss"] = val_total_loss
                    logd["val/diff_loss"] = val_diff_loss
                if val_triplet_loss is not None:
                    logd["val/timbre_triplet_loss"] = val_triplet_loss
                if val_pitch_loss is not None:
                    logd["val/structure_pitch_loss"] = val_pitch_loss
                wandb.log(logd)

            if experiment_dir is not None:
                os.makedirs(experiment_dir, exist_ok=True)
                config_path = os.path.join(experiment_dir, "config.gin")
                if epoch == start_epoch and not os.path.exists(config_path):
                    operative_config = gin.operative_config_str()
                    # Gin can emit transient shortened selectors for this module
                    # (for example ``musdis2.RectifiedFlow``) in operative configs.
                    # Normalize them so saved run configs remain parseable
                    # through the canonical ``mld.pipeline.model`` path.
                    operative_config = operative_config.replace(
                        "musdis2.",
                        "mld.pipeline.model.",
                    )
                    model_class_line = (
                        f"model_class = @{self.__class__.__module__}.{self.__class__.__name__}\n"
                    )
                    if "model_class =" not in operative_config:
                        operative_config = model_class_line + operative_config
                    with open(config_path, "w") as f:
                        f.write(operative_config)

                save_ckpt = (epoch % save_every == 0)
                save_best = (val_total_loss is not None and val_total_loss < best_val)
                if save_ckpt or save_best:
                    state = self.state_dict()
                    ckpt = {
                        "epoch": epoch,
                        "model_state_dict": state,
                        "optimizer_state_dict": opt.state_dict(),
                        "global_step": int(getattr(self, "_global_step", 0)),
                        "train_loss": tr_total_loss,
                        "val_loss": val_total_loss,
                        "best_val_loss": min(
                            float(best_val),
                            float(val_total_loss) if val_total_loss is not None else float("inf"),
                        ),
                        "fit_config": {
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "ema_decay": float(ema_decay),
                            "grad_clip_max_norm": float(grad_clip_max_norm) if grad_clip_max_norm is not None else None,
                        }
                    }
                    if use_ema:
                        ckpt["ema_state_dict"] = ema_model.state_dict()
                    if save_ckpt:
                        torch.save(ckpt, os.path.join(experiment_dir, f"checkpoint_epoch_{epoch}.pt"))
                    if save_best:
                        best_val = val_total_loss
                        torch.save(ckpt, os.path.join(experiment_dir, "best_model.pt"))
        return {"final_train_loss": tr_total_loss, "best_val_loss": best_val}

@gin.configurable
class RectifiedFlow(Base):
    """Rectified flow model with timbre/structure conditioning and auxiliary losses."""

    def __init__(self,
                 latent_normalization="sample",
                 timbre_input_key="latent",
                 group_key=("track", "stem"),
                 time_cond_drop_rate=0.0,
                 cond_drop_rate=0.0,
                 use_cross_timbre_cond=False,
                 use_timbre_aug=False,
                 timbre_aug_input_key="z_timbre_aug",
                 timbre_aug_prob=1.0,
                 timbre_aug_train_only=True,
                 use_struct_aug=False,
                 use_struct_aug_time_cond=True,
                 use_timbre_triplet=False,
                 timbre_triplet_weight=0.0,
                 timbre_triplet_margin=0.2,
                 use_structure_pitch_loss=False,
                 structure_pitch_head=None,
                 structure_pitch_weight=0.0,
                 structure_pitch_pos_weight=None,
                 detach_structure_for_pitch_loss=False,
                 log_structure_pitch_debug=False,
                 use_staged_time_cond_drop=False,
                 stage2_start_step=0,
                 time_cond_drop_rate_stage2=0.0,
                 freeze_timbre_encoder_in_stage1=False,
                 num_flow_steps=50,
                 flow_solver="euler",
                 flow_guidance_timbre=1.0,
                 flow_guidance_structure=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        num_flow_steps = int(num_flow_steps)
        if num_flow_steps <= 0:
            raise ValueError(f"num_flow_steps must be > 0, got {num_flow_steps}")

        self.latent_normalization = str(latent_normalization).lower()
        self.timbre_input_key = str(timbre_input_key)
        self.group_key = _normalize_group_key(group_key)
        self.time_cond_drop_rate = float(time_cond_drop_rate)
        self.cond_drop_rate = float(cond_drop_rate)
        self.use_cross_timbre_cond = bool(use_cross_timbre_cond)
        self.use_timbre_aug = bool(use_timbre_aug)
        self.timbre_aug_input_key = str(timbre_aug_input_key)
        self.timbre_aug_prob = float(timbre_aug_prob)
        self.timbre_aug_train_only = bool(timbre_aug_train_only)
        self.use_struct_aug = bool(use_struct_aug)
        self.use_struct_aug_time_cond = bool(use_struct_aug_time_cond)
        self.use_timbre_triplet = bool(use_timbre_triplet)
        self.timbre_triplet_weight = float(timbre_triplet_weight)
        self.timbre_triplet_margin = float(timbre_triplet_margin)
        self.use_structure_pitch_loss = bool(use_structure_pitch_loss)
        self.structure_pitch_head = structure_pitch_head
        self.structure_pitch_weight = float(structure_pitch_weight)
        self.structure_pitch_pos_weight = (
            None if structure_pitch_pos_weight is None else float(structure_pitch_pos_weight)
        )
        self.detach_structure_for_pitch_loss = bool(detach_structure_for_pitch_loss)
        self.log_structure_pitch_debug = bool(log_structure_pitch_debug)
        self.use_staged_time_cond_drop = bool(use_staged_time_cond_drop)
        self.stage2_start_step = max(0, int(stage2_start_step))
        self.time_cond_drop_rate_stage2 = float(time_cond_drop_rate_stage2)
        self.freeze_timbre_encoder_in_stage1 = bool(freeze_timbre_encoder_in_stage1)

        self.num_flow_steps = num_flow_steps
        self.flow_solver = str(flow_solver).lower()
        self.flow_guidance_timbre = float(flow_guidance_timbre)
        self.flow_guidance_structure = float(flow_guidance_structure)
        self._stage1_timbre_frozen = None
        self._warned_missing_group_ids = False
        self._warned_missing_timbre_aug = False

        if self.latent_normalization not in {"sample", "none"}:
            raise ValueError(
                "latent_normalization must be one of {sample, none}, "
                f"got {latent_normalization!r}."
            )
        if self.use_timbre_triplet:
            if self.timbre_triplet_weight <= 0.0:
                raise ValueError("use_timbre_triplet=True requires timbre_triplet_weight > 0.")
            if self.timbre_triplet_margin <= 0.0:
                raise ValueError("use_timbre_triplet=True requires timbre_triplet_margin > 0.")
            if self.encoder is None:
                raise ValueError("use_timbre_triplet=True requires a timbre encoder (encoder).")
            if self.timbre_input_key != "latent":
                raise ValueError(
                    "use_timbre_triplet=True currently requires timbre_input_key=latent "
                    "so the structure augmentation can act as the negative."
                )
            if not self.use_struct_aug:
                raise ValueError("use_timbre_triplet=True requires use_struct_aug=True.")
        if self.use_timbre_aug:
            if self.timbre_input_key != "latent":
                raise ValueError(
                    "use_timbre_aug=True currently requires timbre_input_key=latent."
                )
            if self.timbre_aug_prob < 0.0 or self.timbre_aug_prob > 1.0:
                raise ValueError(
                    f"timbre_aug_prob must be in [0, 1], got {self.timbre_aug_prob}."
                )
        if self.timbre_input_key == "none":
            if self.encoder is not None:
                raise ValueError("timbre_input_key=none requires encoder=None.")
            if self.use_cross_timbre_cond:
                raise ValueError("timbre_input_key=none is incompatible with use_cross_timbre_cond=True.")
            if self.cond_drop_rate > 0.0:
                raise ValueError("timbre_input_key=none is incompatible with cond_drop_rate > 0.")
            if self.use_timbre_triplet:
                raise ValueError("timbre_input_key=none is incompatible with use_timbre_triplet=True.")
        if self.use_structure_pitch_loss or self.log_structure_pitch_debug:
            if self.encoder_time is None:
                raise ValueError(
                    "structure pitch supervision/debug requires a structure encoder (encoder_time)."
                )
            if self.structure_pitch_head is None:
                raise ValueError(
                    "structure pitch supervision/debug requires a pitch head module (structure_pitch_head)."
                )
            if self.structure_pitch_pos_weight is not None and self.structure_pitch_pos_weight <= 0.0:
                raise ValueError("structure_pitch_pos_weight must be > 0 when provided.")
        if self.use_structure_pitch_loss and self.structure_pitch_weight <= 0.0:
            raise ValueError("use_structure_pitch_loss=True requires structure_pitch_weight > 0.")
        if self.flow_solver not in {"euler", "heun"}:
            raise ValueError(
                f"flow_solver must be one of {{euler, heun}}, got {flow_solver!r}"
            )

        self.to(self._target_device)

    @staticmethod
    def _extract_group_ids(batch_obj, group_key, device):
        if "group_ids" in batch_obj:
            return batch_obj["group_ids"].to(device=device, dtype=torch.long)
        metadatas = batch_obj.get("metadata")
        if not isinstance(metadatas, (list, tuple)):
            return None

        raw_groups = []
        for meta in metadatas:
            if not isinstance(meta, dict):
                return None
            try:
                raw_groups.append(_resolve_metadata_group(meta, group_key))
            except ValueError:
                return None

        group_to_idx = {}
        group_ids = []
        for group in raw_groups:
            if group not in group_to_idx:
                group_to_idx[group] = len(group_to_idx)
            group_ids.append(group_to_idx[group])
        return torch.tensor(group_ids, device=device, dtype=torch.long)

    @staticmethod
    def _downsample_midi_roll(midi_roll, target_steps):
        if midi_roll.ndim != 3:
            raise ValueError(f"midi_roll must be [B, P, T], got {tuple(midi_roll.shape)}")
        target_steps = int(target_steps)
        if target_steps <= 0:
            raise ValueError(f"target_steps must be > 0, got {target_steps}")
        if midi_roll.shape[-1] == target_steps:
            return midi_roll
        if midi_roll.shape[-1] > target_steps:
            return F.adaptive_max_pool1d(midi_roll, output_size=target_steps)
        return F.interpolate(midi_roll, size=target_steps, mode="nearest")

    @staticmethod
    def _log_structure_pitch_debug(pitch_logits, midi_target, global_step):
        pitch_prob = torch.sigmoid(pitch_logits)
        pos_mask = midi_target > 0.5
        neg_mask = ~pos_mask
        eps = 1e-8

        target_pos_rate = midi_target.mean()

        if pos_mask.any():
            pos_probs = pitch_prob[pos_mask]
            pred_on_pos_mean = pos_probs.mean()
            pred_on_pos_median = pos_probs.median()
            pred_on_pos_p25 = torch.quantile(pos_probs, 0.25)
        else:
            pred_on_pos_mean = torch.tensor(0.0, device=pitch_logits.device)
            pred_on_pos_median = torch.tensor(0.0, device=pitch_logits.device)
            pred_on_pos_p25 = torch.tensor(0.0, device=pitch_logits.device)

        if neg_mask.any():
            neg_probs = pitch_prob[neg_mask]
            pred_on_neg = neg_probs.mean()
        else:
            neg_probs = None
            pred_on_neg = torch.tensor(0.0, device=pitch_logits.device)

        pred_pos_03 = pitch_prob > 0.3
        pred_pos_05 = pitch_prob > 0.5
        tp_03 = (pred_pos_03 & pos_mask).sum().float()
        tp_05 = (pred_pos_05 & pos_mask).sum().float()
        pred_count_03 = pred_pos_03.sum().float()
        pred_count_05 = pred_pos_05.sum().float()
        pos_count = pos_mask.sum().float()

        recall_03 = tp_03 / (pos_count + eps)
        recall_05 = tp_05 / (pos_count + eps)
        precision_03 = tp_03 / (pred_count_03 + eps)
        precision_05 = tp_05 / (pred_count_05 + eps)

        loss_per_elem = F.binary_cross_entropy_with_logits(
            pitch_logits,
            midi_target,
            reduction="none",
        )

        if pos_mask.any():
            pos_losses = loss_per_elem[pos_mask]
            bce_pos = pos_losses.mean()
            pos_loss_p95 = torch.quantile(pos_losses, 0.95)
            pos_loss_max = pos_losses.max()
        else:
            bce_pos = torch.tensor(0.0, device=pitch_logits.device)
            pos_loss_p95 = torch.tensor(0.0, device=pitch_logits.device)
            pos_loss_max = torch.tensor(0.0, device=pitch_logits.device)

        if neg_mask.any():
            neg_losses = loss_per_elem[neg_mask]
            bce_neg = neg_losses.mean()
            if pos_mask.any():
                topk_k = int(min(int(pos_mask.sum().item()), int(neg_mask.sum().item())))
            else:
                topk_k = int(min(64, int(neg_mask.sum().item())))
            topk_k = max(1, topk_k)
            topk_neg_probs = torch.topk(neg_probs, k=topk_k).values
            topk_fp_rate_03 = (topk_neg_probs > 0.3).float().mean()
            topk_fp_rate_05 = (topk_neg_probs > 0.5).float().mean()
        else:
            bce_neg = torch.tensor(0.0, device=pitch_logits.device)
            topk_k = 0
            topk_fp_rate_03 = torch.tensor(0.0, device=pitch_logits.device)
            topk_fp_rate_05 = torch.tensor(0.0, device=pitch_logits.device)

        print(
            f"[pitch dbg] step={global_step} "
            f"target_pos_rate={target_pos_rate.item():.4f} "
            f"pred_on_pos={pred_on_pos_mean.item():.4f} "
            f"pred_on_neg={pred_on_neg.item():.4f} "
            f"recall@0.3={recall_03.item():.4f} "
            f"recall@0.5={recall_05.item():.4f} "
            f"precision@0.3={precision_03.item():.4f} "
            f"precision@0.5={precision_05.item():.4f} "
            f"topk_fp_rate@0.3(k={topk_k})={topk_fp_rate_03.item():.4f} "
            f"topk_fp_rate@0.5(k={topk_k})={topk_fp_rate_05.item():.4f} "
            f"bce_pos={bce_pos.item():.4f} "
            f"bce_neg={bce_neg.item():.4f} "
            f"pos_loss_p95={pos_loss_p95.item():.4f} "
            f"pos_loss_max={pos_loss_max.item():.4f} "
            f"pred_on_pos_median={pred_on_pos_median.item():.4f} "
            f"pred_on_pos_p25={pred_on_pos_p25.item():.4f}"
        )

    def _should_run_structure_pitch_probe(self, global_step):
        return (
            self.encoder_time is not None
            and self.structure_pitch_head is not None
            and (self.use_structure_pitch_loss or self.log_structure_pitch_debug)
        )

    def _compute_structure_pitch_probe(self, batch, time_cond_full, device, dtype, global_step):
        structure_pitch_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if not self._should_run_structure_pitch_probe(global_step):
            return structure_pitch_loss

        if "midi_roll" not in batch:
            raise ValueError(
                "structure pitch supervision/debug requires batch[midi_roll]. "
                "Rebuild dataset with --save_midi."
            )
        midi_roll = batch["midi_roll"]
        if midi_roll.ndim != 3:
            raise ValueError(f"midi_roll must have shape [B, P, T], got {tuple(midi_roll.shape)}")

        pitch_input = time_cond_full.detach() if self.detach_structure_for_pitch_loss else time_cond_full
        pitch_logits = self.structure_pitch_head(pitch_input)

        if pitch_logits.ndim == 2:
            pitch_logits = pitch_logits.unsqueeze(-1)
        if pitch_logits.ndim != 3:
            raise ValueError(
                f"structure_pitch_head output must be [B, P, T] or [B, P], got {tuple(pitch_logits.shape)}"
            )

        if pitch_logits.shape[1] != midi_roll.shape[1] and pitch_logits.shape[2] == midi_roll.shape[1]:
            pitch_logits = pitch_logits.transpose(1, 2)
        if pitch_logits.shape[1] != midi_roll.shape[1]:
            raise ValueError(
                f"Pitch head channels mismatch: pred {tuple(pitch_logits.shape)} vs midi_roll {tuple(midi_roll.shape)}"
            )

        midi_target = self._downsample_midi_roll(
            midi_roll.to(dtype=pitch_logits.dtype),
            target_steps=int(pitch_logits.shape[-1]),
        )

        if self.structure_pitch_pos_weight is None:
            structure_pitch_loss = F.binary_cross_entropy_with_logits(pitch_logits, midi_target)
        else:
            pos_weight = torch.tensor(
                self.structure_pitch_pos_weight,
                device=pitch_logits.device,
                dtype=pitch_logits.dtype,
            )
            structure_pitch_loss = F.binary_cross_entropy_with_logits(
                pitch_logits, midi_target, pos_weight=pos_weight
            )

        if self.training and (global_step % 100 == 0):
            self._log_structure_pitch_debug(pitch_logits, midi_target, global_step)

        return structure_pitch_loss

    def _current_time_cond_drop_rate(self, global_step):
        base = float(self.time_cond_drop_rate)
        if not self.use_staged_time_cond_drop:
            return base

        if int(global_step) < int(self.stage2_start_step):
            return base
        return float(self.time_cond_drop_rate_stage2)

    @staticmethod
    def _sample_same_group_swap_indices(group_ids):
        if group_ids is None:
            return None

        bsz = group_ids.size(0)
        idx_swap = torch.arange(bsz, device=group_ids.device)
        for group_id in group_ids.unique():
            idxs = (group_ids == group_id).nonzero(as_tuple=True)[0]
            if idxs.numel() <= 1:
                continue
            shuffled = idxs[torch.randperm(idxs.numel(), device=group_ids.device)]
            shift = int(torch.randint(1, idxs.numel(), size=(1,), device=group_ids.device).item())
            idx_swap[idxs] = shuffled.roll(shifts=shift, dims=0)
        return idx_swap

    def _maybe_apply_stage1_timbre_freeze(self, global_step):
        if not self.training or not self.freeze_timbre_encoder_in_stage1 or self.encoder is None:
            return

        should_freeze = int(global_step) < int(self.stage2_start_step)
        if self._stage1_timbre_frozen is not None and self._stage1_timbre_frozen == should_freeze:
            return

        for p in self.encoder.parameters():
            p.requires_grad_(not should_freeze)
        self._stage1_timbre_frozen = should_freeze

        if should_freeze:
            print(f"Stage 1 active at global_step={global_step}: timbre encoder frozen.")
        else:
            print(f"Stage 2 reached at global_step={global_step}: timbre encoder unfrozen.")

    def _normalize_latents(self, latents):
        if self.latent_normalization == "none":
            return latents
        return (latents - latents.mean(dim=(1, 2), keepdim=True)) / (
            latents.std(dim=(1, 2), keepdim=True) + 1e-6
        )

    def forward(self, latents, timesteps, time_cond=None, cond=None):
        return self.net(latents, timesteps=timesteps, cond=cond, time_cond=time_cond)

    def _sample_flow_t(self, batch_size, device, dtype):
        return torch.rand(batch_size, device=device, dtype=dtype)

    def compute_loss(self, x_t, x0, x1, t, time_cond=None, cond=None):
        pred = self.net(x_t, timesteps=t, cond=cond, time_cond=time_cond)
        target = x1 - x0
        loss = F.mse_loss(pred, target)
        x1_pred = x_t + (1.0 - t.view(-1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)) * pred

        return {
            "total_loss": loss,
            "diff_loss": loss,
            "pred_clean_latent": x1_pred,
        }

    def training_step(self, batch, batch_idx=None):
        global_step = int(getattr(self, "_global_step", 0))
        self._maybe_apply_stage1_timbre_freeze(global_step)
        effective_time_cond_drop_rate = float(self.time_cond_drop_rate)
        if self.training:
            effective_time_cond_drop_rate = self._current_time_cond_drop_rate(global_step)

        latents = batch["latent"]
        if latents.ndim != 3:
            raise ValueError(
                f"RectifiedFlow expects latents with shape [B, C, T], got {tuple(latents.shape)}"
            )

        latents = self._normalize_latents(latents)
        current_batch_size = latents.shape[0]

        latent_time = latents
        latent_cond = latents

        timbre_source = self.get_timbre_source(latent_cond, batch)
        if timbre_source is not None and timbre_source.shape[0] != current_batch_size:
            raise ValueError(
                f"Timbre source batch mismatch: expected {current_batch_size}, got {timbre_source.shape[0]}"
            )
        timbre_aug_bank = None
        if timbre_source is not None and (self.training or not self.timbre_aug_train_only):
            timbre_aug_bank = self.get_timbre_aug_bank(batch, timbre_source)

        need_struct_aug_latent = self.use_struct_aug and (
            (self.training and self.use_struct_aug_time_cond) or self.use_timbre_triplet
        )
        latent_struct_aug = None
        if need_struct_aug_latent:
            latent_struct_aug = self._sample_struct_aug_latent(batch, latents)
            latent_struct_aug = self._normalize_latents(latent_struct_aug)

        if self.use_struct_aug and self.use_struct_aug_time_cond and self.training:
            latent_time = latent_struct_aug

        group_ids = self._extract_group_ids(batch, self.group_key, device=latents.device)

        timbre_source_for_timbre = timbre_source
        timbre_source_for_triplet = timbre_source
        timbre_aug_bank_for_timbre = timbre_aug_bank
        swap_fraction = 0.0
        if self.use_cross_timbre_cond and self.training:
            if group_ids is None:
                if not self._warned_missing_group_ids:
                    print(
                        "Warning: use_cross_timbre_cond=True but no grouped batch ids or "
                        f"metadata {self.group_key} were found. Skipping timbre input swap."
                    )
                    self._warned_missing_group_ids = True
            else:
                idx_swap = self._sample_same_group_swap_indices(group_ids)
                timbre_source_for_timbre = timbre_source[idx_swap]
                timbre_source_for_triplet = timbre_source_for_timbre
                if timbre_aug_bank_for_timbre is not None:
                    timbre_aug_bank_for_timbre = timbre_aug_bank_for_timbre[idx_swap]
                swap_fraction = float(
                    (idx_swap != torch.arange(group_ids.size(0), device=group_ids.device))
                    .float()
                    .mean()
                    .item()
                )
        elif self.use_timbre_triplet and self.use_cross_timbre_cond:
            if group_ids is None:
                if not self._warned_missing_group_ids:
                    print(
                        "Warning: use_timbre_triplet=True but no grouped batch ids or "
                        f"metadata {self.group_key} were found for validation triplet pairs. "
                        "Falling back to original timbre source as positive."
                    )
                    self._warned_missing_group_ids = True
            else:
                idx_swap = self._sample_same_group_swap_indices(group_ids)
                timbre_source_for_triplet = timbre_source[idx_swap]
                swap_fraction = float(
                    (idx_swap != torch.arange(group_ids.size(0), device=group_ids.device))
                    .float()
                    .mean()
                    .item()
                )

        if timbre_aug_bank_for_timbre is not None:
            timbre_source_for_timbre = self._sample_timbre_aug_source(
                timbre_aug_bank_for_timbre,
                timbre_source_for_timbre,
            )

        time_cond_full = self.encode_structure(latent_time)
        cond = self.encode_timbre(timbre_source_for_timbre)

        cond_unswapped = None
        if self.use_cross_timbre_cond and self.use_timbre_triplet:
            cond_unswapped = self.encode_timbre(timbre_source)

        structure_pitch_loss = self._compute_structure_pitch_probe(
            batch,
            time_cond_full,
            device=latents.device,
            dtype=latents.dtype,
            global_step=global_step,
        )

        timbre_triplet_loss = torch.tensor(0.0, device=latents.device)
        if self.use_timbre_triplet:
            anchor = cond_unswapped if cond_unswapped is not None else self.encode_timbre(timbre_source)
            positive = cond if self.training else self.encode_timbre(timbre_source_for_triplet)
            if latent_struct_aug is None:
                raise ValueError(
                    "use_timbre_triplet=True requires latent_struct_aug, "
                    "but no structure augmentation latent was available."
                )
            negative = self.encode_timbre(latent_struct_aug)

            anchor = F.normalize(anchor.to(device=latents.device, dtype=latents.dtype), dim=1)
            positive = F.normalize(positive.to(device=latents.device, dtype=latents.dtype), dim=1)
            negative = F.normalize(negative.to(device=latents.device, dtype=latents.dtype), dim=1)

            timbre_triplet_loss = F.triplet_margin_loss(
                anchor,
                positive,
                negative,
                margin=self.timbre_triplet_margin,
                p=2.0,
            )

            if self.training and global_step % 100 == 0:
                with torch.no_grad():
                    pos_dist = torch.norm(anchor - positive, dim=1)
                    neg_dist = torch.norm(anchor - negative, dim=1)
                    violation_rate = ((pos_dist - neg_dist + self.timbre_triplet_margin) > 0).float().mean()
                    print(
                        f"[triplet dbg] step={global_step} "
                        f"loss={timbre_triplet_loss.item():.6f} "
                        f"pos_dist={pos_dist.mean().item():.4f} "
                        f"neg_dist={neg_dist.mean().item():.4f} "
                        f"margin={self.timbre_triplet_margin:.3f} "
                        f"violation_rate={violation_rate.item():.4f} "
                        f"swap_fraction={swap_fraction:.4f}"
                    )

        cond_for_model = cond
        time_cond_for_model = time_cond_full

        time_cond = None
        if self.training:
            batch_size = current_batch_size
            device = latents.device

            if self.encoder_time is not None and effective_time_cond_drop_rate > 0.0 and (time_cond_for_model is not None):
                drop_mask = (torch.rand(batch_size, device=device) < effective_time_cond_drop_rate)
                drop_mask = drop_mask.view(batch_size, 1, 1)
                dropped = torch.full_like(time_cond_for_model, self.drop_value)
                time_cond = torch.where(drop_mask, dropped, time_cond_for_model)
            else:
                time_cond = time_cond_for_model

            if self.cond_drop_rate > 0.0 and (cond_for_model is not None):
                drop_mask = (torch.rand(batch_size, device=device) < self.cond_drop_rate)
                drop_mask = drop_mask.view(batch_size, 1)
                dropped = torch.full_like(cond_for_model, self.drop_value)
                cond = torch.where(drop_mask, dropped, cond_for_model)
            else:
                cond = cond_for_model
        else:
            time_cond = time_cond_for_model
            cond = cond_for_model

        t = self._sample_flow_t(current_batch_size, device=latents.device, dtype=latents.dtype)
        t_view = t.view(current_batch_size, 1, 1)
        base_noise = torch.randn_like(latents)
        interpolant = (1.0 - t_view) * base_noise + t_view * latents

        loss_dict = self.compute_loss(
            interpolant,
            base_noise,
            latents,
            t,
            time_cond=time_cond,
            cond=cond,
        )
        total_loss = loss_dict["total_loss"]
        if self.use_timbre_triplet:
            total_loss = total_loss + self.timbre_triplet_weight * timbre_triplet_loss
        if self.use_structure_pitch_loss:
            total_loss = total_loss + self.structure_pitch_weight * structure_pitch_loss
        loss_dict["total_loss"] = total_loss
        if self.use_timbre_triplet:
            loss_dict["timbre_triplet_loss"] = timbre_triplet_loss
            loss_dict["timbre_triplet_weight"] = torch.tensor(
                self.timbre_triplet_weight, device=latents.device, dtype=latents.dtype
            )
        if self.use_structure_pitch_loss:
            loss_dict["structure_pitch_loss"] = structure_pitch_loss

        return loss_dict

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx)

    @torch.no_grad()
    def _guided_flow_velocity(self, x, t_batch, cond, time_cond, guidance_timbre=None, guidance_structure=None):
        guidance_timbre = float(
            self.flow_guidance_timbre if guidance_timbre is None else guidance_timbre
        )
        guidance_structure = float(
            self.flow_guidance_structure if guidance_structure is None else guidance_structure
        )

        has_timbre = cond is not None
        has_structure = time_cond is not None

        if not has_timbre and not has_structure:
            return self.net(x, timesteps=t_batch, cond=None, time_cond=None)

        if has_timbre and has_structure and abs(guidance_timbre - 1.0) < 1e-8 and abs(guidance_structure - 1.0) < 1e-8:
            return self.net(x, timesteps=t_batch, cond=cond, time_cond=time_cond)

        if has_timbre and has_structure:
            x_full = torch.cat([x, x, x], dim=0)
            t_full = torch.cat([t_batch, t_batch, t_batch], dim=0)
            cond_full = torch.cat(
                [
                    cond,
                    torch.full_like(cond, self.drop_value),
                    torch.full_like(cond, self.drop_value),
                ],
                dim=0,
            )
            time_full = torch.cat(
                [
                    time_cond,
                    time_cond,
                    torch.full_like(time_cond, self.drop_value),
                ],
                dim=0,
            )
            velocity = self.net(x_full, timesteps=t_full, cond=cond_full, time_cond=time_full)
            v_full, v_struct, v_none = velocity.chunk(3, dim=0)
            return (
                v_none
                + guidance_structure * (v_struct - v_none)
                + guidance_timbre * (v_full - v_struct)
            )

        if has_timbre:
            x_full = torch.cat([x, x], dim=0)
            t_full = torch.cat([t_batch, t_batch], dim=0)
            cond_full = torch.cat([cond, torch.full_like(cond, self.drop_value)], dim=0)
            velocity = self.net(x_full, timesteps=t_full, cond=cond_full, time_cond=None)
            v_full, v_none = velocity.chunk(2, dim=0)
            return v_none + guidance_timbre * (v_full - v_none)

        x_full = torch.cat([x, x], dim=0)
        t_full = torch.cat([t_batch, t_batch], dim=0)
        time_full = torch.cat([time_cond, torch.full_like(time_cond, self.drop_value)], dim=0)
        velocity = self.net(x_full, timesteps=t_full, cond=None, time_cond=time_full)
        v_full, v_none = velocity.chunk(2, dim=0)
        return v_none + guidance_structure * (v_full - v_none)

    @torch.no_grad()
    def sample_with_embeddings(
        self,
        structure_embedding,
        timbre_embedding,
        latent_shape,
        num_inference_steps=None,
        solver=None,
        guidance_timbre=None,
        guidance_structure=None,
        seed=None,
    ):
        if num_inference_steps is None:
            num_inference_steps = self.num_flow_steps
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")

        solver = self.flow_solver if solver is None else str(solver).lower()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"Unknown flow solver {solver}. Use euler or heun.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        x = torch.randn(latent_shape, device=self.device, dtype=torch.float32, generator=generator)
        batch_size = x.shape[0]
        dt = 1.0 / float(num_inference_steps)
        t_values = torch.linspace(
            0.0,
            1.0,
            num_inference_steps + 1,
            device=x.device,
            dtype=x.dtype,
        )[:-1]

        for t_scalar in t_values:
            t_batch = torch.full((batch_size,), float(t_scalar.item()), device=x.device, dtype=x.dtype)
            velocity = self._guided_flow_velocity(
                x,
                t_batch,
                cond=timbre_embedding,
                time_cond=structure_embedding,
                guidance_timbre=guidance_timbre,
                guidance_structure=guidance_structure,
            )
            x_euler = x + dt * velocity

            if solver == "heun":
                t_next_scalar = min(float(t_scalar.item() + dt), 1.0)
                t_next = torch.full((batch_size,), t_next_scalar, device=x.device, dtype=x.dtype)
                velocity_next = self._guided_flow_velocity(
                    x_euler,
                    t_next,
                    cond=timbre_embedding,
                    time_cond=structure_embedding,
                    guidance_timbre=guidance_timbre,
                    guidance_structure=guidance_structure,
                )
                x = x + 0.5 * dt * (velocity + velocity_next)
            else:
                x = x_euler

        return x

    @torch.no_grad()
    def sample(self, time_source, cond_source, num_inference_steps=None, deterministic=False):
        self.eval()

        if num_inference_steps is None:
            num_inference_steps = self.num_flow_steps
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")

        time_cond = None
        if self.encoder_time is not None and time_source is not None:
            time_cond = self.encode_structure(time_source)

        cond = self.encode_timbre(cond_source)

        if time_source is not None:
            latent_shape = tuple(time_source.shape)
        elif cond_source is not None:
            latent_shape = (cond_source.shape[0], self.latent_dim, 1024)
        else:
            raise ValueError("RectifiedFlow.sample requires at least one of time_source or cond_source.")

        return self.sample_with_embeddings(
            structure_embedding=time_cond,
            timbre_embedding=cond,
            latent_shape=latent_shape,
            num_inference_steps=num_inference_steps,
            solver=self.flow_solver,
            guidance_timbre=self.flow_guidance_timbre,
            guidance_structure=self.flow_guidance_structure,
            seed=None,
        )
