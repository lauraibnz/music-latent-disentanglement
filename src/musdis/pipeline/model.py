import os
import torch
import torch.nn as nn
import gin
import copy


@gin.configurable
class Base(nn.Module):
    """Base model for disentanglement - agnostic to decoder architecture."""
    
    def __init__(self,
                 net=None,
                 encoder=None,
                 encoder_time=None,
                 latent_dim=64,
                 drop_value=-4.0,
                 drop_rate=0.2,
                 device="cpu"):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.drop_value = drop_value
        self.drop_rate = drop_rate
        
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

    def encode_conditioning(self, latent_time, latent_cond):
        """Encode conditioning from latents."""
        if self.encoder_time is not None:
            time_cond = self.encoder_time(latent_time)  # Structure - keeps temporal info
        else:
            time_cond = latent_time
            
        if self.encoder is not None:
            cond = self.encoder(latent_cond)            # Timbre - global representation
        else:
            cond = None
            
        return time_cond, cond

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
        
    def cfgdrop(self, datas, bsize, drop_targets=[], drop_rate=None):
        """Classifier-free guidance dropout."""
        if drop_rate is None:
            drop_rate = self.drop_rate
            
        draw = torch.rand(bsize)
        test_drop_all = (draw < drop_rate)

        for i in range(len(datas)):
            test_drop_i = (draw > drop_rate * (i + 1)) & (draw < drop_rate * (i + 2))
            test_drop = (test_drop_all + test_drop_i) if i in drop_targets else test_drop_all
            anti_test_drop = ~test_drop

            test_drop = self._broadcast_to(test_drop.to(datas[i]), datas[i].shape)
            anti_test_drop = self._broadcast_to(anti_test_drop.to(datas[i]), datas[i].shape)

            if datas[i] is None:
                datas[i] = None
            else:
                datas[i] = anti_test_drop * datas[i] + test_drop * torch.ones_like(datas[i]) * self.drop_value

        return datas
        
    def _broadcast_to(self, alpha, shape):
        """Broadcast tensor to target shape."""
        assert type(shape) == torch.Size
        return alpha.reshape(-1, *((1, ) * (len(shape) - 1)))

    def configure_optimizers(self, lr=1e-4):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _move_batch_to_device(self, batch):
        """Move batch tensors to device with consistent dtype."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device=self.device, dtype=torch.float32)
        return batch

    @gin.configurable
    def fit(self,
            dataloader=None,
            validloader=None,
            learning_rate=1e-4,
            batch_size=32,
            epochs=100,
            save_every=10,
            val_every=1,
            wandb_logging=True,
            experiment_dir=None,
            grad_clip_max_norm=None,
            ema_decay=0.0,
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

        for epoch in tqdm(range(1, epochs + 1), desc="Training"):
            # Train epoch
            self.train()
            tr_sum, tr_n = 0.0, 0
            for batch in tqdm(dataloader, leave=False, desc=f"Epoch {epoch}"):
                # Move to device and ensure consistent dtype
                batch = self._move_batch_to_device(batch)
                
                # Training step
                opt.zero_grad(set_to_none=True)
                loss = self.training_step(batch, None)["total_loss"]
                
                # Backward pass
                loss.backward()
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip_max_norm))
                opt.step()
                
                # Update EMA after optimizer step
                if use_ema:
                    _ema_update(ema_model, self, ema_decay)
                tr_sum += float(loss.detach())
                tr_n += 1
            tr_loss = tr_sum / max(1, tr_n)

            val_loss = None
            if validloader is not None and (epoch % val_every == 0):
                self.eval()

                vsum, vn = 0.0, 0
                with torch.no_grad():
                    for batch in tqdm(validloader, leave=False, desc=f"Val {epoch}"):
                        batch = self._move_batch_to_device(batch)
                        vsum += float(self.validation_step(batch, None)["total_loss"])
                        vn += 1
                val_loss = vsum / max(1, vn)

            if wandb_logging:
                logd = {"epoch": epoch, "train/total_loss": tr_loss}
                if val_loss is not None:
                    logd["val/total_loss"] = val_loss
                wandb.log(logd)

            if experiment_dir is not None:
                os.makedirs(experiment_dir, exist_ok=True)
                if epoch == 1:
                    with open(os.path.join(experiment_dir, "config.gin"), "w") as f:
                        f.write(gin.operative_config_str())

                save_ckpt = (epoch % save_every == 0)
                save_best = (val_loss is not None and val_loss < best_val)
                if save_ckpt or save_best:
                    state = self.state_dict()
                    ckpt = {
                        "epoch": epoch,
                        "model_state_dict": state,
                        "optimizer_state_dict": opt.state_dict(),
                        "train_loss": tr_loss,
                        "val_loss": val_loss,
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
                        best_val = val_loss
                        torch.save(ckpt, os.path.join(experiment_dir, "best_model.pt"))
        return {"final_train_loss": tr_loss, "best_val_loss": best_val}


@gin.configurable
class DDPM(Base):
    """
    DDPM/DDIM diffusion model for generating music representations.
    
    This class implements the denoising diffusion probabilistic model (DDPM) with support for
    both full DDPM sampling and DDIM (deterministic) sampling. It uses epsilon-prediction
    and supports cosine beta schedules for improved training stability.
    
    Features:
    - Epsilon-prediction loss (MSE)
    - Cosine or linear beta schedule
    - EMA + gradient clipping via Base class
    - DDPM sampler (full 1000 steps)
    - DDIM sampler (η=0, subsampled timesteps)
    """

    def __init__(self,
                 num_diffusion_steps=1000,
                 beta_start=1e-4,
                 beta_end=2e-2,
                 beta_schedule="cosine",
                 **kwargs):
        super().__init__(**kwargs)

        self.num_diffusion_steps = int(num_diffusion_steps)
        self.beta_schedule = beta_schedule

        # Create noise schedule
        betas = self._make_beta_schedule(beta_start, beta_end, self.num_diffusion_steps, self.beta_schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Register noise schedule buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.clamp(alpha_bars, 1e-8, 1.0))

        # Precompute posterior variance for sampling
        one = torch.tensor([1.0], dtype=alpha_bars.dtype, device=alpha_bars.device)
        alpha_bars_prev = torch.cat([one, self.alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        c0 = (alpha_bars_prev.sqrt() * betas) / (1.0 - self.alpha_bars)
        c1 = (alphas.sqrt() * (1.0 - alpha_bars_prev)) / (1.0 - self.alpha_bars)

        # Register posterior sampling coefficients
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", c0)
        self.register_buffer("posterior_mean_coef2", c1)
        
        # Move to target device after all buffers are registered
        self.to(self._target_device)

    @staticmethod
    def _make_beta_schedule(beta_start, beta_end, num_steps, schedule):
        """Create noise schedule for diffusion process."""
        if schedule == "linear":
            return torch.linspace(float(beta_start), float(beta_end), int(num_steps), dtype=torch.float32)
        elif schedule == "cosine":
            import math
            s = 0.008
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps, dtype=torch.float32)
            a_bar = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            a_bar = a_bar / a_bar[0]
            betas = 1 - (a_bar[1:] / a_bar[:-1])
            return torch.clamp(betas, 1e-8, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def add_noise(self, x0, t):
        """Add noise to clean data according to diffusion schedule."""
        b = x0.shape[0]
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(b, 1, 1)
        xt = alpha_bar_t.sqrt() * x0 + (1.0 - alpha_bar_t).sqrt() * eps
        return xt, eps

    def forward(self, noisy_latents, timesteps, time_cond, cond):
        """Predict noise in noisy latents given conditioning."""
        predicted_noise = self.net(
            noisy_latents,
            timesteps=timesteps,
            cond=cond,             # Global conditioning  
            time_cond=time_cond    # Temporal conditioning
        )
        
        return predicted_noise

    def predict_noise(self, noisy_latents, timesteps, time_cond, cond):
        """Alias for forward method - more explicit about what this does."""
        return self.forward(noisy_latents, timesteps, time_cond, cond)

    def _eps_pred(self, x_t, t, time_cond, cond):
        """Predict noise using the neural network."""
        return self.net(x_t, timesteps=t, cond=cond, time_cond=time_cond)

    def compute_loss(self, x_t, eps, t, time_cond, cond):
        """Compute MSE loss between predicted and true noise."""
        eps_pred = self._eps_pred(x_t, t, time_cond, cond)
        loss = nn.functional.mse_loss(eps_pred, eps)
        return {"total_loss": loss, "diff_loss": loss}

    def training_step(self, batch, batch_idx=None):
        """Training step with encoding and loss computation."""
        latents = batch['latent']
        latents = (latents - latents.mean(dim=(1,2), keepdim=True)) / (latents.std(dim=(1,2), keepdim=True) + 1e-6)
        batch_size = latents.shape[0]
        
        # For disentanglement training, we can use the same latents for both encoders
        latent_time = latents   # Time source
        latent_cond = latents   # Global source
        
        # Encode conditioning
        time_cond, cond = self.encode_conditioning(latent_time, latent_cond)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), 
                                device=latents.device, dtype=torch.long)
        
        # Add noise to latents
        noisy_latents, noise = self.add_noise(latents, timesteps)
        
        # Compute loss
        loss_dict = self.compute_loss(noisy_latents, noise, timesteps, time_cond, cond)
        
        # Add conditioning info for potential logging
        loss_dict.update({
            'time_cond': time_cond,
            'cond': cond
        })
        
        return loss_dict

    def validation_step(self, batch, batch_idx=None):
        """Validation step - same as training step."""
        return self.training_step(batch, batch_idx)

    @torch.no_grad()
    def sample(self, time_source, cond_source, num_inference_steps=None, deterministic=False):
        """Stable DDPM sampling (stochastic, training-matched)."""
        self.eval()

        if num_inference_steps is None:
            num_inference_steps = self.num_diffusion_steps
        num_inference_steps = int(num_inference_steps)
        assert deterministic is False, "DDPM sampler is stochastic; set deterministic=False."

        time_cond, cond = self.encode_conditioning(time_source, cond_source)

        x = torch.randn_like(time_source)
        b = x.shape[0]

        for t in range(self.num_diffusion_steps - 1, -1, -1):
            t_batch = torch.full((b,), t, device=x.device, dtype=torch.long)
            eps = self._eps_pred(x, t_batch, time_cond, cond)

            # Scalars for this step
            a_bar_t     = self.alpha_bars[t].to(x).clamp_min(1e-12)
            a_bar_prev  = self.alpha_bars_prev[t].to(x).clamp_min(1e-12)
            beta_t      = self.betas[t].to(x).clamp_min(1e-12)
            alpha_t     = self.alphas[t].to(x).clamp_min(1e-12)

            # Predict clean latent
            sqrt_a_bar  = a_bar_t.sqrt()
            sqrt_1mab   = (1.0 - a_bar_t).clamp_min(1e-12).sqrt()
            x0_hat      = (x - sqrt_1mab * eps) / sqrt_a_bar

            # Mild, distribution-preserving limiter (prevents runaway)
            x0_hat = torch.tanh(x0_hat / 5.0) * 5.0

            # Posterior mean: reshape scalars to broadcast over [B,C,T]
            c1 = (a_bar_prev.sqrt() * beta_t / (1.0 - a_bar_t)).view(1, 1, 1)
            c2 = (alpha_t.sqrt() * (1.0 - a_bar_prev) / (1.0 - a_bar_t)).view(1, 1, 1)
            mean = c1 * x0_hat + c2 * x

            if t > 0:
                var = self.posterior_variance[t].to(x).clamp_min(1e-20).view(1, 1, 1)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean

        return x

