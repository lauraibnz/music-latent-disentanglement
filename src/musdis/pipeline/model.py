import torch
import torch.nn as nn
import gin
import os


@gin.configurable
class Base(nn.Module):
    """Base model for disentanglement - agnostic to decoder architecture."""
    
    def __init__(self,
                 net=None,                  # Decoder network (UNet, Flow, etc.)
                 encoder=None,              # Global encoder (timbre - averages over time)  
                 encoder_time=None,         # Structure encoder (keeps temporal info)
                 latent_dim=64,             # Dimension of latent representations
                 drop_value=-4.0,           # Value for classifier-free guidance dropout
                 drop_rate=0.2,             # Dropout rate for conditioning
                 device="cpu"):             # Device
        super().__init__()
        
        self.latent_dim = latent_dim
        self.drop_value = drop_value
        self.drop_rate = drop_rate
        
        # Encoders (will be set via gin config)
        self.encoder = encoder
        self.encoder_time = encoder_time
        
        # Decoder network - could be UNet, Flow, etc.
        self.net = net
        
        self.to(device)
        
    @property
    def device(self):
        """Get model device."""
        return next(self.parameters()).device
        
    def encode_conditioning(self, latent_time, latent_cond):
        """
        Encode conditioning from latents.
        Args:
            latent_time: Latents to extract time conditioning from [B, latent_dim, T]
            latent_cond: Latents to extract global conditioning from [B, latent_dim, T]  
        Returns:
            time_cond: Time conditioning (structure) [B, channels, T']
            cond: Global conditioning (timbre) [B, channels]
        """
        time_cond = self.encoder_time(latent_time)  # Structure - keeps temporal info
        cond = self.encoder(latent_cond)            # Timbre - global representation
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
            **kwargs):
        """
        Train the model with configurable parameters from gin.
        """
        import wandb
        from tqdm import tqdm
        
        print(f"Starting training with lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")
        
        # Create optimizer
        optimizer = self.configure_optimizers(lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress")
        for epoch in epoch_pbar:
            # Train epoch
            self.train()
            total_train_loss = 0
            num_train_batches = len(dataloader)
            train_loss_accumulator = {}
            
            train_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
            for batch_idx, batch in enumerate(train_pbar):
                # Move to device and ensure consistent dtype
                batch = self._move_batch_to_device(batch)
                
                # Training step
                optimizer.zero_grad()
                loss_dict = self.training_step(batch, batch_idx)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Accumulate losses for averaging
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:  # Scalar tensors only
                        if k not in train_loss_accumulator:
                            train_loss_accumulator[k] = 0
                        train_loss_accumulator[k] += v.item()
                
                # Update metrics
                total_train_loss += loss.item()
                
                # Update progress bar - show both diffusion loss and total loss
                progress_dict = {'total_loss': f'{loss.item():.4f}'}
                if 'diff_loss' in loss_dict:
                    progress_dict['diff_loss'] = f'{loss_dict["diff_loss"].item():.4f}'
                train_pbar.set_postfix(progress_dict)
            
            # Average accumulated losses
            train_losses = {k: v / num_train_batches for k, v in train_loss_accumulator.items()}
            train_loss = train_losses.get('total_loss', total_train_loss / num_train_batches)
            
            # Validate epoch
            val_losses = None
            val_loss = None
            if validloader is not None and epoch % val_every == 0:
                self.eval()
                total_val_loss = 0
                num_val_batches = len(validloader)
                val_loss_accumulator = {}
                
                with torch.no_grad():
                    val_pbar = tqdm(validloader, desc=f"Val {epoch}", leave=False)
                    for batch_idx, batch in enumerate(val_pbar):
                        # Move to device and ensure consistent dtype
                        batch = self._move_batch_to_device(batch)
                        
                        # Validation step
                        loss_dict = self.validation_step(batch, batch_idx)
                        loss = loss_dict['total_loss']
                        
                        # Accumulate losses for averaging
                        for k, v in loss_dict.items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:  # Scalar tensors only
                                if k not in val_loss_accumulator:
                                    val_loss_accumulator[k] = 0
                                val_loss_accumulator[k] += v.item()
                        
                        total_val_loss += loss.item()
                        
                        # Update progress bar - show both diffusion loss and total loss
                        progress_dict = {'val_total_loss': f'{loss.item():.4f}'}
                        if 'diff_loss' in loss_dict:
                            progress_dict['val_diff_loss'] = f'{loss_dict["diff_loss"].item():.4f}'
                        val_pbar.set_postfix(progress_dict)
                
                # Average accumulated losses
                val_losses = {k: v / num_val_batches for k, v in val_loss_accumulator.items()}
                val_loss = val_losses.get('total_loss', total_val_loss / num_val_batches)
            
            # Log to wandb with detailed loss breakdown
            if wandb_logging:
                log_dict = {"epoch": epoch}
                
                # Log all training losses with train/ prefix
                for k, v in train_losses.items():
                    log_dict[f"train/{k}"] = v
                
                # Log all validation losses with val/ prefix
                if val_losses is not None:
                    for k, v in val_losses.items():
                        log_dict[f"val/{k}"] = v
                
                wandb.log(log_dict)
            
            # Auto-save gin configuration (first epoch only)
            if epoch == 1 and experiment_dir is not None:
                gin_config_path = os.path.join(experiment_dir, "config.gin")
                with open(gin_config_path, "w") as f:
                    f.write(gin.operative_config_str())
            
            # Print progress
            if val_loss is not None:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })
            else:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}'
                })
            
            # Save checkpoints
            if experiment_dir is not None:
                save_checkpoint = epoch % save_every == 0
                save_best = val_loss is not None and val_loss < best_val_loss
                
                if save_checkpoint or save_best:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'fit_config': {
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            **kwargs
                        }
                    }
                    
                    # Save regular checkpoint
                    if save_checkpoint:
                        torch.save(checkpoint, os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pt'))
                    
                    # Save best model
                    if save_best:
                        best_val_loss = val_loss
                        torch.save(checkpoint, os.path.join(experiment_dir, 'best_model.pt'))
                        print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        print("Training completed!")
        return {'final_train_loss': train_loss, 'best_val_loss': best_val_loss}


@gin.configurable
class Diffusion(Base):
    """Diffusion-based implementation of Base model."""
    
    def __init__(self,
                 num_diffusion_steps=1000,  # Number of diffusion steps
                 beta_start=1e-4,           # Noise schedule start
                 beta_end=2e-2,             # Noise schedule end
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_diffusion_steps = num_diffusion_steps
        
        # UNet decoder should be set via gin config
        # No fallback creation - gin config is the single source of truth
        
        # Diffusion noise schedule
        self.register_buffer('betas', self._get_beta_schedule(beta_start, beta_end, num_diffusion_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _get_beta_schedule(self, beta_start, beta_end, num_steps):
        """Linear beta schedule for diffusion."""
        return torch.linspace(beta_start, beta_end, num_steps)
        
    def add_noise(self, latents, timesteps):
        """Add noise to latents according to diffusion schedule."""
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)
        
        # Get alpha values for the timesteps
        alphas_cumprod_t = self.alphas_cumprod[timesteps].view(batch_size, 1, 1)
        
        # Add noise: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        noisy_latents = (alphas_cumprod_t.sqrt() * latents + 
                        (1 - alphas_cumprod_t).sqrt() * noise)
        
        return noisy_latents, noise
        
    def forward(self, noisy_latents, timesteps, time_cond, cond):
        """
        Predict noise in noisy latents given conditioning.
        Args:
            noisy_latents: Noisy latent representations [B, latent_dim, T]
            timesteps: Diffusion timesteps [B] 
            time_cond: Time conditioning [B, channels, T']
            cond: Global conditioning [B, channels]
        Returns:
            predicted_noise: Predicted noise [B, latent_dim, T]
        """
        predicted_noise = self.net(
            noisy_latents,
            timesteps=timesteps,
            cond=cond,             # Global conditioning  
            time_cond=time_cond    # Temporal conditioning
        )
        
        return predicted_noise
    
    def predict_noise(self, noisy_latents, timesteps, time_cond, cond):
        """
        Alias for forward method - more explicit about what this does.
        Predict noise in noisy latents given conditioning.
        """
        return self.forward(noisy_latents, timesteps, time_cond, cond)
    
    def compute_loss(self, noisy_latents, noise, timesteps, time_cond, cond):
        """
        Compute diffusion loss (noise prediction) given encoded inputs.
        Args:
            noisy_latents: Noisy latent representations [B, latent_dim, T]
            noise: Target noise [B, latent_dim, T]
            timesteps: Diffusion timesteps [B]
            time_cond: Time conditioning [B, channels, T']
            cond: Global conditioning [B, channels]
        Returns:
            Dictionary with loss components
        """
        # Predict noise
        predicted_noise = self.forward(noisy_latents, timesteps, time_cond, cond)
        
        # Compute loss (MSE between predicted and actual noise)
        diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return {
            'total_loss': diffusion_loss,
            'diff_loss': diffusion_loss,
        }
    
    def training_step(self, batch, batch_idx=None):
        """Training step with encoding and loss computation."""
        latents = batch['latent']  # [B, latent_dim, T]
        batch_size = latents.shape[0]
        
        # For disentanglement training, we can use the same latents for both encoders
        # In the future, you might want to use different sources
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
        """Validation step - same as training step but with no_grad context."""
        return self.training_step(batch, batch_idx)
    
    @torch.no_grad()
    def sample(self, time_source, cond_source, num_inference_steps=50):
        """
        Generate latents using proper DDPM sampling.
        Args:
            time_source: Latents to extract time conditioning from [B, latent_dim, T]
            cond_source: Latents to extract global conditioning from [B, latent_dim, T]
            num_inference_steps: Number of denoising steps
        Returns:
            generated_latents: Generated latents [B, latent_dim, T]
        """
        # Encode conditioning
        time_cond, cond = self.encode_conditioning(time_source, cond_source)
        
        # Start from pure noise
        latents = torch.randn_like(time_source)
        
        # Create proper timestep schedule (evenly spaced)
        timesteps = torch.linspace(self.num_diffusion_steps - 1, 0, num_inference_steps, 
                                  dtype=torch.long, device=latents.device)
        
        # Denoising loop with proper DDPM equations
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(latents.shape[0])
            
            # Predict noise
            predicted_noise = self.forward(latents, t_batch, time_cond, cond)
            
            # Get schedule values
            alpha_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if i < len(timesteps) - 1:
                # Not the final step - use proper DDPM equations
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
                
                # Predict x_0 from current latents
                pred_x0 = (latents - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
                
                # Clamp predicted x_0 to reasonable range
                pred_x0 = torch.clamp(pred_x0, -10, 10)
                
                # Compute mean of posterior q(x_{t-1} | x_t, x_0)
                posterior_variance = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
                posterior_mean_coef1 = (alpha_t_prev.sqrt() * beta_t) / (1 - alpha_t)
                posterior_mean_coef2 = ((1 - beta_t).sqrt() * (1 - alpha_t_prev)) / (1 - alpha_t)
                
                posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * latents
                
                # Add noise (except for t=0)
                if t > 0:
                    noise = torch.randn_like(latents)
                    latents = posterior_mean + posterior_variance.sqrt() * noise
                else:
                    latents = posterior_mean
                    
            else:
                # Final step: predict x_0 directly
                latents = (latents - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
                latents = torch.clamp(latents, -10, 10)
        
        return latents
