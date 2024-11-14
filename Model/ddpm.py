import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator: torch.generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self_alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, original_sample: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_cumprod.to(device=original_sample.device, dtype=original_sample.dtype)
        timesteps = timesteps.to(original_sample.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise = torch.randn(original_sample.shape, generator=self.generator, device=original_sample.device, dtype= original_sample.dtype)
        noisy_samples = (sqrt_alpha_prod * original_sample) + (sqrt_one_minus_alpha_prod * noise)
        return noisy_samples
