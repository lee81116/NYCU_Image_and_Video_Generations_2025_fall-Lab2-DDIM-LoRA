from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t).to(x.device)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        elif mode == "cosine":
            ######## TODO ########
            # Implement the cosine beta schedule (Nichol & Dhariwal, 2021).
            # Hint:
            # 1. Define alphā_t = f(t/T) where f is a cosine schedule:
            #       alphā_t = cos^2( ( (t/T + s) / (1+s) ) * (π/2) )
            #    with s = 0.008 (a small constant for stability).
            # 2. Convert alphā_t into betas using:
            #       beta_t = 1 - alphā_t / alphā_{t-1}
            # 3. Return betas as a tensor of shape [num_train_timesteps].
            s = 0.008
            timesteps = num_train_timesteps
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps, dtype = torch.float32) / timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
            
        else:
            raise NotImplementedError(f"{mode} is not implemented.")
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        
        self.schedule_mode = mode      

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    
    
    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        if predictor == "noise": #### TODO
            return self.step_predict_noise(x_t, t, net_out)
        elif predictor == "x0": #### TODO
            return self.step_predict_x0(x_t, t, net_out)
        elif predictor == "mean": #### TODO
            return self.step_predict_mean(x_t, t, net_out)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")

    
    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        Noise prediction version (the standard DDPM formulation).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            eps_theta: predicted noise ε̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, and alpha_bar_t from the scheduler.
        # 2. Compute the predicted mean μ_θ(x_t, t) = 1/√α_t * (x_t - (β_t/√(1-ᾱ_t)) * ε̂_θ).
        # 3. Compute the posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
        # 4. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0.
        # 5. Return the final sample at t-1.
        sample_prev = None

        # beta_t = extract(self.betas, t, x_t)
        # alpha_t = extract(self.alphas, t, x_t)
        # alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        # t_prev = (t - 1).clamp(min=0)
        # alpha_bar_t_prev = extract(
        #     self.alphas_cumprod, t_prev, x_t
        # )
        # mu_theta = 1 / torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta)
        # tilde_beta_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
        # sample_prev = mu_theta + (torch.sqrt(tilde_beta_t)) * torch.randn_like(x_t) if t > 0 else mu_theta
        
        # The original code for step_predict_noise. Because cosine's scheduler's time is very different from linear and quad, it will make the sampling very unstable.
        # Especially this term (β_t / √(1-ᾱ_t)), which is quite huge when t is large.
        # Ex. For timestep t = 999, Cosine Scheduler's term = 0.9990000128746033
        # While Linear Scheduler's one = 0.020000403746962547
        # And Quad Scheduler's one = 0.020007340237498283
        # This make mu_theta from cosine scheduler very unstable, because the model need to predict noise with very high precision.
        # Even with a solid model, t = 999 almost decide the whole sampling result, which is not very desirable.

        # The following implementation is more stable for cosine scheduler.

        # 1. Predict x_0 from the predicted noise (eps_theta).
        #    x_0_pred = (1/sqrt(ᾱ_t)) * (x_t - sqrt(1-ᾱ_t) * ε̂_θ)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        
        x0_pred = (
            (1.0 / torch.sqrt(alpha_bar_t)) * (x_t -  torch.sqrt(1 - alpha_bar_t) * eps_theta)
        )

        # 2. Clip the predicted x_0 to the valid range [-1, 1].
        x0_pred.clamp_(-1.0, 1.0)

        # 3. Compute the posterior mean using the clipped x_0.
        #    This formula is derived from q(x_{t-1} | x_t, x_0).
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_t_prev = extract(self.alphas_cumprod, t_prev, x_t)
        beta_t = extract(self.betas, t, x_t)
        
        # Coefficients for the posterior mean q(x_{t-1} | x_t, x_0)
        # Coef1 = (sqrt(ᾱ_{t-1}) * β_t) / (1 - ᾱ_t)
        # Coef2 = (sqrt(α_t) * (1 - ᾱ_{t-1})) / (1 - ᾱ_t)
        q_posterior_mean_coef1 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1.0 - alpha_bar_t)
        q_posterior_mean_coef2 = (torch.sqrt(extract(self.alphas, t, x_t)) * (1.0 - alpha_bar_t_prev)) / (1.0 - alpha_bar_t)
        
        model_mean = q_posterior_mean_coef1 * x0_pred + q_posterior_mean_coef2 * x_t

        # 4. Compute the posterior variance.
        tilde_beta_t = ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)) * beta_t
        
        # 5. Add Gaussian noise scaled by the posterior standard deviation.
        noise = torch.randn_like(x_t) if t > 0 else 0.0
        log_tilde_beta_t = torch.log(tilde_beta_t.clamp(min=1e-20))
        sample_prev = model_mean + (0.5 * log_tilde_beta_t).exp() * noise
        
        #######################
        return sample_prev

    
    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        """
        x0 prediction version (alternative DDPM objective).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            x0_pred: predicted clean image x̂₀(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        sample_prev = None
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_t_prev = extract(
            self.alphas_cumprod, t_prev, x_t
        )
        x0_pred.clamp_(-1.0, 1.0)
        mean = torch.sqrt(alpha_t)*(1 - alpha_bar_t_prev)/(1 - alpha_bar_t) * x_t + torch.sqrt(alpha_bar_t_prev)*beta_t/(1-alpha_bar_t) * x0_pred
        
        tilde_beta_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
        log_tilde_beta_t = torch.log(tilde_beta_t.clamp(min=1e-20))
        sample_prev = mean + (0.5 * log_tilde_beta_t).exp() * torch.randn_like(x_t) if t > 0 else mean
        #######################
        return sample_prev

    
    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        """
        Mean prediction version (directly outputting the posterior mean).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            mean_theta: network-predicted posterior mean μ̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        sample_prev = None
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_t_prev = extract(
            self.alphas_cumprod, t_prev, x_t
        )
        # tilde_beta_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
        # sample_prev = mean_theta + (torch.sqrt(tilde_beta_t)) * torch.randn_like(x_t) if t > 0 else mean_theta

        x0_pred = ((1 - alpha_bar_t) * mean_theta - torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) * x_t) / (torch.sqrt(alpha_bar_t_prev) * beta_t)
        x0_pred.clamp_(-1.0, 1.0)
        q_posterior_mean_coef1 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1.0 - alpha_bar_t)
        q_posterior_mean_coef2 = (torch.sqrt(extract(self.alphas, t, x_t)) * (1.0 - alpha_bar_t_prev)) / (1.0 - alpha_bar_t)
        
        new_mean = q_posterior_mean_coef1 * x0_pred + q_posterior_mean_coef2 * x_t
        tilde_beta_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
        sample_prev = new_mean + (torch.sqrt(tilde_beta_t)) * torch.randn_like(x_t) if t > 0 else new_mean
        #######################
        return sample_prev

    
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps       = torch.randn(x_0.shape, device='cuda')

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        x_t = None
        eps = eps.to(x_0.device)
        alphas_prod_t = extract(self.alphas_cumprod, t, x_0)
        x_t = x_0 * torch.sqrt(alphas_prod_t) + eps * torch.sqrt(1.0 - alphas_prod_t)
        #######################

        return x_t, eps



class DDIMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode: str = "linear",
        num_inference_timesteps: int = 50,
        eta: float = 0.0,
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        self.eta = float(eta)
        self.set_inference_timesteps(num_inference_timesteps)

    def set_inference_timesteps(self, num_inference_timesteps: int):
        """
        Define the inference schedule (a subset of training timesteps, descending order).
        Inputs:
            num_inference_timesteps (int): number of inference steps (e.g., 50).
        """
        ######## TODO ########
        # Hint:
        #   - Define the DDIM inference schedule based on the given num_inference_timesteps.
        #   - The schedule should be a subset of training timesteps, ordered in descending fashion.
        #   - Store the result in `self.timesteps` (as a torch tensor) 
        #   - Store the step ratio in `self._ddim_step_ratio` for later use when computing previous t.
        #   - Compute a `step_ratio` that maps inference steps to training steps.
        # DO NOT change the code outside this part.
        self.num_inference_timesteps = num_inference_timesteps
        self._ddim_step_ratio = self.num_train_timesteps / self.num_inference_timesteps
        timesteps = (torch.arange(0, num_inference_timesteps) * self._ddim_step_ratio).round()
        timesteps = torch.flip(timesteps,dims=(0,))
        self.timesteps = timesteps.long()
        #######################

    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor):
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor, predictor: str):
        """
        One step DDIM update: x_t -> x_{t_prev} with deterministic/stochastic control via eta.

        Input:
            x_t: [B,C,H,W]
            t: current absolute timestep index
            eps_theta: predicted noise
            predictor: predictor type
        Output:
            sample_prev: x at previous inference timestep
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        assert predictor == "noise", "In assignment 2, we only implement DDIM with noise predictor."
        # 1. get previous step value
        t_index = (self.timesteps == t).nonzero(as_tuple=True)[0]
        t_prev = self.timesteps[t_index + 1] if t_index < len(self.timesteps) - 1 else torch.tensor(-1)
        t_prev = t_prev.to(x_t.device)

        alpha_prod_t = extract(self.alphas_cumprod, t, x_t)
        if t_prev >= 0:
            alpha_prod_t_prev = extract(self.alphas_cumprod, t_prev, x_t)
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)


        predict_x0 = (x_t - torch.sqrt(1 - alpha_prod_t) * eps_theta) / torch.sqrt(alpha_prod_t)

        sigma_t = self.eta \
                * torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) \
                * torch.sqrt(1 - alpha_prod_t / alpha_prod_t_prev)

        mu_tilde = torch.sqrt(alpha_prod_t_prev) * predict_x0 + torch.sqrt(1 - alpha_prod_t_prev - sigma_t**2) * eps_theta

        sample_prev =  mu_tilde + sigma_t * torch.randn_like(x_t)


        #######################
        return sample_prev