"""
RES4LYF Noise Blending Nodes
Provides SwarmUI-style variation seed functionality for RES4LYF samplers
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to import from ComfyUI-RES4LYF
res4lyf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ComfyUI-RES4LYF")
if os.path.exists(res4lyf_path) and res4lyf_path not in sys.path:
    sys.path.insert(0, res4lyf_path)

# Import from RES4LYF
try:
    from noise_classes import prepare_noise, NOISE_GENERATOR_CLASSES_SIMPLE
except ImportError:
    print("Warning: Could not import from ComfyUI-RES4LYF. Make sure it's installed.")
    NOISE_GENERATOR_CLASSES_SIMPLE = {"gaussian": None}  # Fallback


def slerp(val, low, high):
    """Spherical linear interpolation between tensors"""
    # Ensure tensors have same shape
    if low.shape != high.shape:
        raise ValueError(f"Tensor shapes don't match: {low.shape} vs {high.shape}")
    
    # Normalize tensors
    low_norm = low / torch.norm(low, dim=(-2, -1), keepdim=True)
    high_norm = high / torch.norm(high, dim=(-2, -1), keepdim=True)
    
    # Flatten for dot product
    low_flat = low_norm.view(low_norm.shape[0], -1)
    high_flat = high_norm.view(high_norm.shape[0], -1)
    
    dot = (low_flat * high_flat).sum(1)
    
    # Handle high similarity case
    if dot.mean() > 0.9995:
        return low * (1 - val) + high * val
    
    omega = torch.acos(torch.clamp(dot, -1, 1))
    so = torch.sin(omega)
    
    # Handle division by zero
    res = torch.where(
        so > 0.0001,
        (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1).unsqueeze(-1) * low + 
        (torch.sin(val * omega) / so).unsqueeze(-1).unsqueeze(-1) * high,
        low * (1 - val) + high * val  # Linear interpolation fallback
    )
    
    return res


class RES4LYFNoiseBlend:
    """
    Blends two noise patterns using spherical linear interpolation (SLERP).
    This provides SwarmUI-style variation seed functionality.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "var_seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "var_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "noise_type": (list(NOISE_GENERATOR_CLASSES_SIMPLE.keys()), {"default": "gaussian"}),
                "normalize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_with_noise",)
    FUNCTION = "blend_noise"
    CATEGORY = "RES4LYF/noise"
    DESCRIPTION = "Blends two noise patterns for variation seed functionality (SwarmUI-style)"
    
    def blend_noise(self, model, latent, base_seed, var_seed, var_strength, noise_type, normalize, alpha=1.0, k=1.0):
        # Get the latent image tensor
        latent_image = latent["samples"]
        batch_size = latent_image.shape[0]
        
        # Get sigma values from model
        sigmin = float(model.model.model_sampling.sigma_min)
        sigmax = float(model.model.model_sampling.sigma_max)
        
        blended_noises = []
        
        for i in range(batch_size):
            # Generate base noise
            torch.manual_seed(base_seed + i)
            noise_sampler_base = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_type)(
                x=latent_image[i:i+1], 
                seed=base_seed + i, 
                sigma_min=sigmin, 
                sigma_max=sigmax
            )
            
            # Set fractal parameters if applicable
            if hasattr(noise_sampler_base, 'alpha'):
                noise_sampler_base.alpha = alpha
                noise_sampler_base.k = k
                if hasattr(noise_sampler_base, 'scale'):
                    noise_sampler_base.scale = 0.1
            
            base_noise = noise_sampler_base(sigma=sigmax, sigma_next=sigmin)
            
            # Generate variation noise
            torch.manual_seed(var_seed + i)
            noise_sampler_var = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_type)(
                x=latent_image[i:i+1], 
                seed=var_seed + i, 
                sigma_min=sigmin, 
                sigma_max=sigmax
            )
            
            if hasattr(noise_sampler_var, 'alpha'):
                noise_sampler_var.alpha = alpha
                noise_sampler_var.k = k
                if hasattr(noise_sampler_var, 'scale'):
                    noise_sampler_var.scale = 0.1
            
            var_noise = noise_sampler_var(sigma=sigmax, sigma_next=sigmin)
            
            # Normalize if requested
            if normalize:
                if base_noise.std() > 0:
                    base_noise = (base_noise - base_noise.mean()) / base_noise.std()
                if var_noise.std() > 0:
                    var_noise = (var_noise - var_noise.mean()) / var_noise.std()
            
            # Blend using SLERP
            if var_strength > 0:
                blended = slerp(var_strength, base_noise, var_noise)
            else:
                blended = base_noise
                
            blended_noises.append(blended)
        
        # Stack all noises
        final_noise = torch.cat(blended_noises, dim=0)
        
        # Create output latent with blended noise
        output = latent.copy()
        output["noise"] = final_noise
        output["noise_blended"] = True
        output["base_seed"] = base_seed
        output["var_seed"] = var_seed
        output["var_strength"] = var_strength
        
        return (output,)


class RES4LYFApplyBlendedNoise:
    """
    Applies the blended noise to a latent image.
    Can either replace the latent (for initial noise) or add to it.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mode": (["replace", "add"], {"default": "replace"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_noise"
    CATEGORY = "RES4LYF/noise"
    DESCRIPTION = "Applies blended noise to latent image (use 'replace' for initial noise)"
    
    def apply_noise(self, latent, strength, mode):
        if "noise" not in latent:
            raise ValueError("No blended noise found in latent. Use RES4LYFNoiseBlend first.")
        
        output = latent.copy()
        noise = latent["noise"] * strength
        
        if mode == "replace":
            # Replace samples with noise (for initial noise)
            output["samples"] = noise.to(latent["samples"].device, latent["samples"].dtype)
        else:
            # Add noise to existing samples
            output["samples"] = latent["samples"] + noise.to(latent["samples"].device, latent["samples"].dtype)
        
        # Preserve variation seed info for downstream nodes
        if "base_seed" in latent:
            output["base_seed"] = latent["base_seed"]
        if "var_seed" in latent:
            output["var_seed"] = latent["var_seed"]
        if "var_strength" in latent:
            output["var_strength"] = latent["var_strength"]
        
        # Clean up the noise from the latent
        if "noise" in output:
            del output["noise"]
        if "noise_blended" in output:
            del output["noise_blended"]
            
        return (output,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "RES4LYFNoiseBlend": RES4LYFNoiseBlend,
    "RES4LYFApplyBlendedNoise": RES4LYFApplyBlendedNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RES4LYFNoiseBlend": "RES4LYF Noise Blend (SwarmUI-style)",
    "RES4LYFApplyBlendedNoise": "RES4LYF Apply Blended Noise",
}
