import torch

def slerp(val, low, high):
    """Spherical linear interpolation between tensors"""
    # Get original shape and flatten to 2D for easier computation
    orig_shape = low.shape
    low = low.reshape(low.shape[0], -1)
    high = high.reshape(high.shape[0], -1)
    
    # Normalize
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    
    # Compute dot product
    dot = (low_norm * high_norm).sum(1)
    
    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # If vectors are very similar, use linear interpolation
    if dot.mean() > 0.9995:
        result = (1 - val) * low + val * high
    else:
        # Calculate angle between vectors
        omega = torch.acos(dot)
        so = torch.sin(omega)
        
        # Perform SLERP
        result = torch.where(
            so > 0.0001,
            (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + 
            (torch.sin(val * omega) / so).unsqueeze(1) * high,
            (1 - val) * low + val * high  # Fallback to linear
        )
    
    # Reshape back to original shape
    return result.reshape(orig_shape)

class LatentSlerp:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "blend_slerp"
    CATEGORY = "latent/transform"
    
    def blend_slerp(self, latent1, latent2, strength):
        # Get samples
        samples1 = latent1["samples"]
        samples2 = latent2["samples"]
        
        # Ensure same device and dtype
        device = samples1.device
        dtype = samples1.dtype
        samples2 = samples2.to(device=device, dtype=dtype)
        
        # Apply SLERP
        blended = slerp(strength, samples1, samples2)
        
        # Create output latent
        output = latent1.copy()
        output["samples"] = blended
        
        return (output,)

NODE_CLASS_MAPPINGS = {
    "LatentSlerp": LatentSlerp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSlerp": "Latent SLERP",
}
