# comfyui-reyrb/__init__.py
"""
ComfyUI-ReyRB: Custom nodes collection
"""

print("[ComfyUI-ReyRB] Loading...")

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import RES4LYF nodes
try:
    from .res4lyf.noise_blend import NODE_CLASS_MAPPINGS as RES4LYF_MAPPINGS
    from .res4lyf.noise_blend import NODE_DISPLAY_NAME_MAPPINGS as RES4LYF_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(RES4LYF_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(RES4LYF_DISPLAY_MAPPINGS)
    print(f"[ComfyUI-ReyRB] Loaded {len(RES4LYF_MAPPINGS)} RES4LYF nodes")
except ImportError as e:
    print(f"[ComfyUI-ReyRB] Failed to load RES4LYF nodes: {e}")
except Exception as e:
    print(f"[ComfyUI-ReyRB] Unexpected error loading RES4LYF nodes: {e}")
    import traceback
    traceback.print_exc()

print(f"[ComfyUI-ReyRB] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print(f"[ComfyUI-ReyRB] Node names: {list(NODE_CLASS_MAPPINGS.keys())}")

# Export the mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
