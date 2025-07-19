# comfyui-reyrb/__init__.py
"""
ComfyUI-ReyRB: Custom nodes collection
"""

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import RES4LYF nodes
try:
    from .res4lyf import NODE_CLASS_MAPPINGS as RES4LYF_MAPPINGS
    from .res4lyf import NODE_DISPLAY_NAME_MAPPINGS as RES4LYF_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(RES4LYF_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(RES4LYF_DISPLAY_MAPPINGS)
except ImportError as e:
    print(f"Failed to load RES4LYF nodes: {e}")

# Import other node collections here as you add them
# try:
#     from .other_nodes import NODE_CLASS_MAPPINGS as OTHER_MAPPINGS
#     NODE_CLASS_MAPPINGS.update(OTHER_MAPPINGS)
# except ImportError:
#     pass

# Export the mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Optional: Add version info
__version__ = "0.1.0"
