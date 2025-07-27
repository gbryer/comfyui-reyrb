import json

class ClipVisionOutputToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "precision": ("INT", {"default": 6, "min": 1, "max": 10}),
                "max_elements": ("INT", {"default": 0, "min": 0, "max": 100000})
            }
        }

    RETURN_TYPES = ("TEXT",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "convert"
    CATEGORY = "custom"

    def convert(self, clip_vision_output, precision, max_elements):
        tensor = clip_vision_output["image_embeds"]

        # Convert to flat list of floats
        flat = tensor.flatten().tolist()

        # Limit number of elements if needed
        if max_elements > 0:
            flat = flat[:max_elements]

        # Round floats to desired precision
        rounded = [round(v, precision) for v in flat]

        # Convert to JSON string
        text = json.dumps(rounded)

        return (text,)


NODE_CLASS_MAPPINGS = {
    "ClipVisionOutputToText": ClipVisionOutputToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipVisionOutputToText": "CLIP Vision Output to Text"
}