from .qwen25_caption import Qwen25Captioner,CLIPDynamicTextEncode

NODE_CLASS_MAPPINGS = {
    "Qwen25Captioner": Qwen25Captioner,
    "CLIPDynamicTextEncode": CLIPDynamicTextEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen25Captioner": "Qwen-2.5 Captioner",
    "CLIPDynamicTextEncode": "CLIP Dynamic Text Encode(cy)",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
