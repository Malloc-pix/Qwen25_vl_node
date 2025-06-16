from .qwen25_caption import Qwen25Captioner

NODE_CLASS_MAPPINGS = {
    "Qwen25Captioner": Qwen25Captioner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen25Captioner": "Qwen-2.5 Captioner",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']