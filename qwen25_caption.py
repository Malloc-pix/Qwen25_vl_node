import torch
from PIL import Image
import folder_paths
import os
import math
import numpy as np
import comfy.model_management as mm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(folder_paths.models_dir, "LLM")

_model = None
_processor = None

def resizeByLongeSide(pil_img, target_size=512):
    width, height = pil_img.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    new_height = math.ceil(new_height / 8) * 8
    new_width = math.ceil(new_width / 8) * 8
    return pil_img.resize((new_width, new_height), Image.LANCZOS)

def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy(), 0, 255).astype(np.uint8))

class Qwen25Captioner:
    @classmethod
    def INPUT_TYPES(s):
        query = "Please describe the person answer according to the following format:\n" \
                "Hair color: ,\n" \
                "Body type: ,\n" \
                "Gender: ,\n" \
                "Age: ,\n" \
                "Race: ,\n"

        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ([
                    "Qwen2.5-VL-7B-Instruct",  # 可在 models/LLM 中预下载
                    "Qwen-VL-Max",
                ], {"default": "Qwen2.5-VL-7B-Instruct"}),

                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),

                "quant": (["16", "4", "8"], {"default": "16"}),  # 16 表示无量化

                "query": ("STRING", {"multiline": True, "default": query}),
                "cached": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Qwen2.5-VL"

    def inference(self, image, model_name, precision, quant, query, cached):
        try:
            global _processor, _model
            device = mm.get_torch_device()

            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map[precision]

            model_path = os.path.join(model_directory, model_name)

            if not cached or _model is None:
                if quant == "16":
                    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        trust_remote_code=True
                    ).eval().to(device)  
                else:
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=(quant == "4"),
                        load_in_8bit=(quant == "8"),
                        llm_int8_threshold=6.0,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch_dtype,
                    )
                    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        device_map="auto",
                        trust_remote_code=True,
                        quantization_config=quant_cfg,
                        low_cpu_mem_usage=True,
                    ).eval()

                _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            model = _model
            processor = _processor

            if image.dim() == 2:
                image = image.unsqueeze(0)

            torch_img = image.squeeze(0)
            pil_image = resizeByLongeSide(tensor2pil(torch_img))

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": query}
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                output_texts = processor.batch_decode(
                    generated_ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            response = output_texts[0]

            if not cached:
                del _model
                del _processor
                _model = None
                _processor = None
            comfy.model_management.soft_empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            mm.free_memory(mm.get_total_memory(device), device)
            mm.soft_empty_cache()
            raise e

        return (response,)
