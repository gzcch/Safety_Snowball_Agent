# filename: llava_one_vision_assistant.py
import torch
import copy
from PIL import Image
import requests

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

class LLaVAOneVisionAssistant:
    def __init__(
        self,
        model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        device="cuda",
    ):

        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(
            model_name
        )
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.device = device


        self.model.eval()  # 推理模式


    def eval(self, image_paths, question, temperature=0.0, max_new_tokens=512):


        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # 用 <image> 占位
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        # 3. 读入图像
        images = []
        for path in image_paths:
            if path.startswith("http"):
                img = Image.open(requests.get(path, stream=True).raw).convert("RGB")
            else:
                img = Image.open(path).convert("RGB")
            images.append(img)
        if len(images) == 0:
            images = None
        inputs = self.processor(
            images=images,
            text=prompt,
            return_tensors="pt",
            padding=True
        ).to(self.model.device, torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        answer = self.processor.decode(outputs[0], skip_special_tokens=True)

        return answer


    def icl_response(self, image_paths, context, current_question,
                     temperature=0.0, max_new_tokens=512):

        conversation = []

        for i in range(0, len(context), 2):
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": context[i]}
                ],
            })
            if (i + 1) < len(context):
                conversation.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": context[i+1]}
                    ],
                })


        conversation.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": current_question},
            ],
        })

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )


        images = []
        for path in image_paths:
            if path.startswith("http"):
                img = Image.open(requests.get(path, stream=True).raw).convert("RGB")
            else:
                img = Image.open(path).convert("RGB")
            images.append(img)
        if len(images) == 0:
            images = None

        inputs = self.processor(
            images=images,
            text=prompt,
            return_tensors="pt",
            padding=True
        ).to(self.model.device, torch.float16)


        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )


        answer = self.processor.decode(outputs[0], skip_special_tokens=True)

        return answer
