# filename: phi_vision_assistant.py

import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

class PhiAssistant:


    def __init__(
        self,
        model_id="microsoft/Phi-3.5-vision-instruct",
        device="cuda",
        use_flash_attn=True,
        num_crops=4,
        max_image_limit=16,
        load_in_8bit=False,
    ):

        self.model_id = model_id
        self.device = device
        self.num_crops = num_crops
        self.max_image_limit = max_image_limit

        # 额外设置 attn 实现
        attn_implementation = "flash_attention_2" if use_flash_attn else "eager"


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation=attn_implementation,
        )


        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            num_crops=self.num_crops
        )

        self.model.eval()

    def _prepare_images_and_placeholder(self, image_paths):
        images = []
        placeholder = ""
        for i, path in enumerate(image_paths):
            if i >= self.max_image_limit:
                print(f"Warning: 已超过最大图像数量限制 {self.max_image_limit}，后续图像将被忽略。")
                break
            if path.startswith("http://") or path.startswith("https://"):
                img = Image.open(requests.get(path, stream=True).raw)
            else:
                img = Image.open(path)
            images.append(img)
            placeholder += f"<|image_{i+1}|>\n"  # i+1 纯粹是为了与官方示例一致
        return images, placeholder

    def eval(self, image_paths, question, max_new_tokens=512, temperature=0.0, do_sample=False):


        images, placeholders = self._prepare_images_and_placeholder(image_paths)


        messages = [
            {
                "role": "user",
                "content": placeholders + question  # 将图像 placeholder + 文本问题连接起来
            }
        ]


        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if len(images)==0:
            images = None

        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, **generation_args)


        output_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

    def icl_response(self, image_paths, context, current_question, max_new_tokens=512, temperature=0.0, do_sample=False):

        images, placeholders = self._prepare_images_and_placeholder(image_paths)


        messages = []
        role_user = "user"
        role_assistant = "assistant"
        for i in range(0, len(context), 2):
            user_text = context[i]
            assistant_text = ""
            if i+1 < len(context):
                assistant_text = context[i+1]
            messages.append({"role": role_user, "content": user_text})
            messages.append({"role": role_assistant, "content": assistant_text})

        messages.append({"role": role_user, "content": placeholders + current_question})


        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if len(images)==0:
            images = None

        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, **generation_args)

        output_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response
