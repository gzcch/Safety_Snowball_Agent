import os
import re
from io import BytesIO

import requests
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def is_valid_string(s):
    return s.startswith("llm.") and (s.endswith(".q_proj") or s.endswith(".k_proj") or s.endswith(".v_proj") or s.endswith(".o_proj"))

class VILA:
    def __init__(self, model_path, model_base=None, conv_mode=None):
        disable_torch_init()
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode

        self.model_name = get_model_name_from_path(self.model_path)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            self.model_path, self.model_name, self.model_base
        )

        if self.conv_mode is None:
            if "llama-2" in self.model_name.lower():
                self.conv_mode = "llava_llama_2"
            elif "v1" in self.model_name.lower():
                self.conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                self.conv_mode = "mpt"
            else:
                self.conv_mode = "llava_v0"

    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if not os.path.exists(image_file):
                raise FileNotFoundError(f"Image file not found: {image_file}")
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(self, image_files):
        images = []
        for image_file in image_files:
            image = self.load_image(image_file)
            images.append(image)
        return images

    def eval(
        self,
        image_paths,
        question,
        temperature=0.2,
        top_p=None,
        num_beams=1,
        max_new_tokens=512,
    ):
        images = self.load_images(image_paths)
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        qs = question
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                # Automatically prepend image tokens
                if self.model.config.mm_use_im_start_end:
                    qs = (image_token_se + "\n") * len(images) + qs
                else:
                    qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
    def get_activation_answer_result(self, image_paths, question, answer = ['Sorry, I can\'t help you with that', 'Sure, the detailed information you requested is provided below:']):
        # Hook setup
        # ['Sure, the detailed information you requested is provided below:', 'Based on the image, the detailed information you requested is provided below:'
        activations = {}

        def get_activation(name):
            # 用一个list来保存每个token的激活
            def hook(model, input, output):
                if name not in activations:
                    activations[name] = []
                # detach()之后，按token的顺序保存激活值
                activations[name].append(output.detach().cpu())

            return hook
        # .self_attn.q_proj
        for name, _ in self.model.named_modules():
            if is_valid_string(name):
                submodule = self.model.get_submodule(name)

                # 注册钩子
                submodule.register_forward_hook(get_activation(name))
                # print('FIND')
                # break
            # print(submodule)

        # Build messages


        qs = question

        if len(image_paths)!= 0:
            images = self.load_images(image_paths)
            images_tensor = process_images(
                images, self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if DEFAULT_IMAGE_TOKEN not in qs:
                    # Automatically prepend image tokens
                    if self.model.config.mm_use_im_start_end:
                        qs = (image_token_se + "\n") * len(images) + qs
                    else:
                        qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
        self.activations = {}
        res_lens = []
        for ans in answer:
            # Clear activations before each run


            # Build conversation
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], ans)
            prompt = conv.get_prompt()

            # Tokenize input
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.model.device)

            # Get the length of the assistant's reply in tokens
            ans_tokens = self.tokenizer(ans, return_tensors="pt")["input_ids"][0]
            res_len = len(ans_tokens)
            res_lens.append(res_len)
            # print(images_tensor)
            # print(images_tensor.shape)
            # Perform inference
            with torch.no_grad():
                if len(image_paths) != 0:
                    outputs = self.model(input_ids, images=[images_tensor.half().cuda()])
                else:
                    outputs = self.model(input_ids)

            # Store activations

        #     print(generated)
        # print('#####')
        # print(output_text[0])
        # print('########')
        # print(activations)

        act_image_list = []
        act_text_list = []
        results = {
            "image_paths": image_paths,
            "question": question,
            "answer": answer,
            "activations": {},
            "differences": {}
        }
        for name in activations.keys():
            # print(res_len_image)
            # print(res_len)

            act_1 = activations[name][-1].squeeze(0)[-res_lens[1]:, :]
            act_0 = activations[name][-2].squeeze(0)[-res_lens[0]:, :]
            # print(activations[name][-1].squeeze(0).shape)
            # print(activations[name][-2].squeeze(0).shape)
            # Calculate absolute difference and mean them
            abs_diff_1 = act_1.mean(dim=0)
            abs_diff_0 = act_0.mean(dim=0)
            #torch.abs(act_image).mean() torch.abs(act_text).mean()
            # Calculate the squared difference and RMS difference
            squared_diff = torch.square(act_0.mean(dim=0) - act_1.mean(dim=0))
            rms_diff = torch.sqrt(torch.mean(squared_diff))

            # Store activations and differences
            results["activations"][name] = {
                "act_image": abs_diff_1,
                "act_text": abs_diff_0
            }
            results["differences"][name] = rms_diff

        return results

    def get_latent_embedding(self, image_paths, question, answer=['Sorry, I can\'t help you with that',
                                                                  'Sure, the detailed information you requested is provided below:']):
        import re
        import torch

        # Load images if any
        if image_paths:
            images = self.load_images(image_paths)
            images_tensor = process_images(
                images, self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)
        else:
            images = None
            images_tensor = None

        # Prepare image token sequences
        IMAGE_PLACEHOLDER = "<image>"
        DEFAULT_IMAGE_TOKEN = "<image>"
        DEFAULT_IM_START_TOKEN = "<im_start>"
        DEFAULT_IM_END_TOKEN = "<im_end>"
        IMAGE_TOKEN_INDEX = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        # Prepare content with and without images

        content_text = question
        content_image = (image_token_se + "\n") * len(images) + question

        # Define the combinations
        combinations = {
            'text_safe': (content_text, answer[0], False),
            'text_dan': (content_text, answer[1], False),
            'image_safe': (content_image, answer[0], True),
            'image_dan': (content_image, answer[1], True),
        }

        results = {}

        for key, (qs_variant, ans, use_image) in combinations.items():
            qs_processed = qs_variant


            # Build conversation
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_processed)
            conv.append_message(conv.roles[1], ans)
            prompt = conv.get_prompt()

            # Tokenize input
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.model.device)

            # Get the length of the assistant's reply in tokens
            ans_tokens = self.tokenizer(ans, return_tensors="pt")["input_ids"][0]
            res_len = len(ans_tokens)

            # Perform inference
            with torch.no_grad():
                if use_image:
                    outputs = self.model(input_ids, images=[images_tensor], return_dict=True, output_hidden_states=True)
                else:
                    outputs = self.model(input_ids, return_dict=True,  output_hidden_states=True)
            # print(outputs)
            # Extract latent features
            latent_features = outputs.hidden_states[-1][0, -res_len:, :].mean(dim=0)

            # Store results
            results[key] = latent_features

        return results

    def icl_response(
        self,
        image_paths,
        context,
        current_question,
        temperature=0.2,
        top_p=None,
        num_beams=1,
        max_new_tokens=512,
    ):
        images = self.load_images(image_paths)
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        conv = conv_templates[self.conv_mode].copy()
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        # Append image tokens
        if len(image_paths) == 1:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n"
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n"
        else:
            qs = ""
            for _ in range(len(image_paths)):
                if self.model.config.mm_use_im_start_end:
                    qs += image_token_se + "\n"
                else:
                    qs += DEFAULT_IMAGE_TOKEN + "\n"

        conv.append_message(conv.roles[0], qs)

        # Append conversation history
        for i in range(0, len(context), 2):
            user_input = context[i]
            assistant_response = context[i + 1]
            conv.append_message(conv.roles[0], user_input)
            conv.append_message(conv.roles[1], assistant_response)

        # Append current question
        conv.append_message(conv.roles[0], current_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
