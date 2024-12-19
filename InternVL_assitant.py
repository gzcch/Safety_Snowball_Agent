import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def is_valid_string(s):
    return s.startswith("llm.") and (s.endswith(".q_proj") or s.endswith(".k_proj") or s.endswith(".v_proj") or s.endswith(".o_proj"))

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL:
    def __init__(self, path):
        self.path = path
        self.device_map = self.split_model('InternVL2-40B')
        self.model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            trust_remote_code=True,
            use_fast=False
        )

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
            'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80
        }.get(model_name, 60)
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        return device_map

    def intervl_eval(self, image_paths, question):
        pixel_values_list = []
        num_patches_list = []
        for image_path in image_paths:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            num_patches_list.append(pixel_values.size(0))
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list, dim=0)

        if len(image_paths) == 1:
            prompt = '<image>\n' + question
        else:
            prompt = ''
            for idx in range(len(image_paths)):
                prompt += f'Image-{idx+1}: <image>\n'
            prompt += question

        generation_config = dict(max_new_tokens=512, do_sample=True)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list
        )
        return response

    def icl_response(self, image_paths, context, current_question):
        pixel_values_list = []
        num_patches_list = []
        for image_path in image_paths:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            num_patches_list.append(pixel_values.size(0))
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list, dim=0)

        if len(image_paths) == 1:
            prompt = '<image>\n'
        else:
            prompt = ''
            for idx in range(len(image_paths)):
                prompt += f'Image-{idx+1}: <image>\n'

        history = []
        for i in range(0, len(context), 2):
            user_input = context[i]
            assistant_response = context[i + 1]
            history.append((user_input, assistant_response))

        generation_config = dict(max_new_tokens=512, do_sample=True)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt + current_question,
            generation_config,
            num_patches_list=num_patches_list,
            history=history
        )
        return response


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

        pixel_values_list = []
        num_patches_list = []
        for image_path in image_paths:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            num_patches_list.append(pixel_values.size(0))
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list, dim=0)

        if len(image_paths) == 1:
            prompt = '<image>\n' + question
        else:
            prompt = ''
            for idx in range(len(image_paths)):
                prompt += f'Image-{idx+1}: <image>\n'
            prompt += question
        # print('####')
        # print(inputs.input_ids.shape)
        # print(inputs_image.input_ids.shape)
        #
        # print('####')
        # inputs = inputs.to("cuda")
        # print(inputs)
        # print('#####')
        with torch.no_grad():
        # Inference
            generated, res_len_2 = self.question_answer_forward(self.tokenizer, pixel_values, prompt, answer[0], num_patches_list = num_patches_list)
            generated, res_len_1 = self.question_answer_forward(self.tokenizer, pixel_values, prompt, answer[1], num_patches_list = num_patches_list)
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

            act_image = activations[name][-1].squeeze(0)[-res_len_1:, :]
            act_text = activations[name][-2].squeeze(0)[-res_len_2:, :]
            # print(activations[name][-1].squeeze(0).shape)
            # print(activations[name][-2].squeeze(0).shape)
            # Calculate absolute difference and mean them
            abs_diff_image = act_image.mean(dim=0)
            abs_diff_text = act_text.mean(dim=0)
            #torch.abs(act_image).mean() torch.abs(act_text).mean()
            # Calculate the squared difference and RMS difference
            squared_diff = torch.square(act_text.mean(dim=0) - act_image.mean(dim=0))
            rms_diff = torch.sqrt(torch.mean(squared_diff))

            # Store activations and differences
            results["activations"][name] = {
                "act_image": abs_diff_image,
                "act_text": abs_diff_text
            }
            results["differences"][name] = rms_diff

        return results



    def question_answer_forward(self, tokenizer, pixel_values, question, answer, generation_config,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        # if num_patches_list is None:
        #     num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []


        template = self.model.get_conv_template(self.model.template)
        template_q = self.model.get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)



        template.append_message(template.roles[0], question)
        template_q.append_message(template.roles[0], question)
        template.append_message(template.roles[1], answer)
        template_q.append_message(template.roles[1], None)
        query = template.get_prompt()
        only_question = template_q.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            only_question = only_question.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        question_input_ids = tokenizer(only_question, return_tensors='pt')['input_ids'].cuda()
        input_ids = model_inputs['input_ids'].cuda()
        res_len = input_ids.shape[1] - question_input_ids.shape[1]
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id


        if pixel_values is not None:
            vit_embeds = self.model.extract_feature(pixel_values)
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

        return outputs, res_len