import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Make sure to have this utility module
import matplotlib.pyplot as plt
import torch

def is_valid_string(s):
    return s.startswith("model.") and (s.endswith(".q_proj") or s.endswith(".k_proj") or s.endswith(".v_proj") or s.endswith(".o_proj"))
class QwenVL2Assistant:
    def __init__(self, model_name="Qwen/Qwen2-VL-72B-Instruct"):
        # Load the model and processor
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_name, torch_dtype="auto", device_map="cuda"
        # ).eval()
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.diff_dict = {}

    def eval(self, image_paths, question):
        # Build messages
        content = []
        content = [{"type": "image", "image": self.resize_image(image_path)} for image_path in image_paths]
        # for image_path in image_paths:
        #     content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def icl_response(self, image_paths, context, current_question):
        # Build messages with context
        messages = []
        for i in range(0, len(context), 2):
            user_input = context[i]
            assistant_response = context[i + 1]
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": assistant_response})

        # Add current user input
        content = [{"type": "image", "image": self.resize_image(image_path)} for image_path in image_paths]
        # for image_path in image_paths:
        #     content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": current_question})
        messages.append({"role": "user", "content": content})

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def compute_change_score(self, activations_M1, activations_M2):
        scores = {}

        for key in activations_M1:
            # Ensure both models have the same activation keys (layers)
            if key in activations_M2:
                activation_M1 = activations_M1[key]
                activation_M2 = activations_M2[key]

                # Compute the squared difference between activations
                squared_diff = torch.square(activation_M1 - activation_M2)

                # Compute the root mean square of the differences
                rms_diff = torch.sqrt(torch.mean(squared_diff))

                # Store the score for the current layer
                scores[key] = rms_diff.item()

        return scores


    def single_activate(self, image_paths, question):
        # Hook setup
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
        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")
        print(inputs.input_ids.shape)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        messages.append({"role": "assistant", "content": output_text[0]})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        print(inputs.input_ids.shape)
        # inputs = inputs.to("cuda")
        # print(inputs)
        # print('#####')
        with torch.no_grad():
        # Inference
            generated = self.model(**inputs)
        #     print(generated)
        # print('#####')
        # print(output_text[0])
        # print('########')
        # print(activations)
        for name in activations.keys():
            print('###')
            for item in activations[name]:
                print(item.shape)
            print('###')
            activations[name] = torch.cat([item.squeeze(0) for item in activations[name] if item.squeeze(0).shape[0] == 1])
            print(activations[name].shape)
            break
        return output_text[0], activations

    def get_activation_result(self, image_paths, question):
        # Hook setup
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
        content_image = []
        content = []
        for image_path in image_paths:
            content_image.append({"type": "image", "image": image_path})
        content_image.append({"type": "text", "text": question})
        content.append({"type": "text", "text": question})
        messages_image = [{"role": "user", "content": content_image}]
        messages = [{"role": "user", "content": content}]
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")
        input_len = inputs.input_ids.shape[1]
        # print('####')
        # print(inputs.input_ids.shape)
        # print(inputs_image.input_ids.shape)
        # print('####')
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        messages.append({"role": "assistant", "content": output_text[0]})
        messages_image.append({"role": "assistant", "content": output_text[0]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        res_len = inputs.input_ids.shape[1] - input_len
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
            generated = self.model(**inputs)
            generated_image = self.model(**inputs_image)
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
            "answer": output_text[0],
            "activations": {},
            "differences": {}
        }
        for name in activations.keys():
            act_image = activations[name][-1].squeeze(0)[-res_len:, :]
            act_text = activations[name][-2].squeeze(0)[-res_len:, :]

            # Calculate absolute difference and mean them
            abs_diff_image = torch.abs(act_image).mean()
            abs_diff_text = torch.abs(act_text).mean()

            # Calculate the squared difference and RMS difference
            squared_diff = torch.square(act_text - act_image)
            rms_diff = torch.sqrt(torch.mean(squared_diff))

            # Store activations and differences
            results["activations"][name] = {
                "act_image": abs_diff_image,
                "act_text": abs_diff_text
            }
            results["differences"][name] = rms_diff

        return results

    def dynamic_activation_patching(self, image_paths,  question, max_steps=128):
        # Hook setup for caching activations
        activations = {}

        def get_activation(name):
            # Hook function to cache activations by token
            def hook(model, input, output):
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach().cpu())

            return hook

        for name, _ in self.model.named_modules():
            if is_valid_string(name):
                submodule = self.model.get_submodule(name)
                submodule.register_forward_hook(get_activation(name))

        # Process inputs M2 and cache activations
        content_image = [{"type": "text", "text": question}]
        for image_path in image_paths:
            content_image.append({"type": "image", "image": image_path})
        content_image.append({"type": "text", "text": question})

        messages_image = [{"role": "user", "content": content_image}]
        messages_image = self.processor.apply_chat_template(messages_image, tokenize=False, add_generation_prompt=True)
        content_text = [{"type": "text", "text": question}]


        messages_text = [{"role": "user", "content": content_text}]
        messages_text = self.processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages_image)
        inputs = self.processor(
            text=[messages_text],
            padding=True,
            return_tensors="pt"
        )

        # Generate M2 (this caches the activations for later use)
        self.model.generate(**inputs, max_new_tokens=max_steps)


        inputs_m1 = self.processor(
            text=[messages_text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Initialize for dynamic activation patching loop
        input_len = inputs_m1.input_ids.shape[1]
        generated_tokens = []

        for step in range(max_steps):
            with torch.no_grad():
                # Replace activations in M1 with cached activations from M2
                for name in activations.keys():
                    if name in activations:
                        # Get cached activations from M2 and patch into M1
                        cached_activation = activations[name][-1].squeeze(0)
                        submodule = self.model.get_submodule(name)

                        # Hook to modify activation before forward pass
                        def patch_activation(module, input, output):
                            output = cached_activation
                            return output

                        # Replace activation of the target neuron in the forward pass
                        hook_handle = submodule.register_forward_hook(patch_activation)

                # Run forward pass on patched model with M1
                generated_ids = self.model.generate(**inputs_m1, max_new_tokens=1)

                # Unpatch the model (remove the hooks)
                hook_handle.remove()

                # Get the next token and update the input for the next step
                new_token_ids = generated_ids[:, input_len:]  # Only take newly generated tokens
                inputs_m1.input_ids = torch.cat([inputs_m1.input_ids, new_token_ids], dim=-1)
                input_len += new_token_ids.shape[1]

                # Decode generated tokens and append to output
                generated_tokens.append(self.processor.decode(new_token_ids[0], skip_special_tokens=True))

            # Break early if generation condition is met (e.g., end token)
            if generated_tokens[-1] == "<end>":
                break

        # Join the generated tokens to form final output text
        final_output = "".join(generated_tokens)
        return final_output, activations

    def activate_text_image(self, image_paths, question):
        # Hook setup
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
        content_image = []
        content = []
        for image_path in image_paths:
            content_image.append({"type": "image", "image": image_path})
        content_image.append({"type": "text", "text": question})
        content.append({"type": "text", "text": question})
        messages_image = [{"role": "user", "content": content_image}]
        messages = [{"role": "user", "content": content}]
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")
        input_len = inputs.input_ids.shape[1]
        print('####')
        print(inputs.input_ids.shape)
        print(inputs_image.input_ids.shape)
        print('####')
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        messages.append({"role": "assistant", "content": output_text[0]})
        messages_image.append({"role": "assistant", "content": output_text[0]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        res_len = inputs.input_ids.shape[1] - input_len
        print('####')
        print(inputs.input_ids.shape)
        print(inputs_image.input_ids.shape)

        print('####')
        # inputs = inputs.to("cuda")
        # print(inputs)
        # print('#####')
        with torch.no_grad():
        # Inference
            generated = self.model(**inputs)
            generated_image = self.model(**inputs_image)
        #     print(generated)
        # print('#####')
        # print(output_text[0])
        # print('########')
        # print(activations)

        act_image_list = []
        act_text_list = []

        for name in activations.keys():
            print(name)
            act_image = activations[name][-1].squeeze(0)[-res_len:, :]
            act_text = activations[name][-2].squeeze(0)[-res_len:, :]
            print(act_text.shape)
            # Calculate absolute difference and mean them
            abs_diff_image = torch.abs(act_image).mean()
            abs_diff_text = torch.abs(act_text).mean()

            # Append the differences to the lists
            act_image_list.append((name, abs_diff_image))
            act_text_list.append((name, abs_diff_text))

            # Calculate the squared difference and RMS difference
            squared_diff = torch.square(act_text - act_image)
            rms_diff = torch.sqrt(torch.mean(squared_diff))

            # Store the RMS difference in the diff_dict
            self.diff_dict[name] = rms_diff

        layers = list(self.diff_dict.keys())
        act_image_vals = [val for _, val in act_image_list]
        act_text_vals = [val for _, val in act_text_list]
        rms_diff_vals = [self.diff_dict[layer] for layer in layers]

        # Plotting the values
        plt.figure(figsize=(10, 6))

        # Plot act_image
        plt.plot(layers, act_image_vals, label='act_image', marker='o', linestyle='-')

        # Plot act_text
        plt.plot(layers, act_text_vals, label='act_text', marker='x', linestyle='--')

        # Plot RMS difference
        plt.plot(layers, rms_diff_vals, label='RMS Difference', marker='s', linestyle='-.')

        # Customize the plot
        plt.xlabel('Layers')
        plt.ylabel('Activation Values')
        plt.title('Activation Differences across Layers')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()


        # Save the plot to a file
        plt.savefig('activation_differences_plot.png')
        # Show the plot
        plt.show()
        # Part A: Find the top 20 names for act_image, act_text, and rms_diff
        top_20_image = sorted(act_image_list, key=lambda x: x[1], reverse=True)[:50]
        top_20_text = sorted(act_text_list, key=lambda x: x[1], reverse=True)[:50]

        # Get the names from the top 20 lists
        top_20_image_names = {name for name, _ in top_20_image}
        top_20_text_names = {name for name, _ in top_20_text}

        # Find names that are in top_20_image but not in top_20_text
        image_not_in_text = top_20_image_names - top_20_text_names

        # Find names that are in top_20_text but not in top_20_image
        text_not_in_image = top_20_text_names - top_20_image_names

        # Save the top 20 results and discrepancies to a file
        with open('top_20_results.txt', 'w') as f:
            f.write("Top 20 act_image (mean absolute difference):\n")
            for item in top_20_image:
                f.write(f"{item[0]}: {item[1]}\n")

            f.write("\nTop 20 act_text (mean absolute difference):\n")
            for item in top_20_text:
                f.write(f"{item[0]}: {item[1]}\n")

            f.write("\nIn top_20_image but not in top_20_text:\n")
            for name in image_not_in_text:
                f.write(f"{name}\n")

            f.write("\nIn top_20_text but not in top_20_image:\n")
            for name in text_not_in_image:
                f.write(f"{name}\n")

        # Return the first output_text and the diff_dict as per the original function
        return output_text[0], self.diff_dict

    def get_activation_result_multi_question(self, image_paths, question_text, question_image):
        # Hook setup
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
        content_image = []
        content = []
        for image_path in image_paths:
            content_image.append({"type": "image", "image": image_path})
        content_image.append({"type": "text", "text": question_image})
        content.append({"type": "text", "text": question_text})
        messages_image = [{"role": "user", "content": content_image}]
        messages = [{"role": "user", "content": content}]
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")
        input_len = inputs.input_ids.shape[1]
        input_len_image = inputs_image.input_ids.shape[1]
        # print('####')
        # print(inputs.input_ids.shape)
        # print(inputs_image.input_ids.shape)
        # print('####')
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generated_ids = self.model.generate(**inputs_image, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_image = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        messages.append({"role": "assistant", "content": output_text[0]})
        messages_image.append({"role": "assistant", "content": output_text_image[0]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        res_len = inputs.input_ids.shape[1] - input_len
        res_len_image = inputs_image.input_ids.shape[1] - input_len_image
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
            generated = self.model(**inputs)
            generated_image = self.model(**inputs_image)
        #     print(generated)
        # print('#####')
        # print(output_text[0])
        # print('########')
        # print(activations)

        act_image_list = []
        act_text_list = []
        results = {
            "image_paths": image_paths,
            "question_text": question_text,
            "question_image": question_image,
            "answer": output_text[0],
            "answer_image": output_text_image[0],
            "activations": {},
            "differences": {}
        }
        for name in activations.keys():
            act_image = activations[name][-1].squeeze(0)[-res_len:, :]
            act_text = activations[name][-2].squeeze(0)[-res_len_image:, :]

            # Calculate absolute difference and mean them
            # abs_diff_image = torch.square(act_image).mean()
            # abs_diff_text = torch.square(act_text).mean()
            abs_diff_image = act_image.mean(dim=0)
            abs_diff_text = act_text.mean(dim=0)
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



    def get_activation_result_image_text(self, image_paths, question_text, question_image):
        # Hook setup
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
        content_image = []
        content = []
        for image_path in image_paths:
            content_image.append({"type": "image", "image": image_path})
        content_image.append({"type": "text", "text": question_image})
        content.append({"type": "image", "image": image_paths[0]})
        content.append({"type": "text", "text": question_text})
        messages_image = [{"role": "user", "content": content_image}]
        messages = [{"role": "user", "content": content}]
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")
        input_len = inputs.input_ids.shape[1]
        input_len_image = inputs_image.input_ids.shape[1]
        # print('####')
        # print(inputs.input_ids.shape)
        # print(inputs_image.input_ids.shape)
        # print('####')
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generated_ids = self.model.generate(**inputs_image, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_image = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        messages.append({"role": "assistant", "content": output_text[0]})
        messages_image.append({"role": "assistant", "content": output_text_image[0]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        res_len = inputs.input_ids.shape[1] - input_len
        res_len_image = inputs_image.input_ids.shape[1] - input_len_image
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
            generated = self.model(**inputs)
            generated_image = self.model(**inputs_image)
        #     print(generated)
        # print('#####')
        # print(output_text[0])
        # print('########')
        # print(activations)

        act_image_list = []
        act_text_list = []
        results = {
            "image_paths": image_paths,
            "question_text": question_text,
            "question_image": question_image,
            "answer": output_text[0],
            "answer_image": output_text_image[0],
            "activations": {},
            "differences": {}
        }
        for name in activations.keys():
            act_image = activations[name][-1].squeeze(0)[-res_len:, :]
            act_text = activations[name][-2].squeeze(0)[-res_len_image:, :]

            # Calculate absolute difference and mean them
            # abs_diff_image = torch.square(act_image).mean()
            # abs_diff_text = torch.square(act_text).mean()
            abs_diff_image = act_image.mean(dim=0)
            abs_diff_text = act_text.mean(dim=0)
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
        content_image = []
        content = []
        for image_path in image_paths:
            content_image.append({"type": "image", "image": image_path})
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        content_image.append({"type": "text", "text": question})
        content.append({"type": "text", "text": question})
        messages_image = [{"role": "user", "content": content_image}]
        messages = [{"role": "user", "content": content}]
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")
        input_len = inputs.input_ids.shape[1]
        input_len_image = inputs_image.input_ids.shape[1]
        # print('####')

        messages.append({"role": "assistant", "content": answer[0]})
        messages_image.append({"role": "assistant", "content": answer[1]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        res_len = inputs.input_ids.shape[1] - input_len
        res_len_image = inputs_image.input_ids.shape[1] - input_len_image
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
            generated = self.model(**inputs)
            generated_image = self.model(**inputs_image)
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

            act_image = activations[name][-1].squeeze(0)[-res_len_image:, :]
            act_text = activations[name][-2].squeeze(0)[-res_len:, :]
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
    def get_activation_single_result(self, image_paths, question, answer = 'Sorry, I can\'t help you with that', answer_mode = 'safe' ):
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
        content_image = []

        for image_path in image_paths:
            content_image.append({"type": "image", "image": image_path})

        content_image.append({"type": "text", "text": question})
        messages_image = [{"role": "user", "content": content_image}]

        # Prepare for inference

        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to("cuda")

        input_len_image = inputs_image.input_ids.shape[1]
        # print('####')


        messages_image.append({"role": "assistant", "content": answer})

        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        res_len_image = inputs_image.input_ids.shape[1] - input_len_image
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

            generated_image = self.model(**inputs_image)
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
            if answer_mode == 'safe':
                act_image = activations[name][-1].squeeze(0)[-res_len_image+1:-res_len_image+2, :]
            else:
                act_image = activations[name][-1].squeeze(0)[-res_len_image:-res_len_image+1, :]

            # print(activations[name][-1].squeeze(0).shape)
            # print(activations[name][-2].squeeze(0).shape)
            # Calculate absolute difference and mean them
            abs_diff_image = act_image.mean(dim=0)

            #torch.abs(act_image).mean() torch.abs(act_text).mean()
            # Calculate the squared difference and RMS difference


            # Store activations and differences
            results["activations"][name] = {
                "act": abs_diff_image,
            }


        return results

    def resize_image(self,image_path):
        """Resize image to 1/4th of the original size."""
        img = Image.open(image_path)
        original_size = img.size  # (width, height)
        new_size = (original_size[0] // 2, original_size[1] // 2)  # Reduce width and height by half
        img = img.resize(new_size, Image.Resampling.LANCZOS)  # Use LANCZOS for downscaling
        return img

    def get_latent_embedding(self, image_paths, question, answer=['Sorry, I can\'t help you with that',
                                                                  'Sure, the detailed information you requested is provided below:']):
        # Initialize results dictionary


        results = {}

        # Prepare content for text only
        content_text = [{"type": "text", "text": question}]

        # Prepare content for image + text with resized images
        content_image = [{"type": "image", "image": self.resize_image(image_path)} for image_path in image_paths]
        content_image.append({"type": "text", "text": question})
        messages_image = [{"role": "user", "content": content_image}]
        messages_text = [{"role": "user", "content": content_text}]
        # messages_image = [{"role": "user", "content": content_image}]
        text = self.processor.apply_chat_template(
            messages_text, tokenize=False, add_generation_prompt=True
        )
        text_image = self.processor.apply_chat_template(
            messages_image, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_image)

        inputs_text = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs_image = self.processor(
            text=[text_image],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Set model to output hidden states
        self.model.config.output_hidden_states = True

        # Process each combination: answer[0]/answer[1] with text/image
        combinations = {
            'text_safe': (content_text, answer[0]),
            'text_dan': (content_text, answer[1]),
            'image_safe': (content_image, answer[0]),
            'image_dan': (content_image, answer[1]),
        }

        for key, (content, ans) in combinations.items():
            messages = [{"role": "user", "content": content}]
            content_ans = [{"type": "text", "text": ans}]
            messages.append({"role": "assistant", "content": content_ans})

            # Prepare text input
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Prepare vision inputs if needed
            if 'image' in key:

                inputs = self.processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                input_len = inputs_image.input_ids.shape[1]

            else:
                inputs = self.processor(
                    text=[text_input],
                    padding=True,
                    return_tensors="pt",
                )
                input_len = inputs_text.input_ids.shape[1]
            res_len = inputs.input_ids.shape[1] - input_len
            # print(res_len)
            # Run inference and get latent features
            with torch.no_grad():
                outputs = self.model(**inputs)
            # print(outputs.hidden_states[-1].shape)
            # print(input_len)
            # Get last hidden state
            latent_features = outputs.hidden_states[-1][0, -1:, :].mean(dim=0)
            # print(latent_features.shape)
            results[key] = latent_features

        return results


    def eval_prefill(self, image_paths, question, answer):
        # Build messages
        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text += answer
        # print(text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]