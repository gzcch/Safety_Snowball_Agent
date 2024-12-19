import torch
# import clip
from PIL import Image
import os
import sys
from syn_image_assistant import CulturalIdentityDiagram

sys.stdout.reconfigure(encoding='utf-8')
import torch
import random
from PIL import Image

import re
# 封装成函数
import torch
import random
from PIL import Image


def check_image_exists_any_key(image_name, data):
    for entry in data:
        for key, value in entry.items():
            if isinstance(value, dict) and value.get("image") == image_name:
                return True
    return False


# 使用方法：
# 假设你已经加载了CLIP模型，并且有图片列表
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

from gpt4v_assistant import GPTImageAssistant

def load_model_by_name(model_name, path):
    if model_name == 'VILA':
        from VILA_assistant import VILA
        return VILA(path, conv_mode='hermes-2')
    elif model_name == 'QwenVL2':
        from Qwenvl_assitant import QwenVL2Assistant
        return QwenVL2Assistant(path)
    elif model_name == 'InternVL':
        from InternVL_assitant import InternVL
        return InternVL(path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process some inputs.')

    parser.add_argument('--api_key', type=str, default='your_api_key_here', help='API key for GPTImageAssistant')
    parser.add_argument('--model_name', type=str, default='QwenVL2', help='Name of the model to load')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2-VL-72B-Instruct', help='Path to the model')
    parser.add_argument('--folder_path', type=str, default='./MMsafetybench/01-Illegal_Activitiy', help='Path to folder containing images')
    parser.add_argument('--input_file', type=str, default='output_json/results_mmsafety.json', help='Path to input JSON file')
    parser.add_argument('--json_path', type=str, default='output_json/results_mmsafety_qwen.json', help='Path to output JSON results file')
    parser.add_argument('--sys_file', type=str, default='prompt/4o_sys.txt', help='Path to sys file')
    parser.add_argument('--prompt_classify_file', type=str, default='prompt/prompt_classify.txt', help='Path to prompt_classify.txt')
    parser.add_argument('--query_file', type=str, default='prompt/query.json', help='Path to query.json')
    parser.add_argument('--seed_prompt_file', type=str, default='prompt/seed_prompt.json', help='Path to seed_prompt.json')
    parser.add_argument('--max_attempts', type=int, default=3, help='Maximum number of attempts for regeneration')

    args = parser.parse_args()

    # Initialize GPTImageAssistant with API key
    api_key = args.api_key
    evaluator_gpt = GPTImageAssistant(api_key)

    # Load the specified model
    evaluator = load_model_by_name(args.model_name, args.model_path)

    from search_api import CrawlerGoogleImages
    import json

    with open(args.query_file, 'r', encoding='utf-8') as file:
        query = json.load(file)

    with open(args.seed_prompt_file, 'r', encoding='utf-8') as file:
        seed_prompt = json.load(file)

    with open(args.prompt_classify_file, 'r', encoding='utf-8') as file:
        prompt_classify = file.read()

    # Process each image
    folder_path = args.folder_path
    sys_file = args.sys_file

    results = []

    from tqdm import tqdm
    input_file = args.input_file

    with open(input_file, 'r') as json_file:
        aba_list = json.load(json_file)
    json_path = args.json_path
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            results_list = json.load(json_file)
            # Create a dictionary mapping image names to result items
            results = results_list
    else:
        results = []

    max_attempts = args.max_attempts

    for item in tqdm(aba_list):
        try:
            flag = 0
            result_item = {}
            for key in item.keys():
                if key == 'self-harm':
                    temp_item = item[key]
                    if check_image_exists_any_key(temp_item['image'], results):
                        print(temp_item['image'])
                        flag = 1
                        break
                    attempt = 0
                    success = False

                    while attempt < max_attempts and not success:
                        image_path = os.path.join(folder_path, temp_item['image'])

                        adv_image_path = os.path.join(folder_path, temp_item['adv_image']).replace("\\", "/")

                        adv_res = evaluator.eval([image_path, adv_image_path], temp_item['question'])
                        adv_res_icl = evaluator.icl_response([image_path, adv_image_path],
                                                             [temp_item['question'], adv_res],
                                                             temp_item['serious_question'])
                        eval_res = evaluator_gpt.policy_evaluation_serious([image_path], temp_item['question'], adv_res,
                                                                           adv_res_icl, seed_prompt['eval-for-serious'][0])

                        numbers = re.findall(r'\d+', eval_res)
                        response_scores = [int(num) for num in numbers]

                        if response_scores[1] <= response_scores[-1]:
                            success = True
                        else:
                            attempt += 1

                        if success or attempt == max_attempts:
                            result_item[key] = {
                                'image': temp_item['image'],
                                'adv_image': temp_item['adv_image'],
                                'eval_res_icl': eval_res,
                                'adv_res_icl': adv_res_icl,
                                'adv_res': adv_res,
                                'eval_res': eval_res,
                                'question': temp_item['question'],
                                'serious_question': temp_item['serious_question'],
                                'score': [response_scores[1], response_scores[-1]]
                            }
                        else:
                            # Handle failure case if necessary
                            pass

                if key == 'violence':
                    temp_item = item[key]
                    if check_image_exists_any_key(temp_item['image'], results):
                        print(temp_item['image'])
                        flag = 1
                        break
                    attempt = 0
                    success = False

                    while attempt < max_attempts and not success:
                        image_path = os.path.join(folder_path, temp_item['image'])
                        adv_image_path = os.path.join(folder_path, temp_item['adv_image']).replace("\\", "/")
                        adv_res = evaluator.eval([image_path, adv_image_path], temp_item['question'])
                        adv_res_icl = evaluator.icl_response([image_path, adv_image_path],
                                                             [temp_item['question'], adv_res],
                                                             temp_item['serious_question'])
                        eval_res = evaluator_gpt.policy_evaluation_serious([image_path, adv_image_path],
                                                                           temp_item['question'], adv_res, adv_res_icl,
                                                                           seed_prompt['eval-for-serious-violence'][0])

                        numbers = re.findall(r'\d+', eval_res)
                        response_scores = [int(num) for num in numbers]

                        if response_scores[1] <= response_scores[-1]:
                            success = True
                        else:
                            attempt += 1

                        if success or attempt == max_attempts:
                            result_item[key] = {
                                'image': temp_item['image'],
                                'adv_image': temp_item['adv_image'],
                                'eval_res_icl': eval_res,
                                'adv_res_icl': adv_res_icl,
                                'adv_res': adv_res,
                                'eval_res': eval_res,
                                'question': temp_item['question'],
                                'serious_question': temp_item['serious_question'],
                                'score': [response_scores[1], response_scores[-1]]
                            }
                        else:
                            # Handle failure case if necessary
                            pass

                if key == 'creating_illegal_objects':
                    temp_item = item[key]
                    if check_image_exists_any_key(temp_item['image'], results):
                        print(temp_item['image'])
                        flag = 1
                        break
                    attempt = 0
                    success = False

                    while attempt < max_attempts and not success:
                        image_path = os.path.join(folder_path, temp_item['image'])
                        adv_image_path = os.path.join(folder_path, temp_item['adv_image']).replace("\\", "/")
                        adv_res = evaluator.eval([image_path, adv_image_path], temp_item['question'])
                        adv_res_icl = evaluator.icl_response([image_path, adv_image_path],
                                                             [temp_item['question'], adv_res],
                                                             temp_item['serious_question'])
                        eval_res = evaluator_gpt.policy_evaluation_serious([image_path, adv_image_path],
                                                                           temp_item['question'], adv_res, adv_res_icl,
                                                                           seed_prompt['eval-for-serious-violence'][0])

                        numbers = re.findall(r'\d+', eval_res)
                        response_scores = [int(num) for num in numbers]

                        if response_scores[1] <= response_scores[-1]:
                            success = True
                        else:
                            attempt += 1

                        if success or attempt == max_attempts:
                            result_item[key] = {
                                'image': temp_item['image'],
                                'adv_image': temp_item['adv_image'],
                                'eval_res_icl': eval_res,
                                'adv_res_icl': adv_res_icl,
                                'adv_res': adv_res,
                                'eval_res': eval_res,
                                'question': temp_item['question'],
                                'serious_question': temp_item['serious_question'],
                                'score': [response_scores[1], response_scores[-1]]
                            }
                        else:
                            # Handle failure case if necessary
                            pass

                if key == 'celebrity':
                    temp_item = item[key]
                    if check_image_exists_any_key(temp_item['image'], results):
                        print(temp_item['image'])
                        flag = 1
                        break
                    attempt = 0
                    success = False

                    while attempt < max_attempts and not success:
                        image_path = os.path.join(folder_path, temp_item['image'])
                        adv_image_path = os.path.join(folder_path, temp_item['adv_image']).replace("\\", "/")

                        adv_res = evaluator.eval([image_path, adv_image_path], temp_item['question'])
                        adv_res_icl = evaluator.icl_response([image_path, adv_image_path],
                                                             [temp_item['question'], adv_res],
                                                             temp_item['serious_question'])
                        eval_res = evaluator_gpt.policy_evaluation_serious([image_path, adv_image_path],
                                                                           temp_item['question'], adv_res, adv_res_icl,
                                                                           seed_prompt['eval-for-serious-privacy'][0])

                        numbers = re.findall(r'\d+', eval_res)
                        response_scores = [int(num) for num in numbers]

                        if response_scores[1] <= response_scores[-1]:
                            success = True
                        else:
                            attempt += 1

                        if success or attempt == max_attempts:
                            result_item[key] = {
                                'image': temp_item['image'],
                                'adv_image': temp_item['adv_image'],
                                'eval_res_icl': eval_res,
                                'adv_res_icl': adv_res_icl,
                                'adv_res': adv_res,
                                'eval_res': eval_res,
                                'question': temp_item['question'],
                                'serious_question': temp_item['serious_question'],
                                'score': [response_scores[1], response_scores[-1]]
                            }
                        else:
                            # Handle failure case if necessary
                            pass

            if flag == 0:
                results.append(result_item)
                print(result_item)
                with open(json_path, 'w') as json_file:
                    json.dump(results, json_file)
        except Exception as e:
            print(f"Error processing item: {item}. Error: {e}")

if __name__ == "__main__":
    main()
