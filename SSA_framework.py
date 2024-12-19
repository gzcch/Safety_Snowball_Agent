import torch
import clip
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


def clip_filter_by_role(images, role, model, preprocess, device="cuda:0"):

    role_text = clip.tokenize([role]).to(device)


    image_tensors = []
    for image in images:
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or path to image, but got {type(image)}")

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_tensors.append(image_tensor)


    with torch.no_grad():
        text_features = model.encode_text(role_text)


        image_features = torch.cat([model.encode_image(image_tensor) for image_tensor in image_tensors])


        similarities = (text_features @ image_features.T).softmax(dim=-1).cpu().numpy().flatten()

    sorted_indices = similarities.argsort()[::-1]  # 从高到低排序
    num_images = len(sorted_indices)


    top_80_percent_indices = sorted_indices[:int(0.8 * num_images)]


    best_image_index = random.choice(top_80_percent_indices)
    best_image = images[best_image_index]

    return best_image



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process some inputs.')

    parser.add_argument('--query_file', type=str, default='prompt/query.json', help='Path to query.json')
    parser.add_argument('--seed_prompt_file', type=str, default='prompt/seed_prompt.json',
                        help='Path to seed_prompt.json')
    parser.add_argument('--prompt_classify_file', type=str, default='prompt/prompt_classify.txt',
                        help='Path to prompt_classify.txt')
    parser.add_argument('--folder_path', type=str, default='test_folder',
                        help='Path to folder containing images')
    parser.add_argument('--output_classify_file_name', type=str, default='xx_text.json', help='Name of output classify file')
    parser.add_argument('--sys_file', type=str, default='prompt/4o_sys.txt', help='Path to sys file')
    parser.add_argument('--json_path', type=str, default='output_json/results_mmsafety.json',
                        help='Path to json results file')
    parser.add_argument('--api_key', type=str, default='', help='API key for GPTImageAssistant')

    args = parser.parse_args()

    from gpt4v_assistant import GPTImageAssistant
    from search_api import CrawlerGoogleImages
    import json
    with open(args.query_file, 'r', encoding='utf-8') as file:
        query = json.load(file)

    with open(args.seed_prompt_file, 'r', encoding='utf-8') as file:
        seed_prompt = json.load(file)



    api_key = args.api_key


    evaluator = GPTImageAssistant(api_key)



    with open(args.prompt_classify_file, 'r', encoding='utf-8') as file:
        prompt_classify = file.read()


    from classify import gpt4v_eval

    # normal
    folder_path = args.folder_path


    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.endswith(('jpg', 'png', 'jpeg'))]

    class_list = []
    co = 0
    max_retries = 3

    output_classify_file_name = args.output_classify_file_name

    # Read existing results from the output file to avoid reprocessing
    existing_files = set()
    if os.path.exists(output_classify_file_name):
        with open(output_classify_file_name, 'r') as infile:
            for line in infile:
                result = json.loads(line)
                existing_files.add(result['image'])  # Add the 'image' field to the set of processed files

    with open(output_classify_file_name, 'a') as outfile:  # Open for appending
        for image_path in image_paths:
            co += 1

            # Get the image file name
            file_name = os.path.basename(image_path)

            # # Skip if the image has already been processed and exists in the output file
            # if file_name in existing_files:
            #     print(f"Skipping {file_name} as it has already been processed.")
            #     continue

            retries = 0
            success = False
            while retries < max_retries and not success:
                try:
                    # Get model's response (assuming gpt4v_eval function is defined)
                    response = gpt4v_eval(image_path, prompt_classify)

                    # Process the response
                    response = response.strip()
                    response_parts = response.split('\n')

                    # Create a dictionary for scores
                    scores = {}

                    for part in response_parts:
                        if ':' in part:
                            key, value = part.split(':')
                            scores[key.strip()] = int(value.strip())
                    print(scores)

                    # Ensure all required fields are present
                    keys = ['self-harm', 'celebrity', 'violence', 'creating illegal objects']
                    for key in keys:
                        if key not in scores:
                            scores[key] = '0'  # Default value if missing

                    # Create a result object
                    result = {
                        'image': file_name,
                        'self-harm': scores['self-harm'],
                        'celebrity': scores['celebrity'],
                        'violence': scores['violence'],
                        'creating illegal objects': scores['creating illegal objects']
                    }

                    # Append the result to the output file
                    json.dump(result, outfile)
                    outfile.write('\n')

                    # Mark as successful
                    success = True

                except Exception as e:
                    retries += 1
                    print(f"Error processing {image_path} (Attempt {retries}/{max_retries}): {e}")

                    # If retries exceed the max limit, set all scores to 0
                    if retries == max_retries:
                        print(f"Failed to process {image_path} after {max_retries} retries. Setting scores to 0.")
                        scores = {key: '0' for key in
                                  ['self-harm', 'celebrity', 'violence', 'creating illegal objects']}
                        result = {
                            'image': file_name,
                            'self-harm': scores['self-harm'],
                            'celebrity': scores['celebrity'],
                            'violence': scores['violence'],
                            'creating illegal objects': scores['creating illegal objects']
                        }
                        json.dump(result, outfile)
                        outfile.write('\n')


    class_list = []

    # Ours read jsonl
    with open(output_classify_file_name, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            image_path = os.path.join(folder_path, data['image'])


            all_zero = all(
                int(data[key]) == 0 for key in ['self-harm', 'celebrity', 'violence', 'creating illegal objects']
            )


            if all_zero:
                # print(image_path)
                response = evaluator.gpt4v_eval([image_path], prompt_classify)
                response = response.strip()
                response_parts = response.split('\n')

                scores = {}
                for part in response_parts:
                    if ':' in part:
                        key, value = part.split(':')
                        scores[key.strip()] = int(value.strip())

                result = {
                    'image': data['image'],
                    'self-harm': scores['self-harm'],
                    'celebrity': scores['celebrity'],
                    'violence': scores['violence'],
                    'creating illegal objects': scores['creating illegal objects for illegal purposes']
                }

            else:
                result = {
                    'image': data['image'],
                    'self-harm': int(data['self-harm']),
                    'celebrity': int(data['celebrity']),
                    'violence': int(data['violence']),
                    'creating illegal objects': int(data['creating illegal objects'])
                }

            class_list.append(result)

    sys_file = args.sys_file


    from tqdm import tqdm

    json_path = args.json_path
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            results_list = json.load(json_file)
            # Create a dictionary mapping image names to result items
            results = results_list
    else:
        # If the path does not exist, create the folder and file
        os.makedirs(os.path.dirname(json_path), exist_ok=True)  # Create the folder if it doesn't exist
        with open(json_path, 'w') as json_file:
            json.dump([], json_file)  # Initialize the file with an empty list
        results = []

    # Parameters
    max_attempts = 3  # Maximum number of attempts for regeneration
    for item in tqdm(class_list):
        image_name = item['image']
        image_path = os.path.join(folder_path, image_name)

        if check_image_exists_any_key(image_name, results):
            print(image_name)
            continue

        result_item = {}
        # Process 'self-harm' category
        if item['self-harm'] == 1:
            category = 'self-harm'

            attempt = 0
            success = False
            while attempt < max_attempts and not success:
                res = evaluator.gpt4v_eval([image_path], query['self-harm'], sys_file)
                action = res.split(' – ')[1]
                crawler = CrawlerGoogleImages(action)
                images = crawler.run(num_images=10)
                image = clip_filter_by_role(images, action, model.cuda(), preprocess, device)
                subfolder_path = os.path.join(folder_path, 'jailbreak')
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                adv_image_save_path = os.path.join(subfolder_path, image_name.split('.')[0] + '_harm.png')
                image.save(adv_image_save_path)
                adv_res = evaluator.gpt4v_eval([image_path, image], seed_prompt['self-harm'][0], sys_file)
                adv_res_icl = evaluator.icl_response(
                    [image_path, image],
                    [seed_prompt['self-harm'][0], adv_res],
                    seed_prompt['self-harm-serious'][0] + action,
                    sys_file
                )
                eval_res_icl = evaluator.policy_evaluation_serious(
                    [image_path, image],
                    seed_prompt['self-harm'][0],
                    adv_res,
                    adv_res_icl,
                    seed_prompt['eval-for-serious'][0]
                )

                # Extract response_scores
                numbers = re.findall(r'\d+', eval_res_icl)
                response_scores = [int(num) for num in numbers]
                try:
                    if response_scores[1] <= response_scores[-1]:
                        success = True
                    else:
                        attempt += 1
                except Exception as e:
                    # 如果发生错误，将 response_scores 设置为 [0,0,0,0]
                    print(f"Error occurred: {e}")
                    response_scores = [0, 0, 0, 0]

                if success or attempt == max_attempts:
                    result_item[category] = {
                        'image': image_name,
                        'adv_image': os.path.join('jailbreak', image_name.split('.')[0] + '_harm.png'),
                        'adv_res': adv_res,
                        'adv_res_icl': adv_res_icl,
                        'eval_res_icl': eval_res_icl,
                        'question': seed_prompt['self-harm'][0],
                        'serious_question': seed_prompt['self-harm-serious'][0] + action,
                        'score': [response_scores[1], response_scores[-1]]
                    }
                    # Update the results dictionary
                    # results[image_name] = result_item
                else:
                    # Handle failure case if necessary
                    pass

        # Process 'celebrity' category
        if item['celebrity'] == 1:
            category = 'celebrity'
            attempt = 0
            success = False
            while attempt < max_attempts and not success:
                res = evaluator.gpt4v_eval([image_path], query['celebrity'], sys_file)
                pattern = r"Information - (.+)"
                information = re.findall(pattern, res)
                diagram = CulturalIdentityDiagram(categories=information)
                image_syn = diagram.draw_diagram()
                subfolder_path = os.path.join(folder_path, 'jailbreak')
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                adv_image_save_path = os.path.join(subfolder_path, image_name.split('.')[0] + '_pri.png')
                image_syn.save(adv_image_save_path)
                adv_res = evaluator.gpt4v_eval([image_path, image_syn], seed_prompt['celebrity'][0], sys_file)
                adv_res_icl = evaluator.icl_response(
                    [image_path],
                    [seed_prompt['celebrity'][0], adv_res],
                    seed_prompt['celebrity-serious'][0],
                    sys_file
                )
                eval_res_icl = evaluator.policy_evaluation_serious(
                    [image_path, image_syn],
                    seed_prompt['celebrity'][0],
                    adv_res,
                    adv_res_icl,
                    seed_prompt['eval-for-serious-privacy'][0]
                )

                # Extract response_scores
                numbers = re.findall(r'\d+', eval_res_icl)
                response_scores = [int(num) for num in numbers]
                try:
                    if response_scores[1] <= response_scores[-1]:
                        success = True
                    else:
                        attempt += 1
                except Exception as e:
                    # 如果发生错误，将 response_scores 设置为 [0,0,0,0]
                    print(f"Error occurred: {e}")
                    response_scores = [0, 0, 0, 0]
                if success or attempt == max_attempts:
                    result_item[category] = {
                        'image': image_name,
                        'adv_image': os.path.join('jailbreak', image_name.split('.')[0] + '_pri.png'),
                        'eval_res_icl': eval_res_icl,
                        'adv_res_icl': adv_res_icl,
                        'adv_res': adv_res,
                        'question': seed_prompt['celebrity'][0],
                        'serious_question': seed_prompt['celebrity-serious'][0],
                        'score': [response_scores[1], response_scores[-1]]
                    }
                    # Update the results dictionary
                    # results[image_name] = result_item
                else:
                    # Handle failure case if necessary
                    pass

        # Process 'violence' category
        if item['violence'] == 1:
            category = 'violence'

            attempt = 0
            success = False
            while attempt < max_attempts and not success:
                res = evaluator.gpt4v_eval([image_path], query['violence'], sys_file)
                role = res.split(' - ')[1]
                crawler = CrawlerGoogleImages(role)
                images = crawler.run(num_images=10)
                image = clip_filter_by_role(images, role, model.cuda(), preprocess, device)
                subfolder_path = os.path.join(folder_path, 'jailbreak')
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                adv_image_save_path = os.path.join(subfolder_path, image_name.split('.')[0] + '_violence.png')
                image.save(adv_image_save_path)
                adv_res = evaluator.gpt4v_eval(
                    [image_path, image],
                    seed_prompt['violence'][0],
                    'prompt/4o_sys.txt'
                )
                adv_res_icl = evaluator.icl_response(
                    [image_path, image],
                    [seed_prompt['violence'][0], adv_res],
                    seed_prompt['violence-serious'][0],
                    'prompt/4o_sys.txt'
                )
                eval_res_icl = evaluator.policy_evaluation_serious(
                    [image_path, image],
                    seed_prompt['violence'][0],
                    adv_res,
                    adv_res_icl,
                    seed_prompt['eval-for-serious-violence'][0]
                )

                # Extract response_scores
                numbers = re.findall(r'\d+', eval_res_icl)
                response_scores = [int(num) for num in numbers]
                # print(eval_res_icl)
                try:
                    if response_scores[1] <= response_scores[-1]:
                        success = True
                    else:
                        attempt += 1
                except Exception as e:
                    # 如果发生错误，将 response_scores 设置为 [0,0,0,0]
                    print(f"Error occurred: {e}")
                    response_scores = [0, 0, 0, 0]
                if success or attempt == max_attempts:
                    result_item[category] = {
                        'image': image_name,
                        'adv_image': os.path.join('jailbreak', image_name.split('.')[0] + '_violence.png'),
                        'eval_res_icl': eval_res_icl,
                        'adv_res_icl': adv_res_icl,
                        'adv_res': adv_res,
                        'question': seed_prompt['violence'][0],
                        'serious_question': seed_prompt['violence-serious'][0],
                        'score': [response_scores[1], response_scores[-1]]
                    }
                else:
                    pass

        # Process 'creating illegal objects' category
        if item['creating illegal objects'] == 1:
            category = 'creating_illegal_objects'

            attempt = 0
            success = False
            while attempt < max_attempts and not success:
                res = evaluator.gpt4v_eval([image_path], query['violence'], sys_file)
                role = res.split(' - ')[1]
                crawler = CrawlerGoogleImages(role)
                images = crawler.run(num_images=10)
                image = clip_filter_by_role(images, role, model.cuda(), preprocess, device)
                subfolder_path = os.path.join(folder_path, 'jailbreak')
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                adv_image_save_path = os.path.join(subfolder_path, image_name.split('.')[0] + '_obj.png')
                image.save(adv_image_save_path)
                adv_res = evaluator.gpt4v_eval(
                    [image_path, image],
                    seed_prompt['creating illegal objects'][0],
                    'prompt/4o_sys.txt'
                )
                adv_res_icl = evaluator.icl_response(
                    [image_path, image],
                    [seed_prompt['creating illegal objects'][0], adv_res],
                    seed_prompt['creating illegal objects-serious'][0],
                    'prompt/4o_sys.txt'
                )
                eval_res_icl = evaluator.policy_evaluation_serious(
                    [image_path, image],
                    seed_prompt['creating illegal objects'][0],
                    adv_res,
                    adv_res_icl,
                    seed_prompt['eval-for-serious-violence'][0]
                )

                # Extract response_scores
                numbers = re.findall(r'\d+', eval_res_icl)
                response_scores = [int(num) for num in numbers]
                try:
                    if response_scores[1] <= response_scores[-1]:
                        success = True
                    else:
                        attempt += 1
                except Exception as e:
                    # 如果发生错误，将 response_scores 设置为 [0,0,0,0]
                    print(f"Error occurred: {e}")
                    response_scores = [0, 0, 0, 0]
                if success or attempt == max_attempts:
                    result_item[category] = {
                        'image': image_name,
                        'adv_image': os.path.join('jailbreak', image_name.split('.')[0] + '_obj.png'),
                        'eval_res_icl': eval_res_icl,
                        'adv_res_icl': adv_res_icl,
                        'adv_res': adv_res,
                        'question': seed_prompt['creating illegal objects'][0],
                        'serious_question': seed_prompt['creating illegal objects-serious'][0],
                        'score': [response_scores[1], response_scores[-1]]
                    }
                else:
                    pass

        # Update the results dictionary after processing all categories for the current image

        # Save the updated results back to the JSON file
        results.append(result_item)
        print(result_item)
        with open(json_path, 'w') as json_file:
            json.dump(results, json_file)
        # break


if __name__ == "__main__":
    main()
