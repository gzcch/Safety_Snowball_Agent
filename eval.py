# import json
# import re
# import numpy as np
#
# def extract_numbers_from_string(response_string):
#     # 定义正则表达式来匹配数字，包含可选的花括号
#     pattern = r'Response_1\s*-\s*\{?(\d+)\}?\s*\n*\s*Response_2\s*-\s*\{?(\d+)\}?'
#
#     # 使用正则表达式查找并提取数字
#     match = re.search(pattern, response_string)
#
#     if match:
#         # 提取匹配的数字
#         num_1 = int(match.group(1))
#         num_2 = int(match.group(2))
#         return num_1, num_2
#     else:
#         return None, None
# # Function to load and parse a JSON file
# def load_json_from_file(filename):
#     try:
#         with open(filename, 'r', encoding='utf-8') as file:
#             data = json.load(file)
#         return data
#     except FileNotFoundError:
#         return f"File {filename} not found."
#     except json.JSONDecodeError:
#         return f"Error decoding JSON in file {filename}."
#
# # Replace 'your_filename.json' with the actual file name
# filename = 'rename_full_create/results_sub_new_prompt_all_cele_sample.json'
# data = load_json_from_file(filename)
#
#
# # Function to extract eval_res_icl scores and eval_res attack success rate
# def extract_scores(data):
#     categories = {}  # Dictionary to store results for each category (celebrity, A, B)
#
#     for entry in data:
#         for key, value in entry.items():
#             # Initialize category if not already in the dictionary
#             if key not in categories:
#                 categories[key] = {
#                     "Response_1": [],
#                     "Response_2": [],
#                     "Attack_Success": []
#                 }
#
#             # Extract eval_res_icl and eval_res
#             eval_res_icl = value["eval_res_icl"]
#
#             numbers = re.findall(r'\d+', eval_res_icl)
#             # Convert all numbers from strings to integers
#             response_scores = [int(num) for num in numbers]
#             print(eval_res_icl)
#             print(numbers)
#             # eval_res = value["eval_res"]
#             # input_str = "Response_1 - M234abc\nResponse_2 - N987"
#             # print(eval_res_icl)
#             # # Using regular expressions to find numbers associated with 'M' and 'N'
#             # response_scores = re.findall(r'Response_\d+[ -:]+(?:\{)?(\d+)(?:\})?', eval_res_icl)
#             # print(response_scores)
#             # Output the result
#             # print(matches)
#             # print(eval_res_icl)
#             # Extract response scores from eval_res_icl
#             # print(eval_res_icl)
#             # response_scores = extract_numbers_from_string(eval_res_icl)
#             # print(response_scores)
#             response_1_score =  response_scores[1]
#             response_2_score = response_scores[-1]
#             categories[key]["Response_1"].append(response_1_score)
#             categories[key]["Response_2"].append(response_2_score)
#
#             # Extract attack success rate (1 means success)
#             # if "1 (Attack successful)" in eval_res:
#             #     categories[key]["Attack_Success"].append(1)
#             # else:
#             #     categories[key]["Attack_Success"].append(0)
#
#     # Calculate average scores for each category
#     average_scores = {}
#     all_response_1 = []
#     all_response_2 = []
#     all_attack_success = []
#
#     # Calculate average scores for each category
#     for category, scores in categories.items():
#         # print(scores["Response_1"])
#         for score in range(len(scores["Response_1"])):
#             if scores["Response_2"][score] == None:
#                 scores["Response_1"][score] = 3
#         for score in range(len(scores["Response_2"])):
#
#             if scores["Response_2"][score] == None:
#                 scores["Response_2"][score] = 3
#
#         print(scores)
#         average_scores[category] = {
#             "Average_Response_1": np.mean(scores["Response_1"]),
#             "Average_Response_2": np.mean(scores["Response_2"]),
#             # "Average_Attack_Success_Rate": np.mean(scores["Attack_Success"]),
#             "Num": len(scores["Response_1"])
#         }
#
#         # Collect scores for overall calculation
#         all_response_1.extend(scores["Response_1"])
#         all_response_2.extend(scores["Response_2"])
#         # all_attack_success.extend(scores["Attack_Success"])
#
#     # Calculate overall averages
#     overall_averages = {
#         "Average_Response_1": np.mean(all_response_1),
#         "Average_Response_2": np.mean(all_response_2),
#         # "Average_Attack_Success_Rate": np.mean(all_attack_success),
#         "Num": len(all_response_1)
#     }
#     average_scores ["all"] = overall_averages
#     return average_scores
#
# # Get the results
# average_scores = extract_scores(data)
# print(average_scores)
# # # Convert the result to a DataFrame for better visualization
# # average_scores_df = pd.DataFrame(average_scores).T
#
# # # Display the results
# # tools.display_dataframe_to_user(name="Average Scores for Each Category", dataframe=average_scores_df)
#
# # # Show the average scores for reference
# # average_scores


import json
from collections import defaultdict

# Replace 'data.json' with your JSON file name
with open('safe_bench/results_internvl_direct.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize dictionaries to hold total scores and counts
# total_scores = defaultdict(lambda: [0, 0])
# counts = defaultdict(int)
#
# # Overall totals
# overall_total_score_0 = 0
# overall_total_score_1 = 0
# overall_count = 0
#
# # Iterate over each item in the data
# # Initialize dictionaries to store total scores and counts per key
# total_scores = {}
# counts = {}

# Initialize lists to store overall totals and counts per score index
overall_total_scores = []
overall_counts = []
total_scores = {}
counts = {}

for item in data:
    for key, value in item.items():
        scores = value.get('score', [])
        if scores:
            # Ensure total_scores[key] and counts[key] are lists of appropriate length
            if key not in total_scores:
                total_scores[key] = [0] * len(scores)
                counts[key] = [0] * len(scores)
            else:
                # Extend lists if necessary
                if len(total_scores[key]) < len(scores):
                    total_scores[key].extend([0] * (len(scores) - len(total_scores[key])))
                    counts[key].extend([0] * (len(scores) - len(counts[key])))
            for i, score in enumerate(scores):
                total_scores[key][i] += score
                counts[key][i] += 1

            # Update overall totals and counts
            if len(overall_total_scores) < len(scores):
                overall_total_scores.extend([0] * (len(scores) - len(overall_total_scores)))
                overall_counts.extend([0] * (len(scores) - len(overall_counts)))
            for i, score in enumerate(scores):
                overall_total_scores[i] += score
                overall_counts[i] += 1

# Calculate and print average scores for each key
print("Average Scores for Each Key:\n")
for key in total_scores:
    print(f"{key}:")
    for i in range(len(total_scores[key])):
        if counts[key][i] != 0:
            avg_score = total_scores[key][i] / counts[key][i]
            print(f"  Average score[{i}]: {avg_score:.2f}")
        else:
            print(f"  Average score[{i}]: N/A")
    print()

# Calculate and print overall average scores
print("Overall Average Scores:")
for i in range(len(overall_total_scores)):
    if overall_counts[i] != 0:
        overall_avg_score = overall_total_scores[i] / overall_counts[i]
        print(f"  Overall average score[{i}]: {overall_avg_score:.2f}")
    else:
        print(f"  Overall average score[{i}]: N/A")
