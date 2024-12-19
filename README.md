# Safe + Safe = Unsafe?: Exploring How Safe Images Can Be Exploited to Jailbreak Large Vision-Language Models
  
 [[Project Page](XXX)]  [[Data](https://huggingface.co/datasets/Chenhangcui/Safe_Attack_Bench)]  [[Paper](https://arxiv.org/pdf/2411.11496)]
 

## Install


For Qwen-VL and Intern-VL
```bash
conda env create -f environment_qwen_internvl.yml
```
For VILA
```bash
conda env create -f environment_vila.yml
```
## Running the Program

1. **Prepare your inputs:**

 
   - `folder_path` 

2. **Run the script to obtain GPT-4o's results:**

   Execute the main program with the following arguments:

   ```bash
   python SSA_framework.py --folder_path test_folder --output_classify_file_name results.json --json_path output_json/results_mmsafety.json --api_key
   ```

   **Arguments**:
   - `--folder_path`: Path to the folder containing images to process.
   - `--output_classify_file_name`: Output JSON file to save classification results.
   - `--json_path`: Path to the JSON output file for processed results.

   - `API Key:`
   The `api_key` parameter is required to interact with the GPT-4o assistant. You can obtain an API key from [OpenAI](https://platform.openai.com).

3. Obtain Results on open-sourced LVLMs:

```bash
python SSA_open_source_framework.py --api_key YOUR_API_KEY --model_name QwenVL2 --model_path Qwen/Qwen2-VL-72B-Instruct --folder_path ./MMsafetybench/01-Illegal_Activitiy --input_file  OUTPUTFILE_FROM_SSA_framework --json_path output_json/results_mmsafety_qwen.json
```

### Arguments:

- `--api_key`:  
  The API key for interacting with the GPTImageAssistant. You can obtain an API key from [OpenAI](https://platform.openai.com).

- `--model_name`:  
  The name of the model you wish to load for evaluation. Examples include `QwenVL2`, `InternVL`, and `VILA`.

- `--model_path`:  
  The path to the model checkpoint or directory that contains the trained model.

- `--folder_path`:  
  The folder containing the images you want to process. The program will scan and classify the images from this folder.

- `--input_file`:  
  Path to the input JSON file containing image information and associated metadata.

- `--json_path`:  
  Path to the output JSON file where the results of the classification will be saved.





[//]: # (### Example Output)

[//]: # ()
[//]: # (The output file &#40;`results.json`&#41; will contain classified data for each image. For example:)

[//]: # ()
[//]: # (```json)

[//]: # ({)

[//]: # (  "image": "image1.jpg",)

[//]: # (  "self-harm": 0,)

[//]: # (  "celebrity": 1,)

[//]: # (  "violence": 0,)

[//]: # (  "creating illegal objects": 0)

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (Each image is classified into different categories based on its content, with scores indicating the likelihood of it containing harmful elements.)

[//]: # ()
[//]: # (---)

[//]: # ( )
[//]: # (### License)

[//]: # ()
[//]: # (This project is licensed under the MIT License - see the [LICENSE]&#40;LICENSE&#41; file for details.)
