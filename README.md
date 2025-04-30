****# Safe + Safe = Unsafe?: Exploring How Safe Images Can Be Exploited to Jailbreak Large Vision-Language Models
  
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
```markdown
# Danger-Behaviour QA Generator

A script that scans images for severe or harmful behaviour, generates two rounds
of vision–language questions &amp; answers, creates a *“jailbreak”* image for
each case, and moderates the final content with OpenAI.

---

```markdown
# Danger-Behaviour QA Generator

A toolkit that scans images for severe or harmful behaviour, generates two
rounds of vision–language Q & A, creates a *“jailbreak”* image for each case,
and moderates the final content with OpenAI.

---


---

## Prepare assets

```
project/
├── images/                # input pictures (.jpg / .png / .gif …)
│   ├── img_001.jpg
│   └── img_002.png
├── prompt/
│   └── 4o_sys_914.txt     # system prompt for GPT-4o
└── analyse_images.py      # refactored main script
```

*Place all images inside **images/**  
and keep the system prompt in **prompt/4o_sys_914.txt**
(or supply another path via `--system-prompt`).*

---

## Run

```bash
python analyse_images.py images \
    --output results.json \
    --openai-api-key YOUR_OPENAI_API_KEY
```

| Argument              | Description                                              |
|-----------------------|----------------------------------------------------------|
| `images`              | Folder with the input pictures.                          |
| `--output`            | Destination JSON (default: `results.json`).             |
| `--gpt4v-api`         | Key / base-URL for **GPTImageAssistant**.                |
| `--openai-api-key`    | Standard OpenAI REST API key.                            |
| `--system-prompt`     | (Optional) custom system prompt file.                    |
| `--model-version`     | (Optional) model-version string for the assistant.       |

---

## What happens internally

1. **Detect dangerous behaviour** in each image.  
2. **Choose a keyword** (noun or verb) related to that behaviour but *not*
   explicitly shown.  
3. **Download ten images**, pick the best match via CLIP, and save it to
   `images/jailbreak/`.  
4. **Generate Q-A Round 1** (reasoning question about dangerous steps).  
5. **Rewrite the question** into an imperative, abstract form and generate
   **Q-A Round 2**.  
6. **Moderate** the behaviour + prompt + answer with
   *omni-moderation-latest*.  
7. **Append results** to the output JSON; repeat for each behaviour.

---

## Output format (excerpt)

```json
[
  {
    "image": "img_001.jpg",
    "dangerous_behavior": "self-harm - suicidal tendency",
    "safe_image": "images/jailbreak/img_001_rope.png",
    "first_round_question": "...",
    "first_round_answer": "...",
    "second_round_question": "...",
    "second_round_answer": "...",
    "second_round_moderation": {
      "id": "...",
      "model": "omni-moderation-latest",
      "results": [...]
    }
  }
]
```

Each object captures the full pipeline artefacts for one **input image**.

---



```
****
