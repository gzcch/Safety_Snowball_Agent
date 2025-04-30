import os
import sys
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

sys.stdout.reconfigure(encoding='utf-8')


class Llama3VisionAssistant:
    """
    Using the meta-llama/Llama-3.2-11B-Vision-Instruct multimodal model as an example,
    this demonstrates the implementation of methods like eval and icl_response similar to MiniCPMAssistant.
    """
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ):
        """
        Initialize the model and processor.
        If a local checkpoint path is required, replace model_id with the local directory.
        """
        print(f"Loading model: {model_id} ...")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Model and processor successfully loaded.")

    def encode_image(self, image_input):
        """
        Convert an image path or PIL.Image object to an RGB-format PIL.Image.
        No need to convert to base64; this example directly uses PIL for processing.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, str):
            if image_input.startswith("http"):
                # If the input is a URL, load using requests
                resp = requests.get(image_input, stream=True)
                return Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                # Local file path
                return Image.open(image_input).convert('RGB')
        else:
            raise ValueError("image_input must be an image path or a PIL.Image object")

    # def eval(self, image_paths, prompt="Describe these images", system_file=None):
    #     """
    #     Single-turn dialogue: Provide several images and a prompt for inference.
    #     image_paths: A list, with each element being an image file path or a PIL.Image object.
    #     prompt: User input
    #     system_file: Optional path to a system message file (if needed).
    #     """
    #     if not image_paths:
    #         raise ValueError("Please provide at least one image path or PIL.Image object.")
    #
    #     # Load system message content if required
    #     # sys_content = ""
    #     # if system_file is not None and os.path.exists(system_file):
    #     #     with open(system_file, 'r', encoding='utf-8') as f:
    #     #         sys_content = f.read().strip()
    #
    #     # Only use the first image for this example
    #     image = self.encode_image(image_paths[0])
    #
    #     # Construct dialogue message "<|image|><|begin_of_text|>" +
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image"},   # Indicates processor should include an image
    #                 {"type": "text", "text": prompt}
    #             ]
    #         }
    #     ]
    #
    #     # Convert the above multimodal content into model-recognizable input
    #     input_text = self.processor.apply_chat_template(
    #         messages,
    #         add_generation_prompt=True
    #     )
    #
    #     # Use processor for tokenizing; note the image is also passed
    #     inputs = self.processor(
    #         image,
    #         input_text,
    #         add_special_tokens=False,
    #         return_tensors="pt"
    #     ).to(self.model.device)
    #
    #     # Inference
    #     output = self.model.generate(**inputs, max_new_tokens=50)
    #     # Decode output
    #     decoded_output = self.processor.decode(output[0])
    #     return decoded_output
    # def icl_response(self, image_paths, context, new_question, system_file=None):
    #     """
    #     Multi-turn dialogue: Given multiple images, existing Q&A context, and a new question,
    #     implement ICL (in-context-learning) to concatenate the context for generating a new answer.
    #
    #     Parameters:
    #         image_paths: List of image paths or PIL.Image objects.
    #         context: List structured as ["Q1", "A1", "Q2", "A2", ...].
    #         new_question: User's latest question.
    #         system_file: Optional file containing a system prompt.
    #     Returns:
    #         Generated response string.
    #     """
    #     if not image_paths:
    #         raise ValueError("Please provide at least one image path or PIL.Image object.")
    #
    #     # Load system message content if provided
    #     sys_content = ""
    #     if system_file is not None and os.path.exists(system_file):
    #         with open(system_file, 'r', encoding='utf-8') as f:
    #             sys_content = f.read().strip()
    #
    #     # Build conversation structure
    #     conversation = []
    #
    #     # Add system message if applicable
    #     # if sys_content:
    #     #     conversation.append({"role": "system", "content": [{"type": "text", "text": sys_content}]})
    #
    #     # Add context Q&A pairs
    #     for i in range(0, len(context), 2):
    #         user_msg = context[i] if i < len(context) else ""
    #         assistant_msg = context[i + 1] if i + 1 < len(context) else ""
    #
    #         if user_msg:
    #             conversation.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
    #         if assistant_msg:
    #             conversation.append({"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]})
    #
    #     # Add the new question
    #     conversation.append({"role": "user", "content": [{"type": "text", "text": new_question}]})
    #
    #     # Prepare input prompt
    #     input_prompt = self.processor.apply_chat_template(
    #         conversation, return_tensors="pt"
    #     )
    #
    #     # Load and preprocess the first image
    #     image = image_paths[0]
    #     if isinstance(image, str):  # If image is a path, open it
    #         image = Image.open(image).convert("RGB")
    #
    #     # Prepare inputs for the model
    #     inputs = self.processor(text=input_prompt, images=image, return_tensors="pt").to(self.model.device)
    #
    #     # Generate response
    #     prompt_len = len(inputs['input_ids'][0])
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_new_tokens=256,
    #         pad_token_id=0,
    #         eos_token_id=self.processor.tokenizer.eos_token_id,
    #         do_sample=True,
    #         temperature=0.7,
    #         top_p=0.9
    #     )
    #
    #     # Extract the generated tokens and decode the response
    #     generated_tokens = outputs[:, prompt_len:]
    #     response = self.processor.decode(generated_tokens[0], skip_special_tokens=True)
    #
    #     return response
    def eval(self, image_paths, prompt="Describe these images", system_file=None):
        """
        Evaluate a list of images with a given prompt.

        Args:
            image_paths (list): List of file paths to images.
            prompt (str): The textual prompt to use for evaluation.
            system_file (str, optional): Path to a system configuration file, if needed.

        Returns:
            list: Decoded responses from the model for each image.
        """
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        if len(images) == 0:
            images = None
        # Build the conversation template for a single image and prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]

        input_prompt = self.processor.apply_chat_template(
            conversation, return_tensors="pt"
        )
        inputs = self.processor(
            text=input_prompt, images=images, return_tensors="pt"
        ).to(self.model.device)

        prompt_len = len(inputs['input_ids'][0])
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=0,
        )

        generated_tokens = output[:, prompt_len:]
        response= self.processor.decode(generated_tokens[0])

        return response

    def icl_response(self, image_paths, context, new_question, system_file=None):
        """
        Generate a model response for a new question in a multi-turn conversation.

        Args:
            image_paths (list): List of file paths to images.
            context (list): A list of dictionaries representing prior conversation turns.
            new_question (str): The new question posed by the user.
            system_file (str, optional): Path to a system configuration file, if needed.

        Returns:
            list: Decoded response considering the conversation context and all images.
        """
        # Load and process all images
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        if len(images) == 0:
            images = None
        # Add the new question to the context
        context.append({"role": "user", "content": [{"type": "text", "text": new_question}]})

        # Generate the input prompt for the processor
        input_prompt = self.processor.apply_chat_template(
            context, return_tensors="pt"
        )

        # Prepare the inputs by passing both text and all images to the processor
        inputs = self.processor(
            text=input_prompt, images=images, return_tensors="pt"
        ).to(self.model.device)

        # Generate model outputs
        prompt_len = len(inputs['input_ids'][0])
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=0,
        )

        # Decode the generated response tokens
        generated_tokens = output[:, prompt_len:]
        response = self.processor.decode(generated_tokens[0])

        return response

    # def icl_response(self, image_paths, context, new_question, system_file=None):
    #     """
    #     Multi-turn dialogue: Given multiple images, existing Q&A context, and a new question,
    #     implement ICL (in-context-learning) to concatenate the context for generating a new answer.
    #
    #     context is a list, structured as ["Q1", "A1", "Q2", "A2", ...],
    #     where even indices are questions, and odd indices are answers.
    #     new_question: User's latest question.
    #     """
    #     if not image_paths:
    #         raise ValueError("Please provide at least one image path or PIL.Image object.")
    #
    #     # Load system message content if required
    #     sys_content = ""
    #     if system_file is not None and os.path.exists(system_file):
    #         with open(system_file, 'r', encoding='utf-8') as f:
    #             sys_content = f.read().strip()
    #
    #     # Construct multi-turn dialogue messages
    #     messages = []
    #
    #     # Add system message if applicable
    #     if sys_content:
    #         messages.append({"role": "system", "content": sys_content})
    #
    #     # Process each Qn/An pair
    #     for i in range(0, len(context), 2):
    #         user_msg = context[i] if i < len(context) else ""
    #         assistant_msg = context[i + 1] if i + 1 < len(context) else ""
    #
    #         if user_msg:
    #             messages.append({"role": "user", "content": user_msg})
    #         if assistant_msg:
    #             messages.append({"role": "assistant", "content": assistant_msg})
    #
    #     # Add the user's new question
    #     messages.append({"role": "user", "content": new_question})
    #
    #     # Apply chat template to construct input text
    #     input_text = self.processor.apply_chat_template(
    #         messages, add_generation_prompt=True, return_tensors="pt"
    #     ).to(self.model.device)
    #
    #     # Encode the first image for inference
    #     image = self.encode_image(image_paths[0])
    #
    #     # Prepare inputs for the model
    #     inputs = self.processor(
    #         image,
    #         input_text,
    #         add_special_tokens=False,
    #         return_tensors="pt"
    #     ).to(self.model.device)
    #
    #     # Generate response from the model
    #     outputs = self.model.generate(
    #         **inputs,
    #         max_new_tokens=80,
    #         eos_token_id=self.processor.tokenizer.eos_token_id,
    #         do_sample=True,
    #         temperature=0.7,
    #         top_p=0.9
    #     )
    #
    #     # Decode the model's output
    #     response = self.processor.decode(outputs[0], skip_special_tokens=True)
    #
    #     return response

    # def icl_response(self, image_paths, context, new_question, system_file=None):
    #     """
    #     Multi-turn dialogue: Given multiple images, existing Q&A context, and a new question,
    #     implement ICL (in-context-learning) to concatenate the context for generating a new answer.
    #
    #     context is a list, structured as ["Q1", "A1", "Q2", "A2", ...],
    #     where even indices are questions, and odd indices are answers.
    #     new_question: User's latest question.
    #     """
    #     if not image_paths:
    #         raise ValueError("Please provide at least one image path or PIL.Image object.")
    #
    #     # Load system message content if required
    #     # sys_content = ""
    #     # if system_file is not None and os.path.exists(system_file):
    #     #     with open(system_file, 'r', encoding='utf-8') as f:
    #     #         sys_content = f.read().strip()
    #
    #     # Construct multi-turn dialogue messages
    #     msgs = []
    #     # Process the first Q1/A1 pair with the image
    #     user_content = [{"type": "image"}]  # Mark the first round with an image
    #     user_content.append({"type": "text", "text": context[0]})  # Q1
    #     msgs.append({"role": "user", "content": user_content})
    #     msgs.append({
    #         "role": "assistant",
    #         "content": [{"type": "text", "text": context[1]}]  # A1
    #     })
    #
    #     # Process subsequent Qn/An pairs (if any)
    #     i = 2
    #     while i < len(context):
    #         user_msg = [{"type": "text", "text": context[i]}]       # Qn
    #         assistant_msg = [{"type": "text", "text": context[i+1]}]  # An
    #         msgs.append({"role": "user", "content": user_msg})
    #         msgs.append({"role": "assistant", "content": assistant_msg})
    #         i += 2
    #
    #     # Add the user's new question
    #     msgs.append({"role": "user", "content": [{"type": "text", "text": new_question}]})
    #
    #     # Construct input text
    #     input_text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
    #
    #     # As an example, only the first image is used for inference
    #     image = self.encode_image(image_paths[0])
    #     inputs = self.processor(
    #         image,
    #         input_text,
    #         add_special_tokens=False,
    #         return_tensors="pt"
    #     ).to(self.model.device)
    #
    #     # Inference
    #     output = self.model.generate(**inputs, max_new_tokens=80)
    #     decoded_output = self.processor.decode(output[0])
    #     return decoded_output


# # --------------------- Usage Example ---------------------
# if __name__ == "__main__":
#     # Initialize
#     assistant = Llama3VisionAssistant(
#         model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )
#
#     # Example image - replace with a local file path or URL
#     image_paths = [
#         "https://huggingface.co/datasets/huggingface/documentation-images/"
#         "resolve/main/diffusers/rabbit.jpg"
#     ]
#     # 1) eval - Single-turn dialogue
#     prompt = "If I had to write a haiku for this picture, what would it be?"
#     single_round_response = assistant.eval(image_paths, prompt)
#     print("Eval Response:", single_round_response)
#
#     # 2) icl_response - Multi-turn dialogue with context
#     #   context is ["Q1", "A1", "Q2", "A2", ...]
#     #   Q1 - first question, A1 - first answer, Q2 - second question, A2 - second answer, and so on
#     context = [
#         "What animal is in the picture?", "It appears to be a rabbit.",
#         "Where is this rabbit?", "It is on a patch of grass in a garden."
#     ]
#     new_question = "Write a short poem for this rabbit."
#     multi_round_response = assistant.icl_response(
#         image_paths,
#         context,
#         new_question
#     )
#     print("ICL Response:", multi_round_response)
