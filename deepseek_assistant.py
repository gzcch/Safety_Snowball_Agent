# filename: deepseek_vl2_assistant.py

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class DeepSeekVL2Assistant:
    """
    基于 deepseek-vl2-small 模型，提供 eval 和 icl_response 两个主要方法，
    用于多模态问答和多轮对话（ICL）。
    """

    def __init__(
        self,
        model_path="deepseek-ai/deepseek-vl2-small",
        device="cuda",
        torch_dtype=torch.bfloat16,
    ):
        """
        初始化 DeepSeek-VL2 模型、处理器等。

        参数：
            model_path (str): 模型的名称或本地路径。
            device (str): 设备，"cuda" 或 "cpu"。
            torch_dtype (torch.dtype): 模型使用的精度，比如 torch.bfloat16、torch.float16 等。
        """

        # 1. 加载处理器
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            model_path
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        # 2. 加载模型
        #   注意：可以通过 from_pretrained 直接加载 DeepseekVLV2ForCausalLM，
        #   或者和你提供的例子一样先加载AutoModelForCausalLM再转换。
        #   这里遵循你的示例用 AutoModelForCausalLM + trust_remote_code=True 即可。
        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 3. 设置精度/设备，并 eval
        self.vl_gpt = self.vl_gpt.to(dtype=torch_dtype, device=device).eval()
        self.device = device

    def eval(self, image_paths, question, max_new_tokens=512, do_sample=False):
        """
        单轮多模态问答（类似 QwenVL2Assistant 里的 eval）。
        传入图片列表和问题，输出模型生成的回答。
        """

        # 1. 构建单轮对话的 conversation 格式
        #    注意对 DeepSeek-VL2，需要 role 写成 <|User|> / <|Assistant|>，并插入 "<image>" 占位符
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": image_paths,  # 这里是一组图
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 2. 加载 PIL Image
        pil_images = load_pil_images(conversation)

        # 3. 调用处理器，得到输入
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",  # 这里可自行指定
        ).to(self.vl_gpt.device)

        # 4. 获得图像 embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # 5. 调用语言模型生成
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )

        # 6. 解码并返回
        answer_ids = outputs[0][prepare_inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(answer_ids.cpu().tolist(), skip_special_tokens=True)
        return answer

    def icl_response(self, image_paths, context, current_question, max_new_tokens=512, do_sample=False):
        """
        多轮对话 ICL (In-Context Learning) 场景下的多模态问答。
        context 为已有对话列表： [user_input1, assistant_answer1, user_input2, assistant_answer2, ...]
        current_question 为用户最新提问（带图像）。
        """

        # 1. 先把已有 context 拼成 conversation
        #    这里假设 context: [user_text, assistant_text, user_text, assistant_text, ...]
        #    DeepSeek-VL2 需要 role = <|User|>, <|Assistant|> 格式
        conversation = []
        for i in range(0, len(context), 2):
            user_text = context[i]
            assistant_text = ""
            if i + 1 < len(context):
                assistant_text = context[i + 1]

            conversation.append({"role": "<|User|>", "content": user_text, "images": []})
            conversation.append({"role": "<|Assistant|>", "content": assistant_text})

        # 2. 最后一轮用户消息 + 图像
        conversation.append(
            {
                "role": "<|User|>",
                "content": f"<image>\n{current_question}",
                "images": image_paths,
            }
        )
        # 3. 最后一轮 assistant 留空
        conversation.append({"role": "<|Assistant|>", "content": ""})

        # 4. 加载 PIL Image
        pil_images = load_pil_images(conversation)

        # 5. 调用处理器
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",  # 这里可自行指定
        ).to(self.vl_gpt.device)

        # 6. 获得图像 embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # 7. 生成
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )

        # 8. 解码并返回
        answer_ids = outputs[0][prepare_inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(answer_ids.cpu().tolist(), skip_special_tokens=True)
        return answer
