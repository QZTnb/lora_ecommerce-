import os
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # 替换Seq2Seq
from peft import PeftModel
from src.config.config import Config


class Evaluator:
    def __init__(self, model_path):
        self.config = Config()
        self.model_path = model_path
        # 加载Qwen Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        # 加载Qwen基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cpu"  # 强制使用CPU，避免MPS设备问题
        )
        # 加载Lora权重
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.eval()  # 评估模式

        # 初始化评估指标
        self.bleu = evaluate.load("bleu")
        # 跳过ROUGE评估，避免安装额外依赖
        self.rouge = None

        # 补全pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_summary(self, review):
        """生成摘要（适配Qwen Chat格式）"""
        # 构建Qwen Chat提示词
        prompt = f"<|im_start|>user\n请为以下电商评论生成简洁的摘要：{review}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            prompt,
            max_length=self.config.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # 使用CPU，避免MPS设备问题
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        # 生成摘要（Decoder-only生成逻辑）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.SUMMARY_MAX_LENGTH,  # 只生成新token（摘要）
                num_beams=4,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 解码并提取摘要（去掉提示词部分）
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除提示词部分，只保留摘要
        if "assistant\n" in full_text:
            summary = full_text.split("assistant\n")[-1].strip()
        else:
            summary = full_text.strip()
        # 移除可能的前缀
        summary = summary.replace("评价内容：", "").strip()
        return summary

    def evaluate(self, test_dataset):
        """评估模型（逻辑不变）"""
        predictions = []
        references = []

        for example in test_dataset:
            review = example["review"]
            reference = example["summary"]
            prediction = self.generate_summary(review)

            predictions.append(prediction)
            references.append(reference)

        bleu_score = self.bleu.compute(predictions=predictions, references=references)
        
        # 只计算BLEU分数，跳过ROUGE评估
        rouge_score = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

        return {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "predictions": predictions,
            "references": references
        }