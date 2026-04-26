import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # 替换Seq2Seq为CausalLM
from peft import LoraConfig, get_peft_model
from src.config.config import Config


class LoraModel:
    def __init__(self):
        self.config = Config()
        # 加载Qwen的Tokenizer（添加trust_remote_code=True）
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True,
            padding_side="right"  # Qwen建议padding在右侧
        )
        # 加载Qwen模型（CausalLM）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        # Qwen tokenizer添加pad_token（如果不存在）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def get_lora_config(self):
        """获取适配Qwen的Lora配置"""
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            # Qwen的目标模块（Decoder-only架构的Attention层）
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"  # 替换SEQ_2_SEQ_LM为CAUSAL_LM
        )
        return lora_config

    def get_peft_model(self):
        """获取PEFT模型"""
        lora_config = self.get_lora_config()
        peft_model = get_peft_model(self.model, lora_config)
        return peft_model

    def tokenize_function(self, examples):
        """适配Qwen Chat的分词函数（Decoder-only格式）"""
        # Qwen Chat需要构建对话模板：用户输入评论，要求生成摘要
        inputs = []
        for review, summary in zip(examples["review"], examples["summary"]):
            # 遵循Qwen Chat的对话格式
            input_text = f"<|im_start|>user\n请为以下电商评论生成简洁的摘要：{review}<|im_end|>\n<|im_start|>assistant\n{summary}<|im_end|>"
            inputs.append(input_text)

        # 处理输入
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt" if torch.cuda.is_available() else "np"
        )

        # 处理标签
        labels = model_inputs["input_ids"].copy() if isinstance(model_inputs["input_ids"], torch.Tensor) else model_inputs["input_ids"].copy()
        
        # 标签mask（padding部分设为-100，避免计算损失）
        if isinstance(labels, torch.Tensor):
            labels[model_inputs["attention_mask"] == 0] = -100
        else:
            labels[model_inputs["attention_mask"] == 0] = -100
        
        model_inputs["labels"] = labels

        return model_inputs