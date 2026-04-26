import os
import torch
from transformers import TrainingArguments, Trainer as TransformersTrainer  # 替换Seq2Seq为基础TrainingArguments
from src.config.config import Config
from src.data.data_processor import DataProcessor
from src.models.model import LoraModel


class LoraTrainer:
    def __init__(self, data_path):
        self.config = Config()
        self.data_processor = DataProcessor()
        self.lora_model = LoraModel()
        self.dataset = self.data_processor.process(data_path)
        self.tokenized_dataset = self.dataset.map(
            self.lora_model.tokenize_function,
            batched=True,
            remove_columns=self.dataset['train'].column_names
        )
        self.model = self.lora_model.get_peft_model()

    def get_training_args(self):
        """获取适配Qwen的训练参数"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)

        training_args = TrainingArguments(  # 替换Seq2SeqTrainingArguments
            output_dir=self.config.MODEL_DIR,
            evaluation_strategy="epoch",
            learning_rate=self.config.LEARNING_RATE,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            weight_decay=self.config.WEIGHT_DECAY,
            save_total_limit=3,
            num_train_epochs=self.config.EPOCHS,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_dir=os.path.join(self.config.OUTPUT_DIR, "logs"),
            logging_steps=100,
            seed=self.config.SEED,
            # Qwen专属配置
            fp16=torch.cuda.is_available(),  # 混合精度训练（GPU建议开启）
            gradient_checkpointing=False,  # 关闭梯度检查点，避免梯度问题
        )
        return training_args

    def train(self):
        """训练模型"""
        training_args = self.get_training_args()

        trainer = TransformersTrainer(  # 替换Seq2SeqTrainer
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation']
        )

        trainer.train()

        # 保存模型
        model_save_path = os.path.join(self.config.MODEL_DIR, "lora_model")
        self.model.save_pretrained(model_save_path)
        self.lora_model.tokenizer.save_pretrained(model_save_path)

        return model_save_path