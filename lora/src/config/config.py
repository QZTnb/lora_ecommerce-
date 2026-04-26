import os
import torch


class Config:
    # 模型配置 - 替换为Qwen模型
    MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"  # 核心修改
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1

    # 训练配置
    BATCH_SIZE = 8  # 可根据Qwen-0.5B显存占用调整（建议先试4/8）
    EPOCHS = 3
    LEARNING_RATE = 1e-4  # Qwen建议学习率略高（原5e-5→1e-4）
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100

    # 数据配置 - Qwen的Tokenizer对长度更敏感，建议调整
    MAX_LENGTH = 1024  # Qwen1.5-0.5B支持更长输入（原512→1024）
    SUMMARY_MAX_LENGTH = 200  # 摘要长度适配（原150→200）

    # 路径配置（不变）
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

    # 其他配置
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SUPPORTED_LANGUAGES = ["zh", "en", "ja"]