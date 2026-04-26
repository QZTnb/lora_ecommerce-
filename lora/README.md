# Lora微调LLM多语言电商评论摘要系统

## 项目简介

本项目利用 Lora（Low-Rank Adaptation）技术对 Qwen 大模型进行参数高效微调，实现对中、英、日三种语言的电商评论自动分析与摘要生成。

## 技术栈

- **Python** | **PyTorch** | **Transformers** | **PEFT** | **Lora** | **Hugging Face** | **BLEU评估**

## 核心功能

1. **参数高效微调**：基于 LoRA 技术，训练参数量降低 90%，显著减少算力与显存消耗
2. **多语言支持**：设计中/英/日三语统一处理架构，实现多语言电商评论自动分析与摘要生成
3. **端到端训练 Pipeline**：搭建数据处理、模型训练、自动评估、推理优化的完整流程
4. **量化评估**：集成 BLEU 指标量化效果，支持模型性能评估
5. **部署优化**：支持 CPU 推理与批量处理，可快速保存部署

## 项目结构

```
├── data/                  # 数据目录
│   └── sample_reviews.csv # 示例数据
├── src/                   # 源代码目录
│   ├── config/            # 配置模块
│   │   └── config.py      # 配置文件
│   ├── data/              # 数据处理模块
│   │   └── data_processor.py # 数据处理器
│   ├── models/            # 模型模块
│   │   └── model.py       # 模型定义
│   ├── train/             # 训练模块
│   │   └── train.py       # 训练脚本
│   ├── evaluate/          # 评估模块
│   │   └── evaluate.py    # 评估脚本
│   └── inference/         # 推理模块
│       └── inference.py   # 推理脚本
├── main.py                # 主脚本
├── deploy.py              # 部署脚本
├── app.py                 # 命令行应用
├── streamlit_app.py       # Web应用
├── requirements.txt       # 依赖文件
└── README.md              # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 2.1.0+
- Transformers 4.37.0+
- PEFT 0.7.0+
- 其他依赖见 requirements.txt

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python main.py
```

### 2. 评估模型

训练完成后，脚本会自动评估模型性能，输出 BLEU 分数。

### 3. 推理应用

#### 命令行应用

```bash
python app.py
```

#### Web 应用

```bash
streamlit run streamlit_app.py
```

### 4. 模型部署

```bash
python deploy.py
```

## 模型配置

可在 `src/config/config.py` 中修改以下配置：

- `MODEL_NAME`：预训练模型名称（默认：Qwen/Qwen1.5-0.5B-Chat）
- `LORA_R`：Lora 秩（默认：8）
- `LORA_ALPHA`：Lora alpha 参数（默认：16）
- `BATCH_SIZE`：批量大小（默认：8）
- `EPOCHS`：训练轮数（默认：3）
- `LEARNING_RATE`：学习率（默认：1e-4）
- `MAX_LENGTH`：输入最大长度（默认：1024）
- `SUMMARY_MAX_LENGTH`：摘要最大长度（默认：200）
- `SUPPORTED_LANGUAGES`：支持的语言列表（默认：["zh", "en", "ja"]）

## 数据格式

训练数据需要包含两列：
- `review`：原始评论内容
- `summary`：对应的摘要内容

## 示例数据

项目提供了 `data/sample_reviews.csv` 示例数据，包含中、英、日三种语言的评论和摘要。

## 性能评估

模型训练完成后，会自动计算以下指标：
- **BLEU**：评估生成摘要与参考摘要的相似度

## 部署说明

1. **CPU 推理**：模型支持在 CPU 环境下运行，适合部署到资源有限的环境
2. **批量处理**：支持批量生成摘要，提高处理效率
3. **模型保存**：训练完成后会自动保存模型到 `src/models/lora_model` 目录

## 注意事项

1. 首次运行时会自动下载预训练模型，可能需要较长时间
2. 训练过程中需要一定的 GPU 资源，建议在 GPU 环境下运行
3. 对于大型数据集，可能需要调整批量大小和其他训练参数
4. 支持的语言：中文、英文、日文

## 示例输出

| 语言 | 原始评论 | 生成摘要 |
|------|---------|---------|
| 中文 | 这个产品非常好，质量上乘，价格合理，物流也很快。我非常满意，会推荐给朋友。 | 产品非常好，质量上乘，价格合理，物流也很快。非常满意，会推荐给朋友。 |
| 英文 | The product is excellent, high quality, reasonable price, and fast shipping. I am very satisfied and will recommend it to friends. | 商品优秀，质量高，价格合理，物流快，推荐给朋友。 |
| 日文 | この製品は非常に良く、品質が優れており、価格も合理的で、配送も速いです。非常に満足しています。 | 商品品质优秀，价格合理，配送速度快，非常满意。 |
