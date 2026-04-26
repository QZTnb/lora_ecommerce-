import os
from src.train.train import LoraTrainer
from src.evaluate.evaluate import Evaluator
from src.inference.inference import Inference

# 示例数据路径
data_path = "data/sample_reviews.csv"

# 训练模型
def train_model():
    print("开始训练模型...")
    trainer = LoraTrainer(data_path)
    model_path = trainer.train()
    print(f"模型训练完成，保存路径: {model_path}")
    return model_path

# 评估模型
def evaluate_model(model_path):
    print("开始评估模型...")
    from src.data.data_processor import DataProcessor
    data_processor = DataProcessor()
    dataset = data_processor.process(data_path)
    test_dataset = dataset['test']
    
    evaluator = Evaluator(model_path)
    results = evaluator.evaluate(test_dataset)
    print("评估结果:")
    print(f"BLEU分数: {results['bleu']['bleu']}")
    print(f"ROUGE-1: {results['rouge']['rouge1']}")
    print(f"ROUGE-2: {results['rouge']['rouge2']}")
    print(f"ROUGE-L: {results['rouge']['rougeL']}")
    return results

# 推理示例
def inference_example(model_path):
    print("开始推理示例...")
    inference = Inference(model_path)
    
    # 示例评论
    sample_reviews = [
        "这个产品非常好，质量上乘，价格合理，物流也很快。我非常满意，会推荐给朋友。",
        "The product is excellent, high quality, reasonable price, and fast shipping. I am very satisfied and will recommend it to friends.",
        "この製品は非常に良く、品質が優れており、価格も合理的で、配送も速いです。非常に満足しています。"
    ]
    
    for i, review in enumerate(sample_reviews):
        summary = inference.generate_summary(review)
        print(f"\n评论 {i+1}:")
        print(f"原始评论: {review}")
        print(f"生成摘要: {summary}")

if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs("data", exist_ok=True)
    
    # 训练模型
    model_path = train_model()
    
    # 评估模型
    evaluate_model(model_path)
    
    # 推理示例
    inference_example(model_path)
