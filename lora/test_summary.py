#!/usr/bin/env python3
"""
测试摘要生成功能
"""
from src.inference.inference import Inference

# 模型路径
model_path = "src/models/lora_model"

print("加载模型...")
try:
    inference = Inference(model_path)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 测试评论
test_reviews = [
    "这个产品非常好，质量上乘，价格合理，物流也很快。我非常满意，会推荐给朋友。",
    "The product is excellent, high quality, reasonable price, and fast shipping. I am very satisfied and will recommend it to friends.",
    "この製品は非常に良く、品質が優れており、価格も合理的で、配送も速いです。非常に満足しています。"
]

print("\n测试摘要生成:")
print("-" * 60)

for i, review in enumerate(test_reviews):
    print(f"\n评论 {i+1}:")
    print(f"原始评论: {review}")
    summary = inference.generate_summary(review)
    print(f"生成摘要: {summary}")
    print("-" * 60)

print("测试完成！")