#!/usr/bin/env python3
"""
模型部署脚本
用于快速部署和使用训练好的 LoRA 微调模型
"""
import os
import argparse
from src.inference.inference import Inference

def main():
    parser = argparse.ArgumentParser(description="部署 LoRA 微调的电商评论摘要模型")
    parser.add_argument('--model_path', type=str, default='src/models/lora_model',
                        help='模型路径')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量处理大小')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 {args.model_path} 不存在")
        print("请先运行训练脚本: python main.py")
        return
    
    print(f"加载模型: {args.model_path}")
    try:
        inference = Inference(args.model_path)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    print("\n=== 模型部署成功 ===")
    print("模型已准备就绪，可以开始生成摘要")
    print("\n使用示例:")
    print("1. 单个评论摘要:")
    print("   summary = inference.generate_summary('这个产品非常好，质量上乘，价格合理')")
    print("\n2. 批量评论摘要:")
    print("   reviews = ['评论1', '评论2', '评论3']")
    print("   summaries = inference.batch_generate(reviews)")
    
    # 测试示例
    print("\n=== 测试示例 ===")
    test_reviews = [
        "这个产品非常好，质量上乘，价格合理，物流也很快。我非常满意，会推荐给朋友。",
        "The product is excellent, high quality, reasonable price, and fast shipping. I am very satisfied and will recommend it to friends.",
        "この製品は非常に良く、品質が優れており、価格も合理的で、配送も速いです。非常に満足しています。"
    ]
    
    print("测试多语言摘要生成:")
    for i, review in enumerate(test_reviews):
        summary = inference.generate_summary(review)
        print(f"\n评论 {i+1}:")
        print(f"原始: {review}")
        print(f"摘要: {summary}")
    
    if args.output_file:
        print(f"\n保存测试结果到: {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, review in enumerate(test_reviews):
                summary = inference.generate_summary(review)
                f.write(f"评论 {i+1}:\n")
                f.write(f"原始: {review}\n")
                f.write(f"摘要: {summary}\n\n")
        print("保存成功!")

if __name__ == "__main__":
    main()