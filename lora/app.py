#!/usr/bin/env python3
"""
应用脚本：使用已训练的 LoRA 模型生成电商评论摘要
"""
import sys
from src.inference.inference import Inference

def main():
    # 模型路径
    model_path = "src/models/lora_model"
    
    print("加载模型...")
    try:
        inference = Inference(model_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    print("\n=== 电商评论摘要生成器 ===")
    print("输入 'exit' 退出程序")
    print("\n请输入评论内容：")
    
    while True:
        try:
            review = input("评论: ")
            
            if review.lower() == 'exit':
                print("退出程序...")
                break
            
            if not review.strip():
                print("请输入有效的评论内容")
                continue
            
            print("生成摘要中...")
            summary = inference.generate_summary(review)
            print(f"生成的摘要: {summary}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n退出程序...")
            break
        except Exception as e:
            print(f"生成摘要时出错: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()