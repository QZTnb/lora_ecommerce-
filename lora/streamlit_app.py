import streamlit as st
from src.inference.inference import Inference
import time

# 页面配置
st.set_page_config(
    page_title="电商评论摘要生成器",
    page_icon="📝",
    layout="wide"
)

# 标题和介绍
st.title("📝 电商评论摘要生成器")
st.markdown("使用 LoRA 微调的 Qwen 模型生成多语言电商评论摘要")

# 侧边栏
with st.sidebar:
    st.header("模型配置")
    model_path = st.text_input("模型路径", value="src/models/lora_model")
    
    # 加载模型按钮
    if st.button("加载模型"):
        with st.spinner("加载模型中..."):
            try:
                st.session_state.inference = Inference(model_path)
                st.success("模型加载成功！")
            except Exception as e:
                st.error(f"模型加载失败: {e}")

# 主界面
st.header("评论输入")

# 多语言示例
with st.expander("查看示例评论"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("中文示例")
        st.code("这个产品非常好，质量上乘，价格合理，物流也很快。我非常满意，会推荐给朋友。")
    
    with col2:
        st.subheader("英文示例")
        st.code("The product is excellent, high quality, reasonable price, and fast shipping. I am very satisfied and will recommend it to friends.")
    
    with col3:
        st.subheader("日文示例")
        st.code("この製品は非常に良く、品質が優れており、価格も合理的で、配送も速いです。非常に満足しています。")

# 评论输入
review = st.text_area(
    "请输入评论内容",
    height=150,
    placeholder="在此输入您的评论..."
)

# 生成摘要按钮
if st.button("生成摘要"):
    if not review.strip():
        st.error("请输入有效的评论内容")
    elif "inference" not in st.session_state:
        st.error("请先加载模型")
    else:
        with st.spinner("生成摘要中..."):
            start_time = time.time()
            try:
                summary = st.session_state.inference.generate_summary(review)
                end_time = time.time()
                
                # 显示结果
                st.success("摘要生成成功！")
                st.subheader("生成的摘要")
                st.info(summary)
                st.caption(f"生成时间: {end_time - start_time:.2f} 秒")
            except Exception as e:
                st.error(f"生成摘要时出错: {e}")

# 历史记录
if "history" not in st.session_state:
    st.session_state.history = []

# 显示历史记录
if st.session_state.history:
    st.header("历史记录")
    for i, (review_text, summary_text, timestamp) in enumerate(reversed(st.session_state.history)):
        with st.expander(f"记录 {i+1} - {timestamp}"):
            st.markdown(f"**评论:** {review_text}")
            st.markdown(f"**摘要:** {summary_text}")

# 添加到历史记录的函数
def add_to_history(review, summary):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append((review, summary, timestamp))
    # 只保留最近10条记录
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)
