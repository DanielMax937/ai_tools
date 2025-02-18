import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
    api_key=os.environ.get("ARK_API_KEY")
)


def format_assistant_response(response_chunk, ai):

    """格式化助手的响应块，包括思考过程和引用"""
    content = ""
    
    # 处理主要内容
    if hasattr(response_chunk.choices[0].delta, 'content'):
        res = response_chunk.choices[0].delta.content
        if ai["status"] == "think" and res and res.strip():
            content += "=" * 15 + "结束思考" + "=" * 15 + "\n\n"
            ai["status"] = "content"
        content += response_chunk.choices[0].delta.content
        
    # 处理思考过程
    if hasattr(response_chunk.choices[0].delta, 'reasoning_content'):
        if ai["status"] == "initial":
            content += "=" * 15 + "思考过程" + "=" * 15 + "\n"
            ai["status"] = "think"
        content += response_chunk.choices[0].delta.reasoning_content
            
    # 处理引用
    if hasattr(response_chunk, 'references'):
        ref = response_chunk.references
        print("===========")
        for item in ref:
            print(item)
        print("===========")
        # if ai["status"] == "content" and ref and ref.strip():
        #     content += "\n参考资料：\n"
        #     ai["status"] = "finish"
        # content += ("\n").join(ref)
            
    return content

def chat(message, history):
    """
    处理聊天消息并返回响应（流式输出）
    
    Args:
        message (str): 用户输入的消息
        history (list): 聊天历史记录
        
    Returns:
        generator: 生成器，用于流式输出响应
    """
    # 构建消息历史
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    # 添加当前消息
    messages.append({"role": "user", "content": message})
    
    try:
        # 调用 API（启用流式输出）
        response = client.chat.completions.create(
            model="bot-20250218193443-c7vhp",
            messages=messages,
            stream=True  # 启用流式输出
        )
        
        # 流式处理响应
        partial_message = ""
        has_reasoning = False
        has_references = False
        ai = {"status": "initial"}
        for chunk in response:
            if chunk:
                content = format_assistant_response(chunk, ai)
                if content:
                    partial_message += content
                    yield partial_message
        ai = {"status": "initial"}
        
    except Exception as e:
        yield f"抱歉，发生了错误：{str(e)}"

# 创建 Gradio 界面
demo = gr.ChatInterface(
    fn=chat,
    title="AI 助手",
    # description="这是一个基于字节跳动 AI API 的聊天助手，支持流式输出。你可以询问任何问题，我会实时回答。",
    examples=[
        ["你好，请介绍一下你自己"],
        ["帮我写一个 Python 函数来计算斐波那契数列"],
        ["解释一下什么是机器学习"],
    ],
    # theme=gr.themes.Soft(),
)

# 启动应用
if __name__ == "__main__":
    demo.launch(
        share=True, 
        server_name="0.0.0.0",
        show_api=False  # 隐藏 API 文档按钮
    ) 