import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 通过 messages 数组实现上下文管理
messages = [
    {'role': 'user', 'content': '你是幻方量化公司私募的量化团队负责人，同时也是一位具有丰富经验的行业顶级量化师。你的背景还有整个幻方量化多年积累的模型和训练数据资源作为支撑，可以随时提供你使用。公司拟将1000万人民币的资金要投入A股市场，现在需要你设计交易策略和支撑算法。'}
]

completion = client.chat.completions.create(
    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=messages
)

print("="*20+"第一轮对话"+"="*20)
# 通过reasoning_content字段打印思考过程
print("="*20+"思考过程"+"="*20)
print(completion.choices[0].message.reasoning_content)
# 通过content字段打印最终答案
print("="*20+"最终答案"+"="*20)
print(completion.choices[0].message.content)
