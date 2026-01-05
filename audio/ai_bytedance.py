import os
from openai import OpenAI

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Openai客户端，从环境变量中读取您的API Key
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
    # 从环境变量中获取您的 API Key
    api_key=os.environ.get("ARK_API_KEY")
)

completion = client.chat.completions.create(
    model="bot-20250218193443-c7vhp",  # bot-20250212213506-gpp89 为您当前的智能体的ID，注意此处与Chat API存在差异。差异对比详见 SDK使用指南
    messages=[  # 通过会话传递历史信息，模型会参考上下文消息
        {"role": "user", "content": "设计一个北海道旅游的攻略，包括景点、美食、住宿、交通等"},
        # {"role": "assistant", "content": "花椰菜又称菜花、花菜，是一种常见的蔬菜。"},
        # {"role": "user", "content": "再详细点"},
    ],
    stream=True
)

# for chunk in completion:
    # if chunk.references:
    #     print(chunk.references)
    # if not chunk.choices:
    #     continue
    # if chunk.choices[0].delta.reasoning_content:
    #     print(chunk.choices[0].delta.reasoning_content, end="")  # 对于R1模型，输出reasoning content
    # elif chunk.choices[0].delta.content:
    #     print(chunk.choices[0].delta.content, end="")


print(completion.choices[0].message.content)

# if hasattr(completion, "references"):
#     print("<references>")
#     print(completion.references)
#     print("</references>")
# if hasattr(completion.choices[0].message, "reasoning_content"):
#     print("<think>")
#     print(completion.choices[0].message.reasoning_content)  # 对于R1模型，输出reasoning content
#     print("</think>")
