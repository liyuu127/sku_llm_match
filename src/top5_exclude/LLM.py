from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = current_dir + "/../../.env"
# 加载 .env 文件
load_dotenv(env_path)

qwen3_8B = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("siliconflow_api_key"),

)

GLM4_9B = ChatOpenAI(
    model="THUDM/GLM-4-9B-0414",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("siliconflow_api_key"),

)
glm_z1_9b = ChatOpenAI(
    model="THUDM/GLM-Z1-9B-0414",
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("siliconflow_api_key"),

)

longcat_flash_chat = ChatOpenAI(
    model="meituan/longcat-flash-chat:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("open_router_api_key"),
)

glm4_5_air = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("open_router_api_key"),
)

qwen3_14b = ChatOpenAI(
    model="qwen/qwen3-14b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("open_router_api_key"),
)
qwen3_next_80b = ChatOpenAI(
    model="qwen3-next-80b-a3b-instruct",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ali_api_key"),
)
qwen3_30b_instruct = ChatOpenAI(
    model="qwen3-30b-a3b-instruct-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ali_api_key"),
)
qwen3_235b_thinking = ChatOpenAI(
    model="qwen3-235b-a22b-thinking-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ali_api_key"),
)
qwen3_235b_instruct = ChatOpenAI(
    model="qwen3-235b-a22b-instruct-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ali_api_key"),
)
qwen3_max = ChatOpenAI(
    model="qwen3-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ali_api_key"),
)
qwen3_30b_thinking = ChatOpenAI(
    model="qwen3-30b-a3b-thinking-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ali_api_key"),
)
