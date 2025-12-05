### 环境初始化
uv sync

### 配置llm
根目录创建.env文件，根据src/top5_exclude/LLM.py中模型选择配置服务密钥
如:

siliconflow_api_key = "sk-xxxx"

ali_api_key = "sk-xxxx"

open_router_api_key = "sk-or-v1-xxxxx"

硅基流动：https://cloud.siliconflow.cn/me/account/ak

阿里云百炼：https://bailian.console.aliyun.com/?tab=model#/api-key


### 运行
放置文件到data目录，修改sku_top5.py中文件路径，
执行src/top5_exclude/sku_top5.py