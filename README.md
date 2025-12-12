### 1.环境初始化
环境基础要求：python3+uv

进入到仓库根目录执行：

```bash
uv sync
```

### 2. 配置llm
用于第二阶段商品排除和相似商品匹配，推荐使用30B及以上模型，平台可选择硅基流动或阿里云百炼，硅基流动提供GLM-4-9B-0414、Qwen3-8B免费模型，阿里云百炼提供每个模型100w免费tokens额度。

下面是平台api-key获取地址：

1. 硅基流动：https://cloud.siliconflow.cn/me/account/ak

2. 阿里云百炼：https://bailian.console.aliyun.com/?tab=model#/api-key



获取api_key后，在根目录创建.env文件（可参考.env-example）如:

```properties
siliconflow_api_key = "sk-xxxx"

ali_api_key = "sk-xxxx"

open_router_api_key = "sk-or-v1-xxxxx"
```

api-key配置完成后，可在src/top5_exclude/LLM.py中配置云平台中使用的模型，并在代码中进行引用。

```python
qwen3_30b_instruct = ChatOpenAI(
    model="qwen3-30b-a3b-instruct-2507",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ali_api_key"),
)

model = qwen3_30b_instruct.with_structured_output(ExcludeSelectNoReason).with_retry(stop_after_attempt=2)

```



### 3.运行

#### 3.1.输入与输出
程序读取目录位于data目录下，程序执行前需先上传原表和匹配目标表到目录下。文件要求格式为xlsx,列名包含商品ID，商品名称，规格，折扣价，原价，销售，图片字段。

如文件上传：

./data/top3相似人工标注数据_需要大模型识别.xlsx

./data/附件2-美团邻侣全量去重商品1109.xlsx

程序输出目录位于output目录下，输出文件为包含图片的多列excel文件，文件名称由前缀、模型名称、提示此版本、uuid后缀组成，具体可参考执行入口文件。

输出文件： ./output/top5相似_glm4-9b_prompt_v2_500_6bc31bc4-d531-4545-a084-b2e169ef5791.xlsx

#### 3.2.脚本执行
进入到src/top5_exclude/sku_top5.py文件，点击main函数执行，或者直接python sku_top5.py执行。执行前需要根据读取文件名修改owner_df_path(自有商品文件路径)和target_df_path（对标商品文件路径），并根据模型信息可同步修改输出文件名称。初次执行可添加iloc[:10]使用前10列查看是否有异常。

```bash
cd src/top5_exclude
python sku_top5.py
```



#### 3.3.模型切换
1. 编辑src/top5_exclude/LLM.py文件，添加或修改模型信息。
2. 编辑src/top5_exclude/llm_exclude_top5.py文件，修改model导入和定义信息。
3. 编辑src/top5_exclude/llm_match_top5.py文件，修改model导入和定义信息。
````python
# model = qwen3_8B.with_structured_output(ExcludeSelect).with_retry(stop_after_attempt=2)
# model = qwen3_30b_instruct.with_structured_output(ExcludeSelect).with_retry(stop_after_attempt=2)
# model = GLM4_9B.with_structured_output(ExcludeSelect).with_retry(stop_after_attempt=2)
model = GLM4_9B.with_structured_output(ExcludeSelectNoReason).with_retry(stop_after_attempt=2)
# model = qwen3_30b_instruct.with_structured_output(ExcludeSelectNoReason).with_retry(stop_after_attempt=2)
```

