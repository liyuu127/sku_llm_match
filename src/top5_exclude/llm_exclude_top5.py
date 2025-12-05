import asyncio
from typing import Tuple, Any, List

from opik import configure
from opik.integrations.langchain import OpikTracer

from top5_exclude.LLM import qwen3_8B, qwen3_30b_instruct, GLM4_9B
from top5_exclude.Limter import RateLimiter
from top5_exclude.utils import pandas_str_to_series
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import pandas as pd


# configure(api_key="dHhF5jDOpifAMNJH9UalSjR0o", workspace="yu-li")
# configure(url="http://172.16.1.170:5173", use_local=True)
# opik_tracer = OpikTracer(project_name="sku_match")


class ExcludeSelect(BaseModel):
    """Model for match top rank. Provide json constraints."""

    exclude_index: List[int] = Field(
        description="返回与origin_product描述明显不一致的商品索引列表,索引范围为1,2,3,4,5，如[1,2]，全部一致返回空列表,如[]",
    )
    exclude_reason: str = Field(
        description="返回匹配结果过程的思考原因，需要体现对输入信息的处理、比对、分析等过程。",
    )


class ExcludeSelectNoReason(BaseModel):
    """Model for match top rank. Provide json constraints."""

    exclude_index: List[int] = Field(
        description="返回与origin_product描述明显不一致的商品索引列表,索引范围为1,2,3,4,5，如[1,2]，全部一致返回空列表,如[]",
    )


exclude_prompt = """
你是一个商品信息匹配助手，请你根据输入的商品描述信息在多个候选商品中选择描述明显不为同一款商品的索引列表。

<Task>
你需要按照下面步骤进行思考处理和返回：
1. 接收 origin_product、top*_candidate 共六个商品信息描述，origin_product 为原始商品信息、top*_candidate 为候选匹配商品信息。
2. 每个商品信息包含品牌、商品名、规格数量3部分信息，分析时先提取 origin_product 输入商品的品牌、商品名、规格数量信息。
3. 提取并对比origin_product与候选商品top*_candidate的品牌、商品名、规格数量信息，判断候选商品是否明显不同，并添加到返回列表中。
4. 根据匹配规则选出与origin_product商品描述明显不是同款商品的候选商品编号列表（如[3,4]）,索引范围为1,2,3,4,5，需要特别注意，如果候选商品与origin_product都相似，返回空列表[]。
</Task>

<Information Extraction Rules>
以下为信息提取规则：
1.商品品牌：通常位于信息前部，一般使用空格分离，部分商品可能不含品牌名。如：旺仔 精选进口乳源原味小馒头 14g_袋，品牌提取为旺仔；部分品牌和商品名重叠，如 元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，品牌为元气森林；
2.商品名称：通常位于信息前部或中部，需要去除描述部分。如：元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，名称提取为电解质水；
3.商品规格数量：
    1）通常位于信息尾部，由规格和数量两部分组成。如：喜力Heineken 11.4°P啤酒 500ml3罐_包，规格为500ml3罐_包，数量为1；
    2）部分商品数量信息位于尾部且只有数字，如：乌苏 11°P乌苏啤酒 500ml_听3（新老包装随机发货），规格数量部分为500ml_听3，规格为500ml_听，数量为3；
    3）如果没有数量部分默认为1，如：旺仔 精选进口乳源原味小馒头 14g_袋，规格为14g_袋，数量为1；
</Information Extraction Rules>

<Match rules>
以下是origin_product和候选商品的明显不一致的情况：
1. 商品品牌明显不同，例如origin_product品牌为可口，candidate品牌为百事；商品品牌如果有部分字段重叠或语义相同可计算为相同。
2. 商品数量和规格任务一个不相同及为不一致，以下是规格数据匹配时处理注意事项：
    1）规格可包括L、kg、千克、盒、件、片、条等市面计量单位，评判时注意规格单位换算，如1L和1000ml为一致；
    2）个、袋、盒、瓶等单个包装规格在整体语义下视为同等规格；
    3）规格中没有体现数量，视为数量为1。
    4）比较商品双方规格数量都不存在视为一致。
3. 商品名称不相同或语义明显为不同商品时不一致。
</Match rules>

以下是输入信息：
<products_info>
origin_product：{origin_product}
top1_candidate：{top1_candidate}
top2_candidate：{top2_candidate}
top3_candidate：{top3_candidate}
top4_candidate：{top4_candidate}
top5_candidate：{top5_candidate}
</products_info>

<Examples>
示例1：
origin_product：Oishi上好佳 非油炸无反式脂肪草莓口味粟米条膨化食品 40g_袋
top1_candidate：上好佳 无反式脂肪鲜虾片膨化食品 40g_袋2
top2_candidate：上好佳 洋葱圈膨化食品 40g_袋2
top3_candidate：上好佳 洋葱圈童年回忆休闲零食40g_袋
top4_candidate：上好佳 粟米条膨化食品 40g_袋
top5_candidate：乐事 薯片青柠味休闲食品小吃零食办公室小吃膨化食品 40g_袋
分析：origin_product和top1_candidate品牌相同、商品名类似、规格数量中数量不同（1和2），所以明显不一致；origin_product和top2_candidate品牌相同、商品名和规格数量都不同，所以明显不一致；origin_product和top3_candidate品牌和规格数量相同、商品名不同，所以明显不一致；origin_product和top4_candidate品牌和规格数量相同、商品名相同或类似，所以一致；origin_product和top5_candidate规格数量相同、品牌和商品名不同，所以明显不一致；综上分析返回不一致的商品索引列表（1,2,3,5）
</Examples>

<Output Format>
请使用以下键值以有效的 JSON 格式进行响应：
"exclude_index": List[int]. 返回与origin_product描述明显不一致的商品索引列表,索引范围为1,2,3,4,5，如[1,2]，全部一致返回空列表,如[]。
"exclude_reason":str. 返回匹配结果过程的思考原因，需要体现对输入信息的处理、比对、分析等过程。
</Output Format>

<Critical Reminder>
1. 特别注意，只返回候选商品中明显和origin_product不为同款商品的索引列表。
</Critical Reminder>
"""

exclude_prompt_no_reason = """
你是一个商品信息匹配助手，请你根据输入的商品描述信息在多个候选商品中选择描述明显不为同一款商品的索引列表。

<Task>
你需要按照下面步骤进行思考处理和返回：
1. 接收 origin_product、top*_candidate 共六个商品信息描述，origin_product 为原始商品信息、top*_candidate 为候选匹配商品信息。
2. 每个商品信息包含品牌、商品名、规格数量3部分信息，分析时先提取 origin_product 输入商品的品牌、商品名、规格数量信息。
3. 提取并对比origin_product与候选商品top*_candidate的品牌、商品名、规格数量信息，判断候选商品是否明显不同，并添加到返回列表中。
4. 根据匹配规则选出与origin_product商品描述明显不是同款商品的候选商品编号列表（如[3,4]）,索引范围为1,2,3,4,5，需要特别注意，如果候选商品与origin_product都相似，返回空列表[]。
</Task>

<Information Extraction Rules>
以下为信息提取规则：
1.商品品牌：通常位于信息前部，一般使用空格分离，部分商品可能不含品牌名。如：旺仔 精选进口乳源原味小馒头 14g_袋，品牌提取为旺仔；部分品牌和商品名重叠，如 元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，品牌为元气森林；
2.商品名称：通常位于信息前部或中部，需要去除描述部分。如：元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，名称提取为电解质水；
3.商品规格数量：
    1）通常位于信息尾部，由规格和数量两部分组成。如：喜力Heineken 11.4°P啤酒 500ml3罐_包，规格为500ml3罐_包，数量为1；
    2）部分商品数量信息位于尾部且只有数字，如：乌苏 11°P乌苏啤酒 500ml_听3（新老包装随机发货），规格数量部分为500ml_听3，规格为500ml_听，数量为3；
    3）如果没有数量部分默认为1，如：旺仔 精选进口乳源原味小馒头 14g_袋，规格为14g_袋，数量为1；
</Information Extraction Rules>

<Match rules>
以下是origin_product和候选商品的明显不一致的情况：
1. 商品品牌明显不同，例如origin_product品牌为可口，candidate品牌为百事；商品品牌如果有部分字段重叠或语义相同可计算为相同。
2. 商品数量和规格任务一个不相同及为不一致，以下是规格数据匹配时处理注意事项：
    1）规格可包括L、kg、千克、盒、件、片、条等市面计量单位，评判时注意规格单位换算，如1L和1000ml为一致；
    2）个、袋、盒、瓶等单个包装规格在整体语义下视为同等规格；
    3）规格中没有体现数量，视为数量为1。
    4）比较商品双方规格数量都不存在视为一致。
3. 商品名称不相同或语义明显为不同商品时不一致。
</Match rules>

以下是输入信息：
<products_info>
origin_product：{origin_product}
top1_candidate：{top1_candidate}
top2_candidate：{top2_candidate}
top3_candidate：{top3_candidate}
top4_candidate：{top4_candidate}
top5_candidate：{top5_candidate}
</products_info>

<Examples>
示例1：
origin_product：Oishi上好佳 非油炸无反式脂肪草莓口味粟米条膨化食品 40g_袋
top1_candidate：上好佳 无反式脂肪鲜虾片膨化食品 40g_袋2
top2_candidate：上好佳 洋葱圈膨化食品 40g_袋2
top3_candidate：上好佳 洋葱圈童年回忆休闲零食40g_袋
top4_candidate：上好佳 粟米条膨化食品 40g_袋
top5_candidate：乐事 薯片青柠味休闲食品小吃零食办公室小吃膨化食品 40g_袋
分析：origin_product和top1_candidate品牌相同、商品名类似、规格数量中数量不同（1和2），所以明显不一致；origin_product和top2_candidate品牌相同、商品名和规格数量都不同，所以明显不一致；origin_product和top3_candidate品牌和规格数量相同、商品名不同，所以明显不一致；origin_product和top4_candidate品牌和规格数量相同、商品名相同或类似，所以一致；origin_product和top5_candidate规格数量相同、品牌和商品名不同，所以明显不一致；综上分析返回不一致的商品索引列表（1,2,3,5）
</Examples>

<Output Format>
请使用以下键值以有效的 JSON 格式进行响应：
"exclude_index": List[int]. 返回与origin_product描述明显不一致的商品索引列表,索引范围为1,2,3,4,5，如[1,2]，全部一致返回空列表,如[]。
</Output Format>

<Critical Reminder>
1. 特别注意，只返回候选商品中明显和origin_product不为同款商品的索引列表。
</Critical Reminder>
"""

# model = qwen3_8B.with_structured_output(ExcludeSelect).with_retry(stop_after_attempt=2)
# model = qwen3_30b_instruct.with_structured_output(ExcludeSelect).with_retry(stop_after_attempt=2)
# model = GLM4_9B.with_structured_output(ExcludeSelect).with_retry(stop_after_attempt=2)
model = GLM4_9B.with_structured_output(ExcludeSelectNoReason).with_retry(stop_after_attempt=2)
# model = qwen3_30b_instruct.with_structured_output(ExcludeSelectNoReason).with_retry(stop_after_attempt=2)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

REQUEST_INTERVAL = 1

import re


def pandas_str_to_series(s) -> Any | None:
    """字符串转为Series"""
    # 判断是否已经是 Series
    s = str(s)
    s = s.strip()

    # s为None或""或nan
    if s.strip() in ["", "nan", "None", "NaN"]:
        return None

    inner = s[s.find("(") + 1: s.rfind(")")]

    pattern = re.compile(r"(\w+)=('[^']*'|[^,]*)")
    data = {k: v.strip("'") if v.strip() != "nan" else None for k, v in pattern.findall(inner)}

    # 3️⃣ 转为 DataFrame
    df = pd.DataFrame([data])
    return df.iloc[0]


async def llm_exclude(product_name, top1, top2, top3, top4, top5) -> tuple[list[int], str]:
    await asyncio.sleep(REQUEST_INTERVAL)
    print(f"正在处理：product_name:{product_name}")
    # prompt_format = exclude_prompt.format(origin_product=product_name,
    #                                       top1_candidate=top1,
    #                                       top2_candidate=top2,
    #                                       top3_candidate=top3,
    #                                       top4_candidate=top4,
    #                                       top5_candidate=top5)
    prompt_format_no_reason = exclude_prompt_no_reason.format(origin_product=product_name,
                                                              top1_candidate=top1,
                                                              top2_candidate=top2,
                                                              top3_candidate=top3,
                                                              top4_candidate=top4,
                                                              top5_candidate=top5)

    try:
        excludeSelect = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_format_no_reason)],
                          # config={"callbacks": [opik_tracer]}
                          ),
            timeout=60)

        # indexs, reason = excludeSelect.exclude_index, excludeSelect.exclude_reason
        indexs = excludeSelect.exclude_index
        # indexs是否有不属于1-5之间的元素
        if not all(1 <= index <= 5 for index in indexs):
            print(f"商品：{product_name} indexs 发生异常: {indexs}")
            return [-1], "error"
    # 捕获所有错误，打印信息
    except Exception as e:
        print(f"商品：{product_name} 发生异常: {e}")
        return [-1], "error"
    return indexs, 'success'


async def process_llm_row_exclude(index, row, owner_df):
    top1 = pandas_str_to_series(row.相似商品1)
    top2 = pandas_str_to_series(row.相似商品2)
    top3 = pandas_str_to_series(row.相似商品3)
    top4 = pandas_str_to_series(row.相似商品4)
    top5 = pandas_str_to_series(row.相似商品5)

    idx, reason = await llm_exclude(row.商品名称,
                                    top1.商品名称 if top1 is not None else "",
                                    top2.商品名称 if top2 is not None else "",
                                    top3.商品名称 if top3 is not None else "",
                                    top4.商品名称 if top4 is not None else "",
                                    top5.商品名称 if top5 is not None else "")

    owner_df.at[index, '排除商品'] = str(idx)
    owner_df.at[index, '排除原因'] = reason


async def llm_exclude_fill(owner_df):
    print("开始LLM排除...")
    owner_df['排除商品'] = ''
    owner_df['排除原因'] = ''
    limiter = RateLimiter(rpm_limit=800, tpm_limit=40000)
    owner_subset = owner_df  # 这里可以改成 owner_df.iloc[:100] 测试
    batch_size = 10  # 每批次处理 10 条，可根据速率限制调整
    delay_seconds = 5  # 每批之间等待 3 秒，可根据 API 限速调整
    for start_idx in range(0, len(owner_subset), batch_size):
        est_tokens = batch_size * 900
        await limiter.record_call(est_tokens)

        end_idx = min(start_idx + batch_size, len(owner_subset))
        batch = owner_subset.iloc[start_idx:end_idx]
        print(f"正在处理第 {start_idx}~{end_idx - 1} 条（共 {len(owner_subset)} 条）...")

        # 按批次创建任务
        tasks = [
            process_llm_row_exclude(i, row, owner_df)
            for i, row in batch.iterrows()
        ]

        # 阻塞等待本批结果（而不是一次性 gather 所有）
        await asyncio.gather(*tasks)

        # 每批之间延迟，防止速率限制
        if end_idx < len(owner_subset):
            print(f"等待 {delay_seconds} 秒以避免触发频率限制...")
            await asyncio.sleep(delay_seconds)

    # 错误记录重试
    await retry_error_rows(owner_df, limiter, 10, 10)


async def handle_error_row(index, row, limiter, df):
    """重新处理错误匹配行"""

    await limiter.record_call(800)

    similar_product1 = pandas_str_to_series(row['相似商品1'])
    similar_product2 = pandas_str_to_series(row['相似商品2'])
    similar_product3 = pandas_str_to_series(row['相似商品3'])
    similar_product4 = pandas_str_to_series(row['相似商品4'])
    similar_product5 = pandas_str_to_series(row['相似商品5'])

    top1 = similar_product1.商品名称 if similar_product1 is not None else ""
    top2 = similar_product2.商品名称 if similar_product2 is not None else ""
    top3 = similar_product3.商品名称 if similar_product3 is not None else ""
    top4 = similar_product4.商品名称 if similar_product4 is not None else ""
    top5 = similar_product5.商品名称 if similar_product5 is not None else ""

    # 模型排序
    idx, reason = await llm_exclude(row.商品名称, top1, top2, top3, top4, top5)
    print(f"  匹配结果：{idx}")

    df.at[index, '排除商品'] = str(idx)
    df.at[index, '排除原因'] = reason


async def run_retry_batch(df, limiter, batch_size=10):
    """批次重试"""
    tasks = []
    for index, row in df.iterrows():
        # 判断是否是error记录，如果不是跳过这条记录
        if row['排除原因'] != 'error':
            continue
        tasks.append(handle_error_row(index, row, limiter, df))
        if len(tasks) >= batch_size:
            await asyncio.gather(*tasks)
            tasks = []
    if tasks:
        await asyncio.gather(*tasks)


async def retry_error_rows(df, limiter, max_rounds=5, batch_size=10):
    """
    df: 原始 DataFrame，包含 '相似商品' 字段
    max_rounds: 最多重试几轮
    batch_size: 每次协程批处理大小
    """

    for round_num in range(1, max_rounds + 1):
        # 过滤需要 retry 的记录
        error_rows = df[df['排除原因'] == 'error']

        retry_count = len(error_rows)
        print(f"\n====== 第 {round_num} 次遍历，需重试记录：{retry_count} 条 ======")

        if retry_count == 0:
            print("所有错误记录已处理完成！")
            break

        await run_retry_batch(df, limiter, batch_size=batch_size)

    else:
        print("\n⚠ 达到最大重试次数，但仍有未修复的错误记录")
