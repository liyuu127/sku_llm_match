import asyncio
import json
import math
import re
import time
import uuid
from typing import Any

import pandas as pd

from top5_exclude.llm_exclude_top5 import llm_exclude_fill
from top5_exclude.llm_match_top5 import llm_match_fill
from top5_exclude.sku_filter_top5 import preprocess_candidate_tokens, process_owner_data_async
from top5_exclude.utils import load_excel, save_excel_async, pic_download
from tqdm import tqdm

# 控制最大并发任务数
MAX_CONCURRENCY = 100
LLM_MATCH = True


def pandas_str_to_series(s) -> Any | None:
    """字符串转为Series"""
    # 判断是否已经是 Series
    s = str(s)
    s = s.strip()

    # s为None或""或nan
    if s in ["", "nan", "None", "NaN"]:
        return None

    inner = s[s.find("(") + 1: s.rfind(")")]

    pattern = re.compile(r"(\w+)=('[^']*'|[^,]*)")
    data = {k: v.strip("'") if v.strip() != "nan" else None for k, v in pattern.findall(inner)}

    # 转为 DataFrame
    df = pd.DataFrame([data])
    return df.iloc[0]


def pic_url_fill(llm_match_df: pd.DataFrame):
    """根据top1,top2,top3,llm_match 信息填充图片"""

    # 1. 对df单元格数据进行简单清洗，转为str,填充空值
    llm_match_df['商品ID'] = llm_match_df['商品ID'].fillna('').astype(str).str.strip()

    # 2. 提取商品名称，获取商品图片信息
    llm_match_df['origin_url'] = llm_match_df['图片'].fillna('').astype(str).str.strip()
    llm_match_df['llm_image_url'] = ''
    llm_match_df['top1_image_url'] = ''
    llm_match_df['top2_image_url'] = ''
    llm_match_df['top3_image_url'] = ''
    llm_match_df['top4_image_url'] = ''
    llm_match_df['top5_image_url'] = ''

    for index, row in llm_match_df.iterrows():
        top5 = pandas_str_to_series(row['相似商品5'])
        if top5 is not None:
            llm_match_df.at[index, 'top5_image_url'] = top5['图片']

        top4 = pandas_str_to_series(row['相似商品4'])
        if top4 is not None:
            llm_match_df.at[index, 'top4_image_url'] = top4['图片']

        top3 = pandas_str_to_series(row['相似商品3'])
        if top3 is not None:
            llm_match_df.at[index, 'top3_image_url'] = top3['图片']

        top2 = pandas_str_to_series(row['相似商品2'])
        if top2 is not None:
            llm_match_df.at[index, 'top2_image_url'] = top2['图片']

        top1 = pandas_str_to_series(row['相似商品1'])
        if top1 is not None:
            llm_match_df.at[index, 'top1_image_url'] = top1['图片']

        llm_p_row = pandas_str_to_series(row['相似商品'])
        if llm_p_row is not None:
            llm_match_df.at[index, 'llm_image_url'] = llm_p_row['图片']

    llm_match_df['top1_image_url'] = llm_match_df['top1_image_url'].fillna('').astype(str).str.strip()
    llm_match_df['top2_image_url'] = llm_match_df['top2_image_url'].fillna('').astype(str).str.strip()
    llm_match_df['top3_image_url'] = llm_match_df['top3_image_url'].fillna('').astype(str).str.strip()
    llm_match_df['top4_image_url'] = llm_match_df['top4_image_url'].fillna('').astype(str).str.strip()
    llm_match_df['top5_image_url'] = llm_match_df['top5_image_url'].fillna('').astype(str).str.strip()
    llm_match_df['llm_image_url'] = llm_match_df['llm_image_url'].fillna('').astype(str).str.strip()

    return llm_match_df


def clean_column(val):
    # pd.isna() 能同时捕捉 None, NaN, NaT
    if pd.isna(val) or val is None or val == "":
        return ""
    return str(val).strip()  # 建议加上 strip 去除前后空格


def format_row_to_json(row):
    data_dict = {
        "商品ID": clean_column(row['商品ID']),
        "商品名称": clean_column(row['商品名称']),
        "规格": clean_column(row['规格']),
        "折扣价": clean_column(row['折扣价']),
        "原价": clean_column(row['原价']),
        "销量": clean_column(row['销售']),
    }
    return json.dumps(data_dict, ensure_ascii=False)


def column_update(df: pd.DataFrame):
    """列值、顺序调整"""

    # '商品ID', '商品名称', '规格', '条码', '折扣价', '原价', '活动', '销售', '店内一级分类', '店内二级分类',
    #        '图片', '库存', '描述', '属性', '好评率', '想买人数', '点赞数', '最小订购数', '标识', '三级分类json',
    #        'tag', '美团一级分类', '美团二级分类', '美团三级分类', 'skuid', '相似商品1', '相似商品2', '相似商品3',
    #        '相似商品', 'human_match_id', 'source', '人工匹配商品名称', 'target_image_url'

    # df = df[['商品ID', '商品名称', '规格', '折扣价', '原价', '']]
    # 商品信息=商品名称+ 规格+ 折扣价+原价
    df['原始商品销量'] = 0
    df_new = df[
        ['商品ID', '商品名称', 'origin_url', '原始商品销量', '排除商品',
         '相似商品', 'llm_image_url', '相似商品1', 'top1_image_url',
         '相似商品2', 'top2_image_url', '相似商品3', 'top3_image_url',
         '相似商品4', 'top4_image_url', '相似商品5', 'top5_image_url', ]]
    df_new.rename(columns={'商品名称': '商品信息'}, inplace=True)

    # 迭代df调整内容
    for index, row in df.iterrows():
        df_new.at[index, '商品信息'] = format_row_to_json(row)
        df_new.at[index, '原始商品销量'] = int(row['销售'])
        llm_match = pandas_str_to_series(row['相似商品'])

        if llm_match is not None:
            df_new.at[index, '相似商品'] = format_row_to_json(llm_match)

        top1 = pandas_str_to_series(row['相似商品1'])
        if top1 is not None:
            df_new.at[index, '相似商品1'] = format_row_to_json(top1)

        top2 = pandas_str_to_series(row['相似商品2'])
        if top2 is not None:
            df_new.at[index, '相似商品2'] = format_row_to_json(top2)

        top3 = pandas_str_to_series(row['相似商品3'])
        if top3 is not None:
            df_new.at[index, '相似商品3'] = format_row_to_json(top3)

        top4 = pandas_str_to_series(row['相似商品4'])
        if top4 is not None:
            df_new.at[index, '相似商品4'] = format_row_to_json(top4)

        top5 = pandas_str_to_series(row['相似商品5'])
        if top5 is not None:
            df_new.at[index, '相似商品5'] = format_row_to_json(top5)

    return df_new


async def main():
    print("开始加载数据...")
    start_time = time.time()
    # owner_df 自有商品
    # target_df 对标商品
    owner_df_path = "../../data/top3相似人工标注数据_需要大模型识别.xlsx"
    target_df_path = "../../data/附件2-美团邻侣全量去重商品1109.xlsx"
    owner_df = load_excel(owner_df_path).iloc[:10]
    # owner_df = load_excel(owner_df_path)
    target_df = load_excel(target_df_path)
    owner_df = owner_df.drop_duplicates(subset=['商品ID'])
    target_df = target_df.drop_duplicates(subset=['商品ID'])
    # 打印前几行数据
    print(owner_df.head())
    print(f"待匹配数据加载完成，共{len(owner_df)}条记录")
    print(target_df.head())
    print(f"匹配目标数据加载完成，共{len(target_df)}条记录")

    tokens = await preprocess_candidate_tokens(target_df)
    print("target_df数据预处理完成")

    print("开始异步处理匹配任务...")
    await process_owner_data_async(owner_df, target_df, tokens)

    # 根据模型名拼装输出文件信息
    suffix = str(uuid.uuid4())
    model_name = "glm4-9b"
    # model_name = "qwen-3-30b"
    prompt_v = "prompt_v2"
    # prompt_v = "prompt_v3"
    prefix = "../../output/top5相似_"

    file_ext = ".xlsx"
    output_path = prefix + model_name + "_" + prompt_v + "_500_" + suffix
    output_path_nopic = prefix + "nopic_" + model_name + "_" + prompt_v + "_500_" + suffix
    output_path_nollm = prefix + "nollm" + model_name + "_" + prompt_v + "_500_" + suffix

    # await save_excel_async(owner_df, output_path_nollm+file_ext)
    if LLM_MATCH:
        await llm_exclude_fill(owner_df)
        await llm_match_fill(owner_df)
        # await save_excel_async(owner_df, output_path_nopic+file_ext)
        pic_url_fill(owner_df)
        df_new = column_update(owner_df)
        await batch_pic_file_download(df_new, file_ext, output_path)

    end_time = time.time()
    print(f"总耗时：{end_time - start_time:.2f}秒")


async def batch_pic_file_download(df_new, file_ext, output_path):
    # 单个文件记录大小
    FILE_BATCH_SIZE = 3
    for i, start_idx in enumerate(tqdm(range(0, len(df_new), FILE_BATCH_SIZE), desc="批次导出进度")):
        # 切片
        df_batch = df_new.iloc[start_idx: start_idx + FILE_BATCH_SIZE].copy()

        # 生成新路径
        new_file_name = f"{output_path}_batch_{i + 1}{file_ext}"
        # 执行函数
        pic_download(df_batch, new_file_name)


if __name__ == '__main__':
    asyncio.run(main())
