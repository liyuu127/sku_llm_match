# sku第一阶段过滤
import asyncio
from typing import List

import pandas as pd
from pandas import Series

from top5_exclude.utils import BRAND_DICTIONARY, extract_brand, clean_text, chinese_tokenize

MAX_CONCURRENCY = 100


def calculate_brand_similarity(brand1: str, brand2: str) -> float:
    """品牌相似度"""
    brand1 = brand1.strip().lower()
    brand2 = brand2.strip().lower()
    if not brand1 or not brand2:
        return 0.5
    if brand1 == brand2:
        return 1.0
    return 0.0


def jaccard_similarity(set1: set, set2: set) -> float:
    """分词相似度"""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def calculate_barcode_similarity(barcode1: str, barcode2: str) -> float:
    """条码相似度"""
    # 判断是否为空
    if not barcode1 or not barcode2:
        return 0.0
    return 1.0 if barcode1 == barcode2 else 0.0


async def calculate_similarity_for_single_product(
        row: Series, p_tokenize: set, p_brand: str,
        candidate_row: pd.Series, candidate_tokens: set
) -> tuple[float, Series]:
    """计算单个商品间相似度"""
    candidate_id = candidate_row.商品ID
    candidate_name = candidate_row.商品名称
    # p_barcode = row.条码
    # candidate_barcode = candidate_row.条码
    # # 条码是否匹配
    # similarity_barcode = calculate_barcode_similarity(p_barcode, candidate_barcode)
    # if similarity_barcode == 1.0:
    #     return 1.0, candidate_row
    candidate_brand = extract_brand(candidate_name, BRAND_DICTIONARY)
    similarity_tokens = jaccard_similarity(p_tokenize, candidate_tokens)
    similarity_brand = calculate_brand_similarity(p_brand, candidate_brand)

    # 品牌不匹配直接过滤
    similarity = similarity_tokens
    # if similarity_brand == 0.0:
    #     similarity = 0.0
    return similarity, candidate_row


async def find_top5_similar_combined_async(index, row, owner_df, candidate_df: pd.DataFrame,
                                           candidate_tokens_list: List[set],
                                           sem: asyncio.Semaphore,
                                           top_n=5):
    """"批量返回多个最相似的商品"""
    async with sem:
        product_name = row.商品名称
        price = row.原价
        p_name = clean_text(product_name)
        p_tokenize = await chinese_tokenize(p_name)
        p_brand = extract_brand(p_name, BRAND_DICTIONARY)

        tasks = []
        for idx, candidate_row in enumerate(candidate_df.itertuples(index=False)):
            candidate_tokens = candidate_tokens_list[idx]
            tasks.append(calculate_similarity_for_single_product(
                row, p_tokenize, p_brand, candidate_row, candidate_tokens
            ))

        results = await asyncio.gather(*tasks)
        # 如果相似度为1.0，直接返回第一个相似度为1.0的记录，top2 top2填充为空
        # if any(r[0] == 1.0 for r in results):
        if False:
            topn_combined = [r[1] for r in results if r[0] == 1.0]
        else:
            # 过滤相似度==0.0的记录
            # results = [r for r in results if r[0] > 0.0]
            sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

            topn_combined = [r[1] for r in sorted_results[:top_n]]
            # 过滤top3价格上下超出price 9倍以上的项
            # top3_combined = [r for r in top3_combined if abs(price - r.原价) < price * 10]

        while len(topn_combined) < top_n:
            topn_combined.append(None)

        # topn_combined按顺序迭代
        owner_df.loc[index, '相似商品1'] = topn_combined[0]
        owner_df.loc[index, '相似商品2'] = topn_combined[1]
        owner_df.loc[index, '相似商品3'] = topn_combined[2]
        owner_df.loc[index, '相似商品4'] = topn_combined[3]
        owner_df.loc[index, '相似商品5'] = topn_combined[4]


async def preprocess_candidate_tokens(candidate: pd.DataFrame) -> List[set]:
    """异步批量预处理分词"""
    tasks = [chinese_tokenize(clean_text(name)) for name in candidate['商品名称']]
    return await asyncio.gather(*tasks)


async def process_batch(owner_df: pd.DataFrame, candidate_df: pd.DataFrame, tokens: List[set],
                        batch_size: int, sem: asyncio.Semaphore):
    """
    直接遍历 owner_df，内部控制批次提交
    """
    tasks = []
    batch_count = 0  # 批次计数器
    total_rows = len(owner_df)

    # 使用 enumerate 获取当前遍历的序号 (i) 用于判断批次
    for i, (index, row) in enumerate(owner_df.iterrows()):

        # 添加任务到列表（此时不执行）
        tasks.append(find_top5_similar_combined_async(index, row, owner_df, candidate_df, tokens, sem))

        # 当任务积攒到 batch_size 时，或者已经是最后一条数据时
        if len(tasks) == batch_size:
            batch_count += 1

            # 1. 并发执行当前批次
            await asyncio.gather(*tasks)

            # 2. 计算当前批次的范围用于打印
            end_pos = i + 1
            start_pos = end_pos - batch_size + 1
            print(f"已完成批次：{batch_count} | 数据范围：{start_pos}~{end_pos} | 进度：{end_pos}/{total_rows}")

            # 3. 清空任务列表，准备下一批
            tasks = []

    # 处理剩余的尾巴（最后一批不足 batch_size 的情况）
    if tasks:
        batch_count += 1
        await asyncio.gather(*tasks)

        # 打印最后一次
        end_pos = total_rows
        start_pos = end_pos - len(tasks) + 1
        print(f"已完成批次：{batch_count} (最终批) | 数据范围：{start_pos}~{end_pos} | 进度：{end_pos}/{total_rows}")


async def process_owner_data_async(owner_df: pd.DataFrame, ele_df: pd.DataFrame, tokens: List[set],
                                   batch_size: int = 100):
    """分批次处理任务"""
    owner_df['相似商品1'] = ''
    owner_df['相似商品2'] = ''
    owner_df['相似商品3'] = ''
    owner_df['相似商品4'] = ''
    owner_df['相似商品5'] = ''
    total_items = len(owner_df)
    num_batches = (total_items + batch_size - 1) // batch_size
    print(f"开始处理数据，共{num_batches}批次，每批{batch_size}条。")
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    await process_batch(owner_df, ele_df, tokens, batch_size, sem)

    owner_df['相似商品1'] = owner_df['相似商品1'].fillna('').astype(str).str.strip()
    owner_df['相似商品2'] = owner_df['相似商品2'].fillna('').astype(str).str.strip()
    owner_df['相似商品3'] = owner_df['相似商品3'].fillna('').astype(str).str.strip()
    owner_df['相似商品4'] = owner_df['相似商品4'].fillna('').astype(str).str.strip()
    owner_df['相似商品5'] = owner_df['相似商品5'].fillna('').astype(str).str.strip()
