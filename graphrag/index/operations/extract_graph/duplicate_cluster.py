import numpy as np
import pandas as pd
from typing import Any
from graphrag.language_model.manager import ModelManager
from graphrag.config.models.graph_rag_config import GraphRagConfig

# 相似度阈值，可根据需要调整
DEFAULT_SIMILARITY_THRESHOLD = 0.9

async def cluster_entities(
    df: pd.DataFrame,
    config: GraphRagConfig,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    对实体按 description 嵌入相似度聚类，合并相似度 >= threshold 的记录。
    保留首条作为聚合代表，合并 frequency、text_unit_ids。
    """
    if df.empty:
        return df
    # 获取嵌入模型配置并实例化
    lm_conf = config.get_language_model_config(config.embed_text.model_id)
    embed_model = ModelManager().get_or_create_embedding_model(
        name="cluster_entity",
        model_type=lm_conf.type,
        **lm_conf.model_dump(),
    )
    # 准备待嵌入文本，使用 description
    texts = df["description"].fillna("").astype(str).tolist()
    # 批量生成嵌入
    embeddings: list[list[float]] = await embed_model.aembed_batch(texts)
    vectors = [np.array(vec) for vec in embeddings]
    n = len(vectors)
    labels = [-1] * n
    cluster_id = 0
    # 贪心聚类
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = cluster_id
        v_i = vectors[i]
        norm_i = np.linalg.norm(v_i)
        for j in range(i + 1, n):
            if labels[j] != -1:
                continue
            v_j = vectors[j]
            norm_j = np.linalg.norm(v_j)
            denom = norm_i * norm_j
            sim = float(np.dot(v_i, v_j) / denom) if denom > 0 else 0.0
            if sim >= threshold:
                labels[j] = cluster_id
        cluster_id += 1
    # 合并同簇
    df = df.copy().reset_index(drop=True)
    df["_cluster_id"] = labels
    merged = []
    for cid, group in df.groupby("_cluster_id", sort=False):
        rep = group.iloc[0].copy()
        # 合并 frequency
        if "frequency" in group.columns:
            rep["frequency"] = int(group["frequency"].sum())
        # 合并 text_unit_ids
        if "text_unit_ids" in group.columns:
            merged_ids = []
            for ids in group["text_unit_ids"]:
                if isinstance(ids, list):
                    merged_ids.extend(ids)
            # 保持顺序去重
            rep["text_unit_ids"] = list(dict.fromkeys(merged_ids))
        merged.append(rep.drop(labels=["_cluster_id"]))
    return pd.DataFrame(merged)

async def cluster_relationships(
    df: pd.DataFrame,
    config: GraphRagConfig,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    对关系按 description 嵌入相似度聚类，合并相似度 >= threshold 的记录。
    保留首条作为聚合代表，合并 weight、text_unit_ids。
    """
    if df.empty:
        return df
    # 获取嵌入模型配置并实例化
    lm_conf = config.get_language_model_config(config.embed_text.model_id)
    embed_model = ModelManager().get_or_create_embedding_model(
        name="cluster_relationship",
        model_type=lm_conf.type,
        **lm_conf.model_dump(),
    )
    # 准备待嵌入文本，使用 description
    texts = df["description"].fillna("").astype(str).tolist()
    embeddings: list[list[float]] = await embed_model.aembed_batch(texts)
    vectors = [np.array(vec) for vec in embeddings]
    n = len(vectors)
    labels = [-1] * n
    cluster_id = 0
    # 贪心聚类
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = cluster_id
        v_i = vectors[i]
        norm_i = np.linalg.norm(v_i)
        for j in range(i + 1, n):
            if labels[j] != -1:
                continue
            v_j = vectors[j]
            norm_j = np.linalg.norm(v_j)
            denom = norm_i * norm_j
            sim = float(np.dot(v_i, v_j) / denom) if denom > 0 else 0.0
            if sim >= threshold:
                labels[j] = cluster_id
        cluster_id += 1
    # 合并同簇
    df = df.copy().reset_index(drop=True)
    df["_cluster_id"] = labels
    merged = []
    for cid, group in df.groupby("_cluster_id", sort=False):
        rep = group.iloc[0].copy()
        # 合并 weight
        if "weight" in group.columns:
            rep["weight"] = float(group["weight"].sum())
        # 合并 text_unit_ids
        if "text_unit_ids" in group.columns:
            merged_ids = []
            for ids in group["text_unit_ids"]:
                if isinstance(ids, list):
                    merged_ids.extend(ids)
            rep["text_unit_ids"] = list(dict.fromkeys(merged_ids))
        merged.append(rep.drop(labels=["_cluster_id"]))
    return pd.DataFrame(merged) 