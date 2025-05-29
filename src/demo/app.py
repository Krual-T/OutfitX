# demo.py

import pickle
import random

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast
from PIL import Image
import gradio as gr

# ──── 来自项目的 imports ─────────────
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.models.datatypes import (
    OutfitCompatibilityPredictionTask,
    OutfitComplementaryItemRetrievalTask,
    OutfitFillInTheBlankTask,
)
from src.project_settings.info import PROJECT_DIR as ROOT_DIR
from src.models.processor import OutfitTransformerProcessorFactory
from src.trains.configs.compatibility_prediction_train_config import CompatibilityPredictionTrainConfig
from src.trains.configs.complementary_item_retrieval_train_config import ComplementaryItemRetrievalTrainConfig
from src.trains.configs.fill_in_the_blank_train_config import FillInTheBlankTrainConfig
from src.trains.datasets import PolyvoreItemDataset
from src.trains.datasets.polyvore.polyvore_compatibility_dataset import PolyvoreCompatibilityPredictionDataset
from src.trains.datasets.polyvore.polyvore_complementary_item_retrieval_dataset import PolyvoreComplementaryItemRetrievalDataset
from src.trains.datasets.polyvore.polyvore_fill_in_the_blank_dataset import PolyvoreFillInTheBlankDataset
# ────────────────────────────────────────

# 全局设备 & Config
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_cp    = CompatibilityPredictionTrainConfig()
cfg_cir   = ComplementaryItemRetrievalTrainConfig()
cfg_fitb  = FillInTheBlankTrainConfig()
cfg_model = OutfitTransformerConfig()

# checkpoint 根目录（../checkpoints/{polyvore_type}/{task}/）
CKPT_ROOT = Path(cfg_cp.checkpoint_dir).parent
precomputed_embedding_dir = ROOT_DIR / 'datasets' / 'polyvore' / 'precomputed_embeddings'
# 每页样本数
CP_PAGE_SIZE   = 10
CIR_PAGE_SIZE  = 10
FITB_PAGE_SIZE = 1

# ─── 预加载全库 Embedding Pool（用于 CIR & FITB） ────────
def load_embeddings(embed_file_prefix: str = "embedding_subset_") -> dict:
    """
    合并所有 embedding_subset_{rank}.pkl 文件，返回包含完整 id 列表和嵌入矩阵的 dict。
    """
    embedding_dir = precomputed_embedding_dir
    prefix = embed_file_prefix
    files = sorted(embedding_dir.glob(f"{prefix}*.pkl"))
    if not files:
        raise FileNotFoundError(f"找不到任何文件: {prefix}*.pkl")

    all_ids = []
    all_embeddings = []

    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            all_ids.extend(data['ids'])
            all_embeddings.append(data['embeddings'])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return {item_id: embedding for item_id, embedding in zip(all_ids, all_embeddings)}


prefix = f"{cfg_model.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
emb_dict = load_embeddings(embed_file_prefix=prefix)



# ─── 统一加载逻辑 ─────────────────────────
def load_task(task_name: str):
    """
    根据 'CP' / 'CIR' / 'FITB'，动态加载 model / dataset / processor
    """

    model = OutfitTransformer(cfg_model)
    if task_name == "CP":
        ckpt = CKPT_ROOT/ "compatibility_prediction"/f"{cfg_model.model_name}_best_AUC.pth"
        dataset_cls = PolyvoreCompatibilityPredictionDataset
        task = OutfitCompatibilityPredictionTask
    elif task_name == "CIR":
        ckpt = CKPT_ROOT/ "complementary_item_retrieval"/f"{cfg_model.model_name}_best_Recall@1.pth"
        dataset_cls = PolyvoreComplementaryItemRetrievalDataset
        task = OutfitComplementaryItemRetrievalTask
    elif task_name == "FITB":
        ckpt = CKPT_ROOT/ "fill_in_the_blank"/f"{cfg_model.model_name}_best_Recall@1.pth"
        dataset_cls = PolyvoreFillInTheBlankDataset
        task = OutfitFillInTheBlankTask
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # load checkpoint
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt_dict['model'])
    model.eval()

    dataset = dataset_cls(
        polyvore_type='nondisjoint',
        mode='test',
        dataset_dir=ROOT_DIR / 'datasets' / 'polyvore',
        embedding_dict=emb_dict,
        load_image=False,  # image via item_id 载入
    )
    processor = OutfitTransformerProcessorFactory.get_processor(
        task=task, cfg=cfg_model, run_mode='test'
    )
    return model.to(DEVICE), dataset, processor


# ---------- 推理函数 ----------
def run_cp_demo(model, dataset, processor, batch_size: int = 10):
    model.eval()
    samples_index = random.sample(range(0, len(dataset)), batch_size)
    raws = [dataset[i] for i in samples_index]
    batch = processor(raws)

    inp = {k: (v if k == 'task' else v.to(DEVICE)) for k, v in batch['input_dict'].items()}
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=False):
        logits = model(**inp).squeeze(-1).cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))

    results = []
    dataset_dir = dataset.dataset_dir
    for i, (query, label) in enumerate(raws):
        results.append({
            "label": label,
            "prob": float(probs[i]),
            "images": [Image.open(dataset_dir / 'images' / f'{item.item_id}.jpg').convert("RGB") for item in query.outfit]
        })
    return results

# ---------- 展示函数 ----------
def display_cp_demo(results):
    components = []
    for item in results:
        components.append(gr.Markdown(f"**标签：{item['label']}｜兼容性分数：{item['prob']:.3f}**"))

        imgs = [
            gr.Image(value=img, type="pil", show_label=False)
            for img in item["images"]
        ]
        row = gr.Row(components=imgs)
        components.append(row)

    return components

# ---------- CSS 样式 ----------
css = """
#scroll-row {
    overflow-x: auto;
    flex-wrap: nowrap;
    gap: 10px;
    padding-bottom: 8px;
}
#scroll-row > div {
    flex: 0 0 auto;
}
#scroll-row img {
    border-radius: 8px;
    max-height: 120px;
    transition: transform 0.2s;
}
#scroll-row img:hover {
    transform: scale(1.05);
}
"""

# ─── CIR 分页渲染 (torch.topk + GPU) ──────────

# ─── FITB 分页渲染 ───────────────────────────

css = """
#scroll-row {
    overflow-x: auto;
    flex-wrap: nowrap;
    gap: 10px;
}
#scroll-row > div {
    flex: 0 0 auto;
}
"""
# ─── Gradio 布局 ──────────────────────────────
with gr.Blocks(css=css) as demo:
    gr.Markdown(
        "<h1 style='text-align:center;'>🌟 基于CNN-Transformer跨模态融合的穿搭推荐模型研究可视化展板</h1>"
    )

    with gr.Tabs():
        with gr.TabItem("服装兼容性预测（CP）"):
            btn = gr.Button("生成 CP 示例 🚀")

            # 文本区域：显示多组标签+分数
            text_output = gr.Markdown()

            # 图片画廊：每个子列表按行渲染
            gallery = gr.Gallery(
                label="Outfits",
                elem_id="cp-gallery",
                show_label=False,
                # rows=batch_size, columns=最大单行图片数（可按需改）
                rows=CP_PAGE_SIZE, columns=CP_PAGE_SIZE
            )

            def full_pipeline():
                results = run_cp_demo(*load_task("CP"))
                # 1) 构造 Markdown 文本：每组一个段落
                md = ""
                for i, item in enumerate(results, 1):
                    md += f"**{i}. 标签：{item['label']}  ｜ 兼容性分数：{item['prob']:.3f}**\n\n"
                # 2) 构造嵌套列表：每个 sublist 是一行 outfit 图像
                nested_imgs = [item["images"] for item in results]
                return md, nested_imgs

            btn.click(fn=full_pipeline, outputs=[text_output, gallery])



if __name__ == "__main__":
    demo.launch(server_port=6006)
