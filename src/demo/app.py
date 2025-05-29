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

# 每页样本数
CP_PAGE_SIZE   = 10
CIR_PAGE_SIZE  = 10
FITB_PAGE_SIZE = 1

# ─── 预加载全库 Embedding Pool（用于 CIR & FITB） ────────
# 直接复用 CP 预计算的 embedding，效果近似
_prefix = f"{cfg_model.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
_ids, _embs = [], []
for pkl in sorted(Path(cfg_cp.precomputed_embedding_dir).glob(f"{_prefix}*.pkl")):
    data = pickle.load(open(pkl, 'rb'))
    _ids.extend(data['ids'])
    _embs.append(data['embeddings'])
_emb_array = np.concatenate(_embs, axis=0)  # (N, d)
POOL_IDS   = _ids                          # list of length N
POOL_EMBS  = torch.from_numpy(_emb_array).to(DEVICE)  # (N, d) on GPU


# ─── 统一加载逻辑 ─────────────────────────
def load_task(task_name: str):
    """
    根据 'CP' / 'CIR' / 'FITB'，动态加载 model / dataset / processor
    """
    emb_dict = None
    model = OutfitTransformer(cfg_model)
    if task_name == "CP":
        ckpt = CKPT_ROOT/ cfg_cp.polyvore_type /"compatibility_prediction"/f"{cfg_model.model_name}_best_AUC.pth"
        dataset_cls = PolyvoreCompatibilityPredictionDataset
        task = OutfitCompatibilityPredictionTask
    elif task_name == "CIR":
        ckpt = CKPT_ROOT/ cfg_cir.polyvore_type /"complementary_item_retrieval"/f"{cfg_model.model_name}_best_Recall@1.pth"
        dataset_cls = PolyvoreComplementaryItemRetrievalDataset
        task = OutfitComplementaryItemRetrievalTask
    elif task_name == "FITB":
        ckpt = CKPT_ROOT/ cfg_fitb.polyvore_type /"fill_in_the_blank"/f"{cfg_model.model_name}_best_Recall@1.pth"
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
    for i, (query, label) in enumerate(raws):
        results.append({
            "label": label,
            "prob": float(probs[i]),
            "images": [Image.open(item.item_id).convert("RGB") for item in query.outfit]
        })
    return results

# ---------- 展示函数 ----------
def display_cp_demo(results):
    with gr.Column() as block:
        for item in results:
            gr.Markdown(f"**标签：{item['label']}｜兼容性分数：{item['prob']:.3f}**")
            with gr.Row(elem_id="scroll-row"):
                for img in item["images"]:
                    with gr.Column():
                        gr.Image(value=img, type="pil", show_label=False)
    return block

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
def render_cir_page(model, dataset, processor, page: int):
    start = (page-1)*CIR_PAGE_SIZE
    end   = min(len(dataset), start+CIR_PAGE_SIZE)
    raws  = [dataset[i] for i in range(start,end)]
    batch = processor(raws)

    # 得到 target embedding (B, d)
    inp = {k:(v if k=='task' else v.to(DEVICE)) for k,v in batch['input_dict'].items()}
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=False):
        t_emb = model(**inp)  # (B, d_embed)

    succ_html, fail_html = [], []
    for i,(query,_) in enumerate(raws):
        # 1) 在 GPU 上算 distance & topk
        emb = t_emb[i].unsqueeze(0)                # (1, d)
        dists = torch.norm(POOL_EMBS - emb, dim=1) # (N,)
        topk_idxs = dists.topk(10, largest=False).indices.cpu().numpy()
        retrieved = [POOL_IDS[j] for j in topk_idxs]

        # 2) 构建左侧 partial outfit
        left_html = ""
        for item in query.outfit:
            path = Path(cfg_cir.dataset_dir)/'images'/f"{item.item_id}.jpg"
            left_html += f"<img src='file={path}' style='height:100px;margin-right:4px'/>"
        left_div = f"<div style='display:flex;overflow-x:auto;width:300px'>{left_html}</div>"

        # 3) 右侧 top10检索，若包含 true_id 则高亮 green border
        true_id = query.target_item.item_id
        right_html = ""
        for rid in retrieved:
            border = "3px solid green" if rid==true_id else "1px solid #ccc"
            path = Path(cfg_cir.dataset_dir)/'images'/f"{rid}.jpg"
            right_html += f"<img src='file={path}' style='height:80px;margin:2px;border:{border}'/>"
        right_div = f"<div style='display:flex;overflow-x:auto;width:300px'>{right_html}</div>"

        entry = f"<div style='margin:6px'>{left_div}<br>{right_div}</div>"
        # 若包含 true_id 则视为成功，否则失败
        (succ_html if true_id in retrieved else fail_html).append(entry)

    # HTML table
    rows = max(len(succ_html), len(fail_html))
    html = "<table><tr><th>✅ 包含 GT</th><th>❌ 不含 GT</th></tr>"
    for i in range(rows):
        L = succ_html[i] if i<len(succ_html) else ""
        R = fail_html[i] if i<len(fail_html) else ""
        html += f"<tr><td style='vertical-align:top'>{L}</td><td style='vertical-align:top'>{R}</td></tr>"
    html += "</table>"
    return html

# ─── FITB 分页渲染 ───────────────────────────
def render_fitb_page(model, dataset, processor, page: int):
    idx = page - 1
    query, candidate_ids, answer_idx = dataset[idx]  # 假设 dataset 返回 3 元组
    batch = processor([(query, candidate_ids, answer_idx)])
    # model 生成 target embedding
    inp = {k:(v if k=='task' else v.to(DEVICE)) for k,v in batch['input_dict'].items()}
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=False):
        t_emb = model(**inp).squeeze(0)  # (d_embed,)

    # Load candidate embeddings & 计算距离
    c_embs = torch.stack([
        torch.from_numpy(POOL_EMBS.cpu().numpy()[POOL_IDS.index(cid)])
        for cid in candidate_ids
    ]).to(DEVICE)  # (4, d)
    dists = torch.norm(c_embs - t_emb.unsqueeze(0), dim=1)  # (4,)
    pred_idx = int(dists.argmin().cpu().item())

    # 左侧 Outfit
    left_html = ""
    for item in query.outfit:
        path = Path(cfg_fitb.dataset_dir)/'images'/f"{item.item_id}.jpg"
        left_html += f"<img src='file={path}' style='height:100px;margin-right:4px'/>"
    left_div = f"<div style='display:flex;overflow-x:auto;width:300px'>{left_html}</div>"

    # 右侧 4 候选，高亮
    right_html = ""
    for i,cid in enumerate(candidate_ids):
        path = Path(cfg_fitb.dataset_dir)/'images'/f"{cid}.jpg"
        if i == pred_idx:
            border = "3px solid green" if i==answer_idx else "3px solid red"
        elif i == answer_idx:
            border = "3px solid green"
        else:
            border = "1px solid #ccc"
        right_html += f"<img src='file={path}' style='height:80px;margin:2px;border:{border}'/>"
    right_div = f"<div style='display:flex;overflow-x:auto;width:300px'>{right_html}</div>"

    html = f"<div style='margin:6px'>{left_div}<br>{right_div}</div>"
    return html

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
with gr.Blocks(css = css) as demo:
    gr.Markdown("# 🌟 基于CNN-Transformer跨模态融合的穿搭推荐模型研究可视化展板")

    with gr.Tabs():
        with gr.TabItem("服装兼容性预测（CP）"):
            btn = gr.Button("生成 CP 示例")
            result_area = gr.Column()
            def full_pipeline():
                results = run_cp_demo(*load_task("CP"))
                return display_cp_demo(results)
            btn.click(fn=full_pipeline, outputs=result_area)



if __name__ == "__main__":
    demo.launch()
