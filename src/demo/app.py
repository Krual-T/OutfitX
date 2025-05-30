# demo.py

import pickle
import random
import base64
from collections import defaultdict

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
        ckpt = CKPT_ROOT/ "complementary_item_retrieval"/f"{cfg_model.model_name}_best_Recall@1.pth"
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
    samples_index = random.sample(range(len(dataset)), batch_size)
    raws = [dataset[i] for i in samples_index]
    batch = processor(raws)

    inp = {k: (v if k=='task' else v.to(DEVICE)) for k,v in batch['input_dict'].items()}
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=False):
        logits = model(**inp).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))

    results = []
    dataset_dir = dataset.dataset_dir
    for i, (query, label) in enumerate(raws):
        # 🚀 只存路径，不 open
        paths = [
            str(dataset_dir / 'images' / f'{item.item_id}.jpg')
            for item in query.outfit
        ]
        results.append({
            "label": label,
            "prob": float(probs[i]),
            "paths": paths,   # 用 paths 字段
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
@torch.no_grad()
def run_cir_demo(model, dataset, processor, batch_size: int = 10):
    model.eval()
    samples_index = random.sample(range(len(dataset)), batch_size)
    raws = [dataset[i] for i in samples_index]
    batch = processor(raws)
    inp = {k: (v if k=='task' else v.to(DEVICE)) for k,v in batch['input_dict'].items()}
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=False):
        y_hats = model(**inp) # (B, D)
    pos_items_id = batch['pos_item_id']
    candidate_pools = dataset.candidate_pools
    base_img_path = dataset.dataset_dir / 'images'
    results = []
    for i,(query,_) in enumerate(raws):
        item_id = int(pos_items_id[i])
        c_id = dataset.metadata[item_id]['category_id']
        pool = candidate_pools[c_id] #（3000，D）
        pool_emb = pool['embeddings'].to(DEVICE)
        with autocast(device_type=DEVICE.type, enabled=True):
            # y_hats[i,:] （1,D）
            dist = torch.cdist(y_hats[i:i+1,:], pool_emb, p=2).squeeze(0)
            top_k_index = torch.topk(dist, k=10, largest=False).indices.cpu().numpy()  # [1,K] k 个 index
        partial_outfit_path = [
            base_img_path/ f'{item.item_id}.jpg' for item in query.outfit
        ]
        item_list = pool['item_ids']
        retrieval_items_id = [base_img_path/ f'{item_list[i]}.jpg' for i in top_k_index]
        results.append({
            'partial_outfit': partial_outfit_path, # [path]
            'retrieval_items': retrieval_items_id, # [path] len=10
            'gt_item': base_img_path/ f'{item_id}.jpg' # path
        })
    return results


# ─── FITB 分页渲染 ───────────────────────────
def run_fitb_demo(model, dataset, processor, batch_size: int = 10):
    model.eval()
    samples_index = random.sample(range(len(dataset)), batch_size)
    raws = [dataset[i] for i in samples_index]
    batch = processor(raws)
    inp = {k: (v if k=='task' else v.to(DEVICE)) for k,v in batch['input_dict'].items()}
    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=False):
        y_hats_embedding = model(**inp).unsqueeze(1) # (B,1, D)
        candidate_item_embeddings = batch['candidate_item_embedding'].to(DEVICE)  # [B,4,D]
        dists = torch.cdist(y_hats_embedding, candidate_item_embeddings, p=2).squeeze(1)  # [B,1,4]->[B,4]
    y_hats_index = torch.argmin(dists, dim=-1).cpu().numpy()  # [B]
    y_index = batch['answer_index'].cpu().numpy()  # [B]
    results = []
    base_img_path = dataset.dataset_dir / 'images'
    for i,index in enumerate(samples_index):
        test_item = dataset.fitb_dataset[index]
        candidate_item_ids = test_item['answers']
        query_items_id = [base_img_path/ f'{item_id}.jpg' for item_id in test_item['question']]
        y_id = base_img_path/ f'{candidate_item_ids[y_index[i]]}.jpg'
        y_hat_id = base_img_path/ f'{candidate_item_ids[y_hats_index[i]]}.jpg'
        results.append({
            'partial_outfit': query_items_id, # [path]
            'y_id': y_id, # path
            'y_hat_id': y_hat_id, # path
            'candidate_items': [base_img_path/ f'{item_id}.jpg' for item_id in candidate_item_ids]
        })
    return results

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
# ─── 在 Blocks 定义里，用一个 HTML 输出区域 ─────────────────
with (gr.Blocks(css=css) as demo):
    gr.Markdown(
        "<h1 style='text-align:center;'>🌟 基于CNN-Transformer跨模态融合的穿搭推荐模型研究可视化展板</h1>"
    )
    with gr.TabItem("服装兼容性预测（CP）"):
        btn = gr.Button("生成 CP 示例 🚀")
        cp_html_output = gr.HTML()
        def cp_pipeline():
            results = run_cp_demo(*load_task("CP"))
            html = ""
            for item in results:
                html += (
                    "<div style='margin-bottom:16px;'>"
                    f"<p style='font-size:24px;'><strong>标签：{item['label']} ｜ 兼容性分数：{item['prob']:.3f}</strong></p>"
                    "<div style='display:flex; overflow-x:auto; white-space:nowrap;'>"
                )
                # 👇 这里改为 Base64 内联
                for path in item["paths"]:
                    img_bytes = Path(path).read_bytes()
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    # data URI：前缀根据你的图类型（jpg/png）
                    html += (
                        f"<img src='data:image/jpeg;base64,{b64}' "
                        "style='display:inline-block; margin-right:8px;width:10%; height:auto;' />"
                    )
                html += "</div></div>"
            return html
        btn.click(fn=cp_pipeline, outputs=cp_html_output)

    with gr.TabItem("服装互补单品检索（CIR）"):
        btn_cir = gr.Button("生成 CIR 示例 👗")
        cir_html_output = gr.HTML()


        def cir_pipeline():
            results = run_cir_demo(*load_task("CIR"))
            html = ""
            for item in results:
                # 整体一行两个区块
                html += "<div style='display:flex; margin-bottom:24px;'>"

                # —— 左侧：Query 部分服装
                html += (
                    "<div style='flex:1; padding-right:16px;'>"
                    "<p style='font-size:20px; font-weight:bold;'>Query 部分服装</p>"
                    "<div style='display:flex; overflow-x:auto; white-space:nowrap;'>"
                )
                for p in item["partial_outfit"]:
                    b64 = base64.b64encode(Path(p).read_bytes()).decode()
                    html += (
                        f"<img src='data:image/jpeg;base64,{b64}' "
                        "style='width:80px; height:auto; margin-right:8px; "
                        "border-radius:6px; box-shadow:0 0 4px rgba(0,0,0,0.2);'/>"
                    )
                html += "</div></div>"

                # —— 右侧：Top-10 检索结果
                gt = str(item["gt_item"])
                recs = [str(p) for p in item["retrieval_items"]]
                found = gt in recs
                if not found:
                    recs = [gt] + recs

                html += (
                    "<div style='flex:1;'>"
                    "<p style='font-size:20px; font-weight:bold;'>Top-10 检索结果</p>"
                    "<div style='display:flex; overflow-x:auto; white-space:nowrap;'>"
                )
                for idx, p in enumerate(recs):
                    b64 = base64.b64encode(Path(p).read_bytes()).decode()
                    # 样式区分
                    if p == gt and found:
                        bd = "4px solid limegreen"
                    elif p == gt and not found and idx == 0:
                        bd = "4px solid red"
                    else:
                        bd = "1px solid #ccc"
                    html += (
                        f"<img src='data:image/jpeg;base64,{b64}' "
                        f"style='width:80px; height:auto; margin-right:8px; "
                        f"border:{bd}; border-radius:6px;'/>"
                    )
                html += "</div></div>"

                html += "</div>"  # 结束这一行
            return html


        btn_cir.click(fn=cir_pipeline, outputs=cir_html_output)

    with gr.TabItem("服装填空（FITB）"):
        btn_cir = gr.Button("生成 FITB 示例 👗")
        fitb_html_output = gr.HTML()
        def fitb_pipeline():
            results = run_fitb_demo(*load_task("FITB"))
            html = ""
            for item in results:
                # 整体一行两个区块
                html += "<div style='display:flex; margin-bottom:24px;'>"
                # —— 左侧：Query 部分服装
                html += (
                    "<div style='flex:1; padding-right:16px;'>"
                    "<p style='font-size:20px; font-weight:bold;'>Query 部分服装</p>"
                    "<div style='display:flex; overflow-x:auto; white-space:nowrap;'>"
                )
                for p in item["partial_outfit"]:
                    b64 = base64.b64encode(Path(p).read_bytes()).decode()
                    html += (
                        f"<img src='data:image/jpeg;base64,{b64}' "
                        "style='width:80px; height:auto; margin-right:8px; "
                        "border-radius:6px; box-shadow:0 0 4px rgba(0,0,0,0.2);'/>"
                    )
                html += "</div></div>"
                recs = [str(p) for p in item['candidate_items']]
                html += (
                    "<div style='flex:1;'>"
                    "<p style='font-size:20px; font-weight:bold;'>选项</p>"
                    "<div style='display:flex; overflow-x:auto; white-space:nowrap;'>"
                )
                for idx, p in enumerate(recs):
                    b64 = base64.b64encode(Path(p).read_bytes()).decode()

                    y_hat_id = str(item['y_hat_id'])
                    y_id = str(item['y_id'])
                    if p == y_id:
                        bd = "4px solid limegreen"
                    elif p == y_hat_id:
                        bd = "4px solid red"
                    else:
                        bd = "1px solid #ccc"
                    html += (
                        f"<img src='data:image/jpeg;base64,{b64}' "
                        f"style='width:80px; height:auto; margin-right:8px; "
                        f"border:{bd}; border-radius:6px;'/>"
                    )
                html += "</div></div>"

                html += "</div>"  # 结束这一行
            return html
        btn_cir.click(fn=fitb_pipeline, outputs=fitb_html_output)


if __name__ == "__main__":
    demo.launch(
        server_port=6006,
        allowed_paths=[str(ROOT_DIR / 'datasets' / 'polyvore' / 'images')]
    )
