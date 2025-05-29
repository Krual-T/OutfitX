import random
from pathlib import Path
import gradio as gr
from PIL import Image
from src.project_settings.info import PROJECT_DIR as ROOT_DIR

# 图像路径
IMAGES_DIR = ROOT_DIR / "datasets" / "polyvore" / "images"
ALL_IMAGES = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
if not ALL_IMAGES:
    raise RuntimeError("请在当前目录下准备一个 images/ 文件夹，里面放 jpg/png 图片")

# 通用标签函数（标注预测 / 正确）
def tag(p, mark):
    return f"{'🟢' if mark else '🔴'} {p.name}"

# 1) CP 兼容性预测
def cp_demo():
    results = []
    for _ in range(8):
        outfit = random.sample(ALL_IMAGES, random.randint(2, 5))
        prob = random.random()
        gt = random.randint(0, 1)
        pred = 1 if prob > 0.5 else 0
        ok = (pred == gt)
        results.append((outfit, prob, gt, ok))

    succ_imgs, fail_imgs = [], []
    for outfit, prob, gt, ok in results:
        caption = f"p={prob:.2f}, gt={gt}"
        images = [Image.open(p).convert("RGB") for p in outfit]
        for img in images:
            if ok:
                succ_imgs.append((img, caption))
            else:
                fail_imgs.append((img, caption))

    return succ_imgs, "", fail_imgs, ""

# 2) CIR 互补项检索
def cir_demo():
    partial = random.sample(ALL_IMAGES, random.randint(2, 4))
    true_id = random.choice(ALL_IMAGES)
    cands = random.sample(ALL_IMAGES, 10)
    if random.random() < 0.5 and true_id not in cands:
        cands[0] = true_id

    partial_imgs = [(Image.open(p).convert("RGB"), f"Partial: {p.name}") for p in partial]
    cand_imgs = [(Image.open(p).convert("RGB"), tag(p, p == true_id)) for p in cands]
    return partial_imgs, cand_imgs

# 3) FITB 填空任务
def fitb_demo():
    outfit_full = random.sample(ALL_IMAGES, random.randint(3, 5))
    target = random.choice(outfit_full)
    outfit_missing = [p for p in outfit_full if p != target]

    cands = random.sample(ALL_IMAGES, 3) + [target]
    random.shuffle(cands)
    pred_idx = random.randint(0, 3)

    missing_imgs = [(Image.open(p).convert("RGB"), f"Known: {p.name}") for p in outfit_missing]

    cand_imgs = []
    for idx, p in enumerate(cands):
        if idx == pred_idx and p == target:
            label = "🟢 predicted ✔"
        elif idx == pred_idx:
            label = "🔴 predicted ✖"
        elif p == target:
            label = "🟢 correct"
        else:
            label = p.name
        cand_imgs.append((Image.open(p).convert("RGB"), label))

    return missing_imgs, cand_imgs

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌟 本地随机 Demo（展示原始图片）")

    with gr.Tabs():
        # CP Tab
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
        with gr.TabItem("CP 兼容性预测"):
            btn = gr.Button("生成 CP 示例")
            succ_gallery = gr.Gallery(label="✅ 成功", columns=10)
            fail_gallery = gr.Gallery(label="❌ 失败", columns=10)
            succ_caps = gr.Textbox(label="成功说明", lines=1)
            fail_caps = gr.Textbox(label="失败说明", lines=1)
            btn.click(fn=cp_demo, outputs=[succ_gallery, succ_caps, fail_gallery, fail_caps])

        # CIR Tab
        with gr.TabItem("CIR 互补项检索"):
            btn2 = gr.Button("生成 CIR 示例")
            cir_partial = gr.Gallery(label="👕 Partial Outfit", columns=4)
            cir_cands = gr.Gallery(label="🎯 Top-10 Candidates", columns=5)
            btn2.click(fn=cir_demo, outputs=[cir_partial, cir_cands])

        # FITB Tab
        with gr.TabItem("FITB 填空任务"):
            btn3 = gr.Button("生成 FITB 示例")
            fitb_partial = gr.Gallery(label="👕 Incomplete Outfit", columns=4)
            fitb_cands = gr.Gallery(label="🧩 Candidates", columns=4)
            btn3.click(fn=fitb_demo, outputs=[fitb_partial, fitb_cands])

# 启动服务，允许本地文件访问
demo.launch(allowed_paths=[str(IMAGES_DIR)])
