import wandb
import time

from src.trains.configs.base_train_config import WANDB_KEY
from src.project_settings.info import PROJECT_NAME

wandb.login(key=WANDB_KEY)
wandb.init(
    project=PROJECT_NAME,
    name="linear_metrics_test",
)

num_epochs = 4
batches_per_epoch = 5

# 🎯 统一斜率 k，但每个指标用不同的起始值
k = 0.01

# 定义每个指标的 intercept（初始值）
intercepts = {
    'train/batch/loss': 0.5,
    'train/batch/accuracy': 0.5,
    'valid/batch/loss': 0.6,
    'valid/batch/accuracy': 0.55,
    'train/epoch/loss': 0.4,
    'train/epoch/AUC': 0.7,
    'train/epoch/F1': 0.6,
    'valid/epoch/loss': 0.45,
    'valid/epoch/AUC': 0.75,
    'valid/epoch/F1': 0.65
}

for epoch in range(num_epochs):
    for batch in range(batches_per_epoch):
        global_batch = epoch * batches_per_epoch + batch

        # 📉 线性生成 batch-level 训练指标
        wandb.log({
            "batch": global_batch,
            "train/batch/loss": max(0.01, intercepts['train/batch/loss'] - k * global_batch),
            "train/batch/accuracy": min(1.0, intercepts['train/batch/accuracy'] + k * global_batch),
        })

    # 📊 epoch-level 训练指标
    wandb.log({
        "epoch": epoch,
        "train/epoch/loss": max(0.01, intercepts['train/epoch/loss'] - k * epoch),
        "train/epoch/AUC": min(1.0, intercepts['train/epoch/AUC'] + k * epoch),
        "train/epoch/F1": min(1.0, intercepts['train/epoch/F1'] + k * epoch),
    })

    for batch in range(batches_per_epoch):
        global_batch = epoch * batches_per_epoch + batch

        # 📉 线性生成 batch-level 验证指标
        wandb.log({
            "batch": global_batch,
            "valid/batch/loss": max(0.01, intercepts['valid/batch/loss'] - k * global_batch),
            "valid/batch/accuracy": min(1.0, intercepts['valid/batch/accuracy'] + k * global_batch),
        })

    # 📊 epoch-level 验证指标
    wandb.log({
        "epoch": epoch,
        "valid/epoch/loss": max(0.01, intercepts['valid/epoch/loss'] - k * epoch),
        "valid/epoch/AUC": min(1.0, intercepts['valid/epoch/AUC'] + k * epoch),
        "valid/epoch/F1": min(1.0, intercepts['valid/epoch/F1'] + k * epoch),
    })
