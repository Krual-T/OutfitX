import wandb
import time
import random
from src.trains.configs.base_train_config import WANDB_KEY
from src.project_settings.info import PROJECT_NAME
wandb.login(key=WANDB_KEY)
wandb.init(
    project=PROJECT_NAME,
    name = "test",
)

num_epochs = 100
batches_per_epoch = 20

for epoch in range(num_epochs):
    for batch in range(batches_per_epoch):

        # 假装训练...
        train_loss = random.uniform(0.2, 0.5)
        train_acc = random.uniform(0.6, 0.9)

        # ✅ 记录 batch 级别指标
        wandb.log({
            "batch": batch+batches_per_epoch*epoch,
            "train/batch/loss": train_loss,
            "train/batch/accuracy": train_acc,
        })

        time.sleep(0.1)  # 模拟训练耗时

    # 假装验证...
    val_loss = random.uniform(0.2, 0.4)
    val_auc = random.uniform(0.75, 0.95)
    val_f1 = random.uniform(0.6, 0.85)

    # ✅ 记录 epoch 级别指标（step 用 global_step 保持对齐）
    wandb.log({
        "epoch": epoch,
        "train/epoch/loss": val_loss,
        "train/epoch/AUC": val_auc,
        "train/epoch/F1": val_f1,
    })
