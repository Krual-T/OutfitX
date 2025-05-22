import click
import torch
# 在这里禁用 cuDNN
torch.backends.cudnn.enabled = False
@click.command()
@click.option('--mode', type=click.Choice(['train-valid', 'test']), default='train-valid', help='运行模式')
def run(mode):
    print(f"🚀 当前运行模式：{mode}")
    from src.trains.trainers.original_cp_trainer import OriginalCompatibilityPredictionTrainer
    with OriginalCompatibilityPredictionTrainer(run_mode=mode) as cp_trainer:
        cp_trainer.run()

if __name__ == '__main__':
    run()
