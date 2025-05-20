import click
import torch.multiprocessing as mp
@click.command()
@click.option('--mode', type=click.Choice(['train-valid', 'test']), default='train-valid', help='运行模式')
def run(mode):
    print(f"🚀 当前运行模式：{mode}")
    from src.trains.trainers.complementary_item_retrieval_trainer import ComplementaryItemRetrievalTrainer as CIRTrainer
    with CIRTrainer(run_mode=mode) as cir_trainer:
        cir_trainer.run()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run()
