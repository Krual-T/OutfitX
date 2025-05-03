from src.trains.trainers import PrecomputeEmbeddingScript

if __name__ == '__main__':
    with PrecomputeEmbeddingScript() as PES:
        PES.run()