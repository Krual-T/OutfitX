# OutfitX: Fashion Outfit Recommendation System
This is my undergraduate graduation project; I'm keeping it as a memento.
A cross-modal fusion model for fashion outfit recommendation, leveraging CNN and Transformer architectures to handle both image and text data for tasks like compatibility prediction, complementary item retrieval, and fill-in-the-blank challenges. This work, named **OutfitX**, draws inspiration from the foundational research presented in *[OutfitTransformer: Learning Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812v2)* and references the open-source reproduction implementation from [outfit-transformer](https://github.com/owj0421/outfit-transformer), which enhances performance and usability for real-world deployment.

## Key Features

* **Multi-task Support**: Handles Compatibility Prediction (CP), Complementary Item Retrieval (CIR), and Fill-in-the-Blank (FITB) tasks.

* **Cross-Modal Fusion**: Combines image and text embeddings using concatenation or mean aggregation strategies, a core design principle of OutfitX.

* **Efficient Encoding**: Utilizes CLIP-based encoders for both image and text processing, optimized to align with OutfitX’s architecture.

* **Scalable Architecture**: Transformer-based global outfit encoder (a key component of OutfitX) for learning contextual relationships between fashion items.

## Installation

1. Clone the repository:

```
git clone \<repository-url>

cd GraduationDesign
```

1. Create and activate the conda environment (configured for OutfitX):

```
conda env create -f environment.yml

conda activate GraduationDesign
```

## Usage

### 1. Precompute Embeddings

Generate embeddings for fashion items to speed up OutfitX’s training process:

```
python -m src.trains.trainers.precompute\_embedding\_script
```

### 2. Train OutfitX

* For Compatibility Prediction (CP) with OutfitX:

```
python -m src.trains.trainers.compatibility\_prediction\_trainer
```

* For Complementary Item Retrieval (CIR) with OutfitX:

```
python -m src.trains.trainers.complementary\_item\_retrieval\_trainer
```

### 3. Run OutfitX Demo

Launch the Gradio interface to interact with the trained OutfitX model:

```
python -m src.demo.test
```

## Project Structure

* `src/models`: Core implementation of **OutfitX** (including its encoders, processors, and Transformer-based modules).

* `src/trains`: Training scripts and configurations tailored for OutfitX’s multi-task learning.

* `src/demo`: Gradio interfaces for visualizing OutfitX’s recommendation outputs.

* `datasets/polyvore`: Fashion dataset (with images and metadata) used to train and evaluate OutfitX.

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file in the repository root for full details. Under the MIT License, you are permitted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the software.

If you use this repository or the OutfitX model in your research or application, please acknowledge the original inspirations: the *OutfitTransformer* paper ([OutfitTransformer: Learning Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812v2)) and its open-source reproduction ([outfit-transformer](https://github.com/owj0421/outfit-transformer)), as well as this OutfitX implementation.

