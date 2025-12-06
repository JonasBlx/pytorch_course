# PyTorch Course – Deep Learning Project Portfolio

## Overview

This repository curates a complete PyTorch learning path spanning computer vision and NLP. The goal is twofold: reinforce fundamentals (fully-connected networks, CNNs, RNN/GRU/LSTM) and demonstrate hands-on mastery of advanced subjects such as transfer learning, attention-based sequence modeling, and bespoke data tooling. Each script is self-contained, reproducible, and written to serve as a portfolio-ready reference during technical interviews.

## Highlights for recruiters & peers

- **Broad coverage**: MNIST classification, CIFAR-10 fine-tuning, Dogs vs Cats transfer learning, Flickr8k captioning, and DE→EN translation via Seq2Seq and Transformer architectures.
- **Sound engineering practices**: clear separation between data/model/training logic, periodic checkpoints (`save_and_load_cnn.py`, `seq2seq_model.py`), TensorBoard logging (`runs/seq2seq_experiment`), and utility scripts (custom datasets, notebook extraction).
- **Modern tooling**: intensive use of `torchvision`, `torchtext`, `spaCy`, `tensorboard`, and tailored `torch.utils.data.Dataset` implementations with logging and monitoring baked in.
- **Industry readiness**: documented configs, explicit hyperparameters, and training pipelines aligned with PyTorch standards (Adam optimizers, gradient clipping, early stopping).

## Repository map

| File / directory | Description | Skills showcased |
| --- | --- | --- |
| `fully_con_script.py` | Fully-connected nets and CNNs trained on MNIST. | PyTorch basics, DataLoader usage, CNN from scratch. |
| `recurrent_nn.py`, `bidirectional_lstm.py` | RNN, GRU, LSTM, BiLSTM for sequence classification on MNIST. | Sequential modeling, recurrent architectures, variant benchmarking. |
| `save_and_load_cnn.py` | End-to-end checkpoint save/load workflow. | Model serialization, long training recovery. |
| `fine_tune_model.py` | VGG16 fine-tuning on CIFAR-10. | Transfer learning, custom classifier heads. |
| `cats_dogs_training.py` + `customdataset.py` | Custom data pipeline + GoogLeNet transfer for Dogs vs Cats. | Dataset engineering, transforms, controlled train/test splits. |
| `flickr8k_training.py` | Vocabulary building and DataLoader for caption generation. | Vision + NLP, spaCy tokenization, custom padding collate. |
| `seq2seq_model.py` | Attention-based German→English translator with TensorBoard, early stopping, LR scheduling. | NMT, attention, experiment tracking, checkpoints. |
| `transformer.py` | Transformer encoder–decoder implemented from scratch. | Modern architectures, multi-head attention, mask handling. |
| Notebooks (`basic_operation.ipynb`, `dense_and_convolutional.ipynb`) & `extract_code.py` | Pedagogical explorations and conversion utility from notebook to script. | Technical communication, productivity tooling. |
| `runs/` | TensorBoard logs for the Seq2Seq experiment. | Experiment management. |

## Installation

1. **Create a Python 3.10+ environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > NLP scripts lazily download spaCy models (`en_core_web_sm`, `de_core_news_sm`) and NLTK/BLEU resources when first needed, so ensure internet access on first run.

## Expected datasets

- **MNIST & CIFAR-10**: downloaded automatically into `dataset/`.
- **Dogs vs Cats**: place images under `dataset/dogs_cats/train` and generate `labels.csv` with `create_cats_dogs_label.py`.
- **Flickr8k**: store images in `dataset/Flickr8k/images` and captions in `dataset/Flickr8k/captions.txt`.
- **Multi30k**: pulled automatically through `torchtext.datasets.Multi30k` (requires network access).

## Running the flagship scenarios

### 1. MNIST image classification
```bash
python fully_con_script.py
python recurrent_nn.py        # evaluate RNN/GRU/LSTM
python bidirectional_lstm.py  # bidirectional LSTM variant
```
Expected outcome: >98% accuracy on the test split with the CNN and BiLSTM models.

### 2. CIFAR-10 VGG16 fine-tuning
```bash
python fine_tune_model.py
```
Key details: the avg-pooling block is replaced with an identity layer and the classifier head is rebuilt for 10 classes.

### 3. Dogs vs Cats transfer with GoogLeNet
```bash
python cats_dogs_training.py
```
The script leverages `CatsAndDogsDataset` to incorporate custom transforms and a deterministic train/test split.

### 4. Flickr8k data preparation
```bash
python flickr8k_training.py
```
Prints batch shapes (images + padded token sequences) ready to feed a CNN-RNN captioning stack.

### 5. German → English translation
```bash
python seq2seq_model.py
tensorboard --logdir runs/seq2seq_experiment
```
Features: additive attention, gradient clipping, manual LR decay with patience, `german2english.pth` checkpointing, and an inference helper `translate_sentence`.

### 6. Transformer from scratch
```bash
python transformer.py
```
The `main` section illustrates how source/target masks are created and how data flows through the encoder–decoder stack.

## Experiment tracking & best practices

- **TensorBoard**: inspect `runs/seq2seq_experiment` to follow training/validation losses of the translation task.
- **Checkpoints**: `save_and_load_cnn.py` and `seq2seq_model.py` document how to persist model + optimizer state for resumable training.
- **Easy customization**: hyperparameters are grouped near the top of each file and rely on modular building blocks (`Identity`, `CatsAndDogsDataset`, `Vocabulary`, `MyCollate`).
- **Reproducibility mindset**: all scripts define transforms, optimization strategies, and device handling (`cuda` vs `cpu`) to minimize run-to-run drift.

## Suggested roadmap

1. Add lightweight unit checks (e.g., verifying loader shapes) to harden the pipeline.
2. Export the strongest models via TorchScript/ONNX to show deployment readiness.
3. Enrich the README with consolidated metrics once long GPU trainings are complete.

## License

Distributed under the MIT license (`LICENSE`); freely reusable for academic or professional work.
