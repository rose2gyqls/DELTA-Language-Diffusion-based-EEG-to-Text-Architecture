# DELTA: Language Diffusion-based EEG-to-Text Architecture

Official implementation of  **DELTA** , a novel framework for decoding EEG signals into text. DELTA shifts the paradigm from traditional direct translation to  **robust restoration** , addressing the challenges of signal instability and error accumulation in autoregressive models.

## 🚀 Key Features

* **Paradigm Shift** : Moves from unstable direct mapping to a restoration-based approach, significantly improving decoding reliability.
* **RVQ Tokenizer** : Converts noisy, continuous EEG waveforms into stable discrete representations using Residual Vector Quantization (RVQ).
* **Language Diffusion Model** : Utilizes **LLaDA** (Large Language Diffusion Model) to generate text through iterative denoising, preventing sequential error propagation.
* **SOTA Performance** : Achieves state-of-the-art results on the ZuCo dataset with a  **BLEU-1 score of 21.9** , outperforming conventional AR models (BART, Pegasus, T5).

## 🏗️ Architecture

The DELTA framework consists of two main stages:

1. **Stage 1: RVQ-based EEG Tokenizer**
   * Input: 105 channels × 8 frequency bands (840-dimensional vector).
   * Process: Hierarchical quantization via 12 codebooks ($M=12$) with a codebook size of 512 ($K=512$).
   * Output: A sequence of robust discrete tokens.
2. **Stage 2: LLaDA-based Text Generation**
   * **EEG Pre-training** : Learning the distribution of EEG token space through random masking and restoration.
   * **Supervised Fine-Tuning (SFT)** : Conditioning the diffusion model on EEG tokens to generate corresponding text.
   * **Efficient Training** : Implementation using **QLoRA** for fine-tuning the LLaDA-8B model.

## 🛠️ Installation & Data Preparation

### Requirements

* Python 3.8+
* PyTorch 2.0+
* Transformers
* PEFT (for QLoRA)
* Accelerate

### Dataset

The project uses the **ZuCo 1.0 & 2.0** datasets.

* EEG signals recorded during natural reading tasks.
* Preprocessed to 105 active channels across 8 frequency bands.
* Data is split into 80% Train, 10% Val, and 10% Test (non-overlapping sentences).

## 📖 Usage

### 1. Training (Decoding)

To train the decoding model, use `train_decoding.py`. The training involves a two-step process: freezing BART parameters initially and then updating the full model.

```
python train_decoding.py \
    --task_name task1 \
    --model_name BrainTranslator \
    --batch_size 32 \
    --learning_rate_step1 1e-4 \
    --learning_rate_step2 5e-5

```

### 2. Inference & Evaluation

To generate text from EEG signals and calculate metrics (BLEU, ROUGE, WER), run the generation script:

```
python generate/run_eeg_generation.py

```

*Key Sampling Parameters:*

* `GEN_STEPS`: 64
* `TEMP`: 0.3
* `REMASKING`: "low_confidence"

## 📊 Experimental Results

Performance comparison on the ZuCo dataset (Word-level granularity):

| Method                 | BLEU-1         | ROUGE-1        | BERT-Score      |
| ---------------------- | -------------- | -------------- | --------------- |
| DeWave                 | 12.9           | 10.2           | 0.702           |
| BART                   | 13.7           | 11.9           | 0.728           |
| T5                     | 16.7           | 11.9           | 0.735           |
| **DELTA (Ours)** | **21.9** | **17.2** | **0.742** |
