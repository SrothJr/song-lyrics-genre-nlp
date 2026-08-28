# Music Genre Classification from Song Lyrics (CSE440)

## Overview
This project tackles the natural language processing (NLP) challenge of classifying songs into one of six primary genres (Pop, Rock, Hip Hop, Indie, Heavy Metal, Dance) based solely on their lyrical content. 

The primary technical challenge of this project is severe class imbalance. Genres such as Rock and Pop heavily dominate the dataset, whereas Heavy Metal and Dance represent a small fraction of the data. The methodology focuses on progressing from classical baselines to advanced transformer ensembles, with a specific emphasis on cost-sensitive learning to mitigate majority-class bias.

## Dataset
* **Input:** Raw song lyrics (English).
* **Target Classes:** Pop, Rock, Hip Hop, Indie, Heavy Metal, Dance.
* **Characteristics:** Highly imbalanced distribution requiring specialized loss functions and evaluation metrics (Macro F1 over pure Accuracy).

## Methodology & Architectural Decisions

### 1. Classical Machine Learning (Baselines)
* **Models:** Multinomial Naive Bayes, Random Forest.
* **Reasoning:** Established a baseline for text classification using TF-IDF vectorization. 
* **Outcome:** Achieved ~55% accuracy but failed to capture semantic context. Minority class recall was exceptionally poor (near 0% for Dance and Heavy Metal).

### 2. Recurrent Neural Networks (RNNs)
* **Models:** SimpleRNN, GRU, LSTM, and Bi-directional variants.
* **Reasoning:** Transitioned to sequential models to capture the temporal dependencies and context of lyrics.
* **Implementation:** Developed natively in PyTorch 2.8. PyTorch was selected over TensorFlow to ensure stable hardware acceleration on Blackwell-architecture GPUs (RTX 5090) using CUDA 12.8.
* **Configurations:**
  * Embed=256, Hidden=128, Dropout=0.5 (Best Bi-LSTM)
  * Embed=128, Hidden=256, Dropout=0.4 (Best GRU)
* **Outcome:** The Bi-LSTM achieved 44% accuracy. While better than baseline recall, the RNNs struggled with the vanishing gradient problem over long lyric sequences, confirming the necessity for attention mechanisms.

### 3. Transformer Architectures
To address the long-term dependency issues of RNNs, the project utilized pre-trained contextual embeddings via the Hugging Face Transformers library. 

#### Model A: Standard BERT-Base
* **Setup:** Standard cross-entropy loss.
* **Result:** 63% Accuracy, 0.53 Macro F1. 
* **Analysis:** High raw accuracy, but heavily biased toward predicting Rock and Pop.

#### Model B: Weighted BERT-Base (Cost-Sensitive Learning)
* **Decision:** Implemented a custom `ClassWeightsTrainer` applying inverse frequency weights to the CrossEntropyLoss function.
* **Reasoning:** Penalizes the model for misclassifying minority classes to force representation learning for Dance and Heavy Metal.
* **Result:** Maintained 63% Accuracy but improved Macro F1 to 0.54. Heavy Metal recall improved from 32% to 42%; Dance improved from 19% to 30%.

#### Model C: Weighted RoBERTa-Base
* **Decision:** Applied the same custom `ClassWeightsTrainer` to RoBERTa.
* **Result:** 55% Accuracy, 0.50 Macro F1.
* **Analysis:** RoBERTa over-corrected due to the class weights. It became hyper-sensitive to minority classes (Heavy Metal recall jumped to 62%), but suffered a drop in precision for majority classes. 

#### Excluded Model: DeBERTa-v3
* **Decision:** Dropped from the final ensemble.
* **Reasoning:** DeBERTa's disentangled attention mechanism proved numerically unstable (gradient explosion/mode collapse) on this specific imbalanced distribution, even when utilizing BF16 precision and aggressive linear learning rate warmups (10% ratio).

### 4. Final Architecture: 2-Model Soft Voting Ensemble
* **Decision:** An ensemble averaging the raw Softmax probabilities of Weighted BERT and Weighted RoBERTa.
* **Reasoning:** Fuses the stability of BERT with the minority-class sensitivity of RoBERTa. BERT anchors the overall accuracy, while RoBERTa's over-correction helps identify rare genres that BERT misses.

## Transformer Training Configurations
All transformers were fine-tuned using the following hyperparameters, optimized for the RTX 5090:
* **Precision:** FP16 (Mixed Precision) for optimal memory bandwidth.
* **Batch Size:** 64 (Effective). 
* **Learning Rate:** 2e-5 with AdamW optimizer.
* **Weight Decay:** 0.01 to prevent overfitting.
* **Epochs:** 20 (Maximum) with `EarlyStoppingCallback`.
* **Early Stopping:** Patience of 3 epochs, monitoring `f1_weighted` to ensure checkpoint selection prioritized class balance over raw validation loss.

## Final Results

| Model | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :--- |
| Random Forest (Baseline) | 0.55 | 0.40 | 0.53 |
| Bi-LSTM | 0.44 | 0.37 | 0.46 |
| Weighted BERT | 0.63 | 0.54 | 0.62 |
| Weighted RoBERTa | 0.55 | 0.50 | 0.56 |
| **Final Ensemble (BERT + RoBERTa)** | **0.63** | **0.55** | **0.63** |

*Note: The Final Ensemble achieved the highest Macro F1 score of the study (0.55), proving the effectiveness of combining a balanced transformer with a highly-sensitive transformer for imbalanced text classification.*

## Deployment
The final weights and tokenizers for the BERT and RoBERTa models have been serialized and exported to the `/deployment_models/` directory for integration into the production inference pipeline.
