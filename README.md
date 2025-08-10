# ResNet-LSTM Multimodal Captioning Framework for Automated Scene Understanding

##  Abstract

This repository implements a **vision–language model** for **end-to-end automated scene understanding** via image captioning. A **ResNet50 convolutional backbone** extracts **2048-dimensional global feature embeddings** from input images, which are projected into a semantic subspace and consumed by an **LSTM decoder** for sequential natural language generation. Training is conducted on **30K+ image–caption pairs** with `<start>` and `<end>` sequence markers, tokenization, and dynamic sequence padding for computational efficiency. The proposed system achieves **BLEU-1: 0.71** and **BLEU-2: 0.55**, outperforming a VGG16-based baseline by **+12% BLEU-1**. Optimizations include **cosine learning rate scheduling**, **dropout regularization (p=0.4)**, and **gradient clipping**, enabling both convergence stability and generalization.

---

##  Architecturural Overview

1. **Visual Encoder (ResNet50)**

   * Pretrained on ImageNet; final FC layers removed.
   * Outputs **2048-D pooled feature vectors** capturing high-level semantics.
   * Features cached in `featuresNew.pkl` to decouple extraction from caption training.

2. **Language Decoder (LSTM)**

   * Embedding layer maps token indices to dense 256-D word vectors.
   * LSTM hidden state initialized with projected ResNet embeddings.
   * Dropout (p=0.4) for regularization; Dense softmax for vocabulary logits.

3. **Training Strategy**

   * Loss: Categorical Cross-Entropy over tokens.
   * Optimizer: Adam with cosine learning rate schedule.
   * Early stopping on validation BLEU score.

<img width="3000" height="1200" alt="resnet_lstm_pipeline" src="https://github.com/user-attachments/assets/cc6bd403-1e5c-4b75-b243-acdc9616e1f7" />

---

##  Performance Metrics

| Model             | BLEU-1   | BLEU-2   | Δ BLEU-1 vs Baseline | Inference Time |
| ----------------- | -------- | -------- | -------------------- | -------------- |
| **ResNet50+LSTM** | **0.71** | **0.55** | **+12%**             | **0.78 s**     |
| VGG16+LSTM        | 0.63     | 0.47     | —                    | 1.10 s         |

<img width="2000" height="1200" alt="bleu_score_comparison" src="https://github.com/user-attachments/assets/c96b971b-67f7-4aae-b540-8fe5afb9ee40" />

---

##  Technical Insights

* **Semantic Richness:** ResNet50 embeddings offer higher-level semantic separation, improving decoder grounding.
* **Structural Accuracy:** BLEU-2 gains indicate better phrase-level syntax and object–relation modeling.
* **Regularization Impact:** Dropout + cosine LR scheduling reduced overfitting, improving generalization on unseen captions.
* **Latency:** Precomputed features enable sub-second captioning on CPU without GPU acceleration.

<img width="2000" height="1200" alt="inference_latency_comparison" src="https://github.com/user-attachments/assets/a0ec24b1-eb5f-42f6-9e03-300752e5e07a" />

---

##  Reproducibility & Training Details

**Dataset Preparation:**

* Image–caption pairs (\~30K+) with one or more captions per image.
* Captions lowercased, tokenized, and wrapped with `<start>` and `<end>` tokens.
* Vocabulary constructed from training captions; rare words removed if below frequency threshold.
* Images resized to **224×224** before feature extraction with ResNet50.

**Feature Extraction:**

```python
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# Model without top layers
model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
```

* Saved as `featuresNew.pkl` for reuse during training/inference.

**Model Training:**

* **Epochs:** 40 (best checkpoint at e40).
* **Batch Size:** chosen to fit GPU memory (based on hardware).
* **Optimizer:** Adam with cosine decay schedule.
* **Regularization:** Dropout p=0.4 on LSTM and embedding layers.
* **Evaluation:** BLEU-1, BLEU-2.

**Files:**

* `ResNet_Model_e40.h5` – Trained model weights.
* `featuresNew.pkl` – Precomputed ResNet50 image features.
* `captions.txt` – Training captions text file.

---

##  Deployment

**Run locally:**

```bash
pip install -r requirements.txt
python app.py
```

* Launches **Gradio web interface**.
* Upload an image → Get caption in **<0.8s**.

---

##  Potential Applications

* **Cultural Heritage Annotation**
* **Medical Imaging Report Drafting**
* **Autonomous Navigation Scene Understanding**

---

##  License

Apache 2.0 License © 2025 SupeRNovA15-ai

---


