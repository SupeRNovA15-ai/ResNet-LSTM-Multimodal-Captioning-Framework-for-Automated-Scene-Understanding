# ResNet-LSTM-Multimodal-Captioning-Framework-for-Automated-Scene-Understanding
End-to-end image captioning using ResNet50 features and an LSTM decoder. Trained on 30K+ pairs, achieves BLEU-1 of 0.71 (+12% vs VGG16). Uses cosine scheduling and dropout for robust training. Real-time captions via Gradio in &lt;0.8s. Applications include cultural,medical, and navigation domains. 
Model Architecture
<p align="center"> <img src="docs/architecture.png" width="80%"> </p>
Pipeline:

ResNet50 Encoder → Extracts 2048-D features from input images.

LSTM Decoder → Generates captions word-by-word from image embeddings.

Training Setup → Cosine learning rate scheduler + dropout (0.4).
