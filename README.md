# Sentiment Analysis Through Video Captioning

This project builds an end-to-end deep learning pipeline that generates captions for short video clips and classifies the sentiment of those captions using transformer-based models. It combines computer vision, natural language processing, and zero-shot classification to understand video content and extract emotional context.

---

## 🎥 Dataset

- MSVD Video Clips: [MSVD Video Dataset](https://www.kaggle.com/datasets/sarthakjain004/msvd-clips)  
- MSVD Caption Corpus: [MSVD Video Corpus](https://www.kaggle.com/datasets/vtrnanh/msvd-dataset-corpus)  
- Test Dataset: [Custom Test Dataset](https://www.kaggle.com/datasets/csecemk128/testing-dataset)

The test dataset includes real-world videos collected from social media accounts of:
- a food vlogger (stored in /Food)
- an animal lover (stored in /Animal)

---

## 🔄 Pipeline Workflow

1. Frame Extraction using OpenCV  
2. Feature Extraction using InceptionV3 (CNN)  
3. Caption Generation using an LSTM-based decoder  
4. Sentiment and Topic Classification using DeBERTa (transformer)  
5. Evaluation using BLEU Score, Accuracy, Precision, Recall, and F1-Score  

---

## ⚙️ Model Configuration

- Encoder: InceptionV3 (pre-trained CNN for frame features)  
- Decoder: LSTM-based sequential caption generator  
- Tokenization: Keras Tokenizer with padding and OOV handling  
- Sentiment Classifier: HuggingFace DeBERTa (zero-shot classification)  
- Loss Function: Categorical Crossentropy  
- Optimizer: Adam  
- Evaluation Metrics: BLEU-1 to BLEU-4, Accuracy, F1-Score, Precision, Recall

---

## 📊 Performance

| Metric                       | Score  |
|------------------------------|--------|
| BLEU-4 Score (Training)      | 77.85% |
| Real-World Accuracy (Food)   | 86%    |
| Real-World Accuracy (Animal) | 93%    |
| F1-Score (Food Test)         | 0.92   |
| F1-Score (Animal Test)       | 0.96   |

> 📌 BLEU scores were computed on the training set. Real-world accuracy was tested on social video clips.

---

## 🖼️ Visual Results

| Dataset | Confusion Matrix | Sample Output |
|---------|------------------|----------------|
| Animal  | [View](visuals/animal_dataset_confusion_matrix.png) | [View](visuals/animal_dataset_sample_predicted_output.png) |
| Food    | [View](visuals/food_dataset_confusion_matrix.png)   | [View](visuals/food_dataset_sample_predicted_output.png)   |

📉 [Loss Graph](visuals/loss_graph.png)

---

## 📄 Output Files

- [Food Test Results (CSV)](outputs/food_test_results.csv)  
- [Animal Test Results (CSV)](outputs/animal_test_results.csv)

---

## 📁 Directory Structure
video-captioning-sentiment-analysis/
├── data/
│ ├── corpus/
│ │ └── annotations.txt
│ ├── dataset.txt
│ ├── tokenizer.pkl
│ └── video_features.pkl
├── models/
│ └── video_caption_model.h5
├── notebook/
│ └── video_captioning_with_sentiment.ipynb
├── outputs/
│ ├── animal_test_results.csv
│ └── food_test_results.csv
├── visuals/
│ ├── animal_dataset_confusion_matrix.png
│ ├── animal_dataset_sample_predicted_output.png
│ ├── food_dataset_confusion_matrix.png
│ ├── food_dataset_sample_predicted_output.png
│ └── loss_graph.png
└── README.md

