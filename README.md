# 🌿 Plant Disease Detector — Leaf Disease Classification

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-green?logo=streamlit)](https://hirushan-n-cvpartb.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/DeepLearning-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Made With ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)](#)

---

## 📌 Project Overview

This is an end-to-end deep learning pipeline to detect **leaf diseases** using image classification. It is developed as the **Part B: Individual Coursework** for the *Computer Vision* module at NIBM – Sri Lanka.

Built using **TensorFlow, MobileNetV2**, and deployed using **Streamlit**, the app allows users to upload an image of a leaf and instantly receive a prediction with class-wise probabilities.

---

## 🎯 Use Case

🌾 **Problem:** crops are prone to several leaf diseases like early blight, yellow curl virus, and bacterial spots, which reduce crop yield drastically.

✅ **Solution:** A deep learning-based tool that:
- Takes a leaf image as input
- Predicts the disease from 10+ classes
- Suggests the confidence and alert on low certainty

---

## 🧠 Models Used

### ✅ Model A — Baseline CNN
- Built from scratch using Conv2D layers
- Low accuracy: **31%**
- Used as a benchmark

### ✅ Model B — MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned on dataset
- Final accuracy: **92%**
- Used for deployment

---

## 🗂️ Dataset

- 📦 Source: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/emmarex/plantdisease)
- Classes used:
  - Tomato_Bacterial_spot
  - Tomato_Early_blight
  - Tomato_Late_blight
  - Tomato_Leaf_Mold
  - Tomato_Septoria_leaf_spot
  - Tomato_Spider_mites_Two_spotted_spider_mite
  - Tomato__Target_Spot
  - Tomato__Tomato_mosaic_virus
  - Tomato__Tomato_YellowLeaf__Curl_Virus
  - Tomato_healthy
- Preprocessing:
  - Resize to 224×224
  - Normalization
  - Augmentation (flip, rotate, brightness)

---

## 🏗️ Project Structure

```
cv_partb/
│
├── data_raw/                # Raw PlantVillage images
├── data/                    # Processed TensorFlow datasets
├── outputs/
│   ├── checkpoints/         # Trained .keras models
│   ├── metrics/             # Evaluation JSONs
│   └── figures/             # Confusion matrices
│
├── streamlit_app.py         # Streamlit web application
├── 01_explore_data.ipynb    # EDA notebook
├── 02_modelA_train.ipynb    # Model A training
├── 03_modelB_train.ipynb    # Model B training
├── 04_eval_compare.ipynb    # Final evaluations
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 📈 Evaluation Results

| Metric     | Model A (CNN) | Model B (MobileNetV2) |
|------------|----------------|------------------------|
| Accuracy   | 31%            | **92%**                |
| Precision  | 19%            | **91%**                |
| Recall     | 31%            | **92%**                |
| F1-Score   | 20%            | **91%**                |

✅ Model B was significantly better and is used for deployment.

---

## 💻 Local Installation

### ✅ 1. Clone the Repository

```bash
git clone htthttps://github.com/Hirushan-N/cv_partb.git
cd cv_partb
```

### ✅ 2. Create & Activate Conda Environment

```bash
conda create -n cv_partb python=3.10 -y
conda activate cv_partb
```

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 4. Run Locally

```bash
streamlit run streamlit_app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🌐 Live Demo

🚀 Try it online here:  
🔗 **[https://hirushan-n-cvpartb.streamlit.app/](https://hirushan-n-cvpartb.streamlit.app/)**

---

## 🧪 Streamlit App Features

- Upload leaf image
- Real-time classification into 10 disease classes
- Shows prediction probability for each class
- Displays confidence score
- Warns when prediction confidence is low

---

---

## 💡 Future Enhancements

- Add Grad-CAM visual explanations
- Extend to more crops and diseases
- Build responsive mobile PWA
- Compress model for mobile inference (TensorFlow Lite)

---

## 📚 References

- [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)
- [TensorFlow Official Docs](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [MobileNetV2 Paper (2018)](https://arxiv.org/abs/1801.04381)

---

## 🧑‍💻 Author

**Mabotuwana Vithanage Nadeesh Hirushan**  
Student at NIBM - Galle, Sri Lanka  
📧 nadeeshhirushan@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/nadeeshhirushan)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
