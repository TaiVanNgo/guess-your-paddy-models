# 🌾 Guess Your Paddy: Intelligent Paddy Plant Diagnosis System

**Guess Your Paddy** is a machine learning-powered system that trains deep learning models to classify diseases, identify rice varieties, and predict the growth stage (age) of paddy plants from image inputs.

This repository focuses on the model development pipeline, including training, evaluation, and exporting the final models used in the **Guess Your Paddy** web application.

Whether you're building AI-powered crop diagnostic tools or benchmarking CNN models for agricultural tasks, this project demonstrates an end-to-end image-based learning pipeline tailored for real-world rice farming scenarios.

## 🚀 Key Features

✅ **Disease Classification**: Detect 9+ common rice diseases from field images (blast, brown spot, tungro, etc.)

🌾 **Variety Identification**: Classify the variety of the paddy among 10 types using advanced CNNs

📅 **Age Estimation**: Predict the age of the crop in days to support timely interventions

💻 **User Interface**: Integrated with a simple web-based UI for real-time inference (see below)

📈 **Robust Models**: Built with fine-tuned CNN architectures (VGG, EfficientNet, ResNet50V2)

## Project Structure

```.
.
├── images/                    # Project images & visualizations
├── inputs/                    # Dataset directory
│   ├── meta_train.csv         # Training dataset metadata
│   ├── train_images/          # Training images
│   └── test_images/           # Test images
├── models/                    # Saved model files
│   └── history/               # Training history files
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── model_training.py      # Model training utilities
│   ├── preprocessing.py       # Data preprocessing utilities
│   └── visualizations.py      # Visualization utilities
├── .gitignore                 # Git ignore file
├── 00_main_notebook_eda.ipynb # Exploratory data analysis
├── 01_task1_disease_classification.ipynb # Disease classification notebook
├── 02_task2_variety_classification.ipynb # Variety classification notebook
├── 03_task3_age_regression.ipynb # Age regression notebook
├── 04_final_prediction.ipynb  # Final prediction notebook
├── README.md                  # Project documentation
├── requirements_tf2.10.txt    # Requirements for TensorFlow ≤2.12
└── requirements.txt           # Requirements for TensorFlow ≥2.13
```

## Model Overview

| Task                   | Model Used            | File Path                              |
| ---------------------- | --------------------- | -------------------------------------- |
| Disease Classification | Fine-tuned Mini VGG16 | `./models/vgg_best.keras`              |
| Variety Classification | EfficientNetB0        | `./models/efficientnet_final.keras`    |
| Age Regression         | ResNet50V2            | `./models/resnet50v2_best_model.keras` |

## 📁 OneDrive Access – Models & Dataset

Due to GitHub file size limitations, the full dataset and final trained models are hosted on OneDrive:

- 🔗 **[Download Models (.keras)](https://rmiteduau-my.sharepoint.com/:u:/g/personal/s3974892_rmit_edu_vn/EWxSvxikHoZIsf-Tgo5IujkBHE9WyciexSFyMs703t-PSw?e=SkljHF)**  
- 🔗 **[Download Dataset (train/test images + meta CSV)](https://your-onedrive-link/dataset)**

> ⚠️ Ensure you download and place the files into the correct folder paths as shown in the structure above (`inputs/` and `models/`).

---

## Setup Instruction

1. Clone the repository:  

```bash
git clone https://github.com/TaiVanNgo/COSC2753-machine-learning-assignment-2
cd paddy-disease-classification
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv source venv/bin/activate 
# On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

Choose the appropriate requirements file based on your environment:

- For newer environments (TensorFlow ≥ 2.13):

```bash
pip install -r requirements.txt
```

- For environments using TensorFlow ≤ 2.12:

```bash
pip install -r requirements_tf2.10.txt
```

4. Download the dataset and place it in the `inputs/` directory.

## 📂 Dataset Contributors

The primary dataset was sourced from the [Paddy Doctor Kaggle Competition](https://www.kaggle.com/competitions/paddy-disease-classification/overview).

Additional datasets for semi-supervised training:

- [Rice Disease Dataset (Mendeley)](https://data.mendeley.com/datasets/fwcj7stb8r/1)
- [Rice Disease Image Dataset (Kaggle)](https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset)
- [HumayAI Rice Diseases (Roboflow)](https://universe.roboflow.com/humayai-kacn9/humay-ai-rice-diseases/dataset/4/download)

## 🌐 Live Demo

Explore the Paddy Doctor platform via our deployed web app: [guessyourpaddy.site](http://guessyourpaddy.site/)

🔗 Website: Guess Your Paddy
Github Guess Your Paddy Website: [Github](https://github.com/phatgg221/guess-your-paddy)

## 🤝 Authors

- Ngo Van Tai ([Github](https://github.com/TaiVanNgo))
- Duong Minh Tri ([Github](https://github.com/TriDuong070803))
- Huynh Thai Duong ([Github](https://github.com/TDuong04))
- Huynh Tan Phat ([Github](https://github.com/phatgg221))