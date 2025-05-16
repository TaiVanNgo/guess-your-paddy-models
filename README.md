# COSC2753 Assignment 2: Paddy Plant Disease Classification

This project uses machine learning techniques to classify paddy plant diseases from images. The system employs various models including VGG16, custom CNN architectures, and hybrid models to accurately identify different disease categories in rice plants.

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

4.Download the dataset and place it in the `inputs` directory or use the provided dataset.
