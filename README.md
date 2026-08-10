# Titanic Survival Prediction (Machine Learning Project)

A machine learning classification pipeline built in Python to predict passenger survival from the classic Kaggle Titanic dataset using a Decision Tree Classifier.

---

## Project Overview

This project applies end-to-end data preprocessing, feature engineering, and a supervised machine learning algorithm to historical passenger records. The objective is to determine the core predictive patterns that influenced survival rates during the Titanic shipwreck.

## Repository Structure

- **`titanic_model.py`** — The primary Python script handling data ingestion, preprocessing, model training, and performance evaluation.
- **`generate_diagrams.py`** — A supplementary script to reproduce the model's logic rules and confusion matrix as high-resolution images.
- **`titanic.csv`** — The local Kaggle dataset used for training and testing.
- **`decision_tree.png`** / **`confusion_matrix.png`** — Exported model visualizations.
- **`Detailed_Machine_Learning_Project_Report.docx`** — The technical report documenting the methodology, results, and evaluation metrics.

## Requirements & Dependencies

To run this project locally, ensure you have Python installed along with the following libraries:

```bash
pip install pandas scikit-learn matplotlib
```

## Running the Code

1. Clone or download this repository.
2. Open the directory in your code editor.
3. To train the model and view the accuracy score in the terminal, run:

```bash
python titanic_model.py
```

4. To locally regenerate the visualization charts, run:

```bash
python generate_diagrams.py
```

## Methodology & Results

- **Data Preprocessing:** Dropped unstructured/unpredictable columns (`PassengerId`, `Name`, `Ticket`, `Cabin`), imputed missing ages with the dataset median, and applied one-hot encoding to categorical string variables (`Sex`, `Embarked`).
- **Model:** Decision Tree Classifier (`max_depth=3`).
- **Performance:** Achieved an evaluation accuracy score of **79.89%** on the testing split, identifying gender and passenger class as the primary split indicators for survival logic.

## Model Visualizations

**Decision Tree Logic:**
!decision_tree.png

**Confusion Matrix:**
!confusion_matrix.png
