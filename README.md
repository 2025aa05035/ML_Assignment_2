# ML Assignment 2 – Classification Models with Streamlit Deployment

## Problem Statement

The objective of this project is to design, implement, and deploy multiple machine learning classification models on a real-world dataset. The project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, model training, performance evaluation using multiple metrics, and deployment of an interactive web application using Streamlit Community Cloud.

The system allows users to upload test data, select a classification model, and visualize model performance through evaluation metrics and confusion matrices.

---

## Dataset Description

**Dataset Used:** Breast Cancer Wisconsin Dataset (from `scikit-learn`)

- **Type:** Binary Classification  
- **Number of Instances:** 569  
- **Number of Features:** 30 numerical features  
- **Target Variable:**  
  - `0` → Malignant  
  - `1` → Benign  

The dataset contains measurements computed from digitized images of breast mass biopsies. It satisfies the assignment requirement of having more than 500 instances and more than 12 features.

The dataset is programmatically loaded from `scikit-learn` and saved as a CSV file (`breast_cancer.csv`) to ensure reproducibility.

---

## Models Used and Evaluation Metrics

The following **six classification models** were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Model Performance Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | High | High | High | High | High | High |
| Decision Tree | Moderate | Moderate | Moderate | Moderate | Moderate | Moderate |
| KNN | Good | Good | Good | Good | Good | Good |
| Naive Bayes | Good | Good | Good | Good | Good | Good |
| Random Forest (Ensemble) | Very High | Very High | Very High | Very High | Very High | Very High |
| XGBoost (Ensemble) | Best | Best | Best | Best | Best | Best |

> Exact metric values are displayed in the terminal output and in the Streamlit application.

---

## Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performs well on linearly separable data and provides a strong baseline |
| Decision Tree | Simple and interpretable but prone to overfitting |
| KNN | Performance depends on feature scaling and choice of K |
| Naive Bayes | Fast and efficient but assumes feature independence |
| Random Forest (Ensemble) | Robust and achieves high accuracy |
| XGBoost (Ensemble) | Best overall performance due to boosting and optimization |

---

## Streamlit Web Application Features

The Streamlit application provides the following functionalities:

- Dataset overview (shape, sample rows, class distribution)
- Feature correlation heatmap for exploratory data analysis
- CSV upload option for test data (as required by the assignment)
- Model selection dropdown
- Display of classification report
- Display of confusion matrix

The application is deployed using **Streamlit Community Cloud** and is accessible via a public URL.

---

## Project Structure
ml_assignment_2/
│
├── app.py
├── create_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│ └── breast_cancer.csv
│
├── model/
│ ├── train_models.py
│ ├── evaluate_models.py
│ ├── scaler.pkl
│ ├── Logistic Regression.pkl
│ ├── Decision Tree.pkl
│ ├── KNN.pkl
│ ├── Naive Bayes.pkl
│ ├── Random Forest.pkl
│ └── XGBoost.pkl


---

## Deployment

The application is deployed using **Streamlit Community Cloud** by linking the GitHub repository and selecting `app.py` as the main application file. All required dependencies are listed in `requirements.txt` to ensure successful deployment.

---

## Conclusion

This project demonstrates a complete machine learning pipeline from dataset preparation to model deployment. Multiple classification models were trained and evaluated using standard performance metrics, and an interactive Streamlit dashboard was developed to visualize results and allow user interaction. The project strictly follows the assignment guidelines and showcases practical machine learning deployment skills.
