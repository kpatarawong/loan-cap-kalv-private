# Loan Default Prediction – Capstone Project

This project builds a machine learning model to predict loan defaults based on applicant financial profiles. It was developed as the capstone project for my B.S. in Computer Science degree and demonstrates end-to-end skills in data preprocessing, model training, evaluation, and interface development using Python.

Project Overview

- Objective: Predict whether a loan applicant is likely to default using structured financial data.
- Model: Random Forest Classifier (with hyperparameter tuning and class balancing)
- Interface: Interactive form built using ipywidgets for simulation and prediction
- Output: Live approval/denial decision with detailed logging
- Visualization: Performance metrics and data distribution charts

Key Features

- Feature engineering (Loan-to-Income ratio)
- Standardization using StandardScaler
- Custom class weighting to address class imbalance
- Integrated approval logic with business-rule checks (e.g., minimum credit score)
- Real-time user input interface for prediction
- Confusion matrix, classification report, and data visualizations

Technology Stack

- Language: Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, ipywidgets, joblib
- Model: RandomForestClassifier
- Interface: Jupyter Notebook with widget-based user interaction
- Deployment: Local notebook with model serialization (joblib)

Dataset

- Source: Public dataset from Kaggle.com, and CSV file hosted on GitHub 
- Features Used:
  - Credit Score
  - Income
  - Loan Amount
  - Debt-to-Income Ratio (DTI)
  - Loan-to-Income Ratio (engineered feature)

Model Details

The Random Forest Classifier was tuned with the following configuration:

RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight={0: 1, 1: 1.3},
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)

- max_depth=10: Limits tree growth to prevent overfitting
- n_estimators=50: Balances performance and efficiency
- class_weight: Adjusted to improve recall for the minority (default) class
- min_samples_split=5: Increases generalization

Business logic was also implemented to deny loans automatically if:
- Credit score is below 650
- Debt-to-Income ratio exceeds 45%

Evaluation Metrics

- Accuracy: Approximately 90% on the test set
- Additional Metrics: Precision, recall, and F1-score from classification report
- Visualization: Confusion matrix and class distribution plots

Usage Instructions via GitHub

1. Clone the repository:
    git clone https://github.com/kpatarawong/loan-cap-kalv-private.git
    cd loan-default-predictor

2. Launch Jupyter Notebook:
    jupyter notebook

3. Open Capstone_Loan.ipynb, run all cells, and use the widget interface to simulate loan applications.

Usage Instructions via Google Colab

Open Google Colab

1. Click the following link to access the application on Google Colab:
https://colab.research.google.com/drive/1nMylb5WNHeIDq08Xuq4Ih8E33
8DuKqf4?usp=sharing

2. Log in with your Google email or create a Google account.

3. Run the Notebook

4. Once opened, click "Runtime" → "Run all" to execute all code cells.

5. Once opened, click "Runtime" → "Run all" to execute all code cells.

6. A pop-up screen will display a warning, as shown below. Click “Run
Anyway.”

7. It will take about 30 seconds to run the code. Wait until you see the
interactive widget below. 

Future Improvements

- Deploy as a standalone web application using Flask or Streamlit
- Integrate explainable AI (e.g., SHAP or LIME) for better model interpretability
- Connect to a database for historical tracking and audit logs
- Expand features (e.g., employment history, credit inquiries)
