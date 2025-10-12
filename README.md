# EasyVisa - Visa Approval Prediction

## Overview
A machine learning project designed to predict visa application outcomes and streamline the visa approval process through data analysis and predictive modeling. This comprehensive classification project was developed for the US Office of Foreign Labor Certification (OFLC) and EasyVisa consulting firm to automate visa approval predictions and provide data-driven insights for the immigration process, processing 25,480 H-1B visa applications with 11 key features.

## Objectives
- Analyze patterns within visa application datasets to understand approval factors
- Develop machine learning algorithms to predict visa approval outcomes (Certified/Denied)
- Identify key factors influencing visa certification decisions (prevailing wage, education, experience)
- Create suitable applicant profiles for approved/denied applications
- Reduce processing time for visa applications through automated pre-assessment
- Improve accuracy of visa approval decisions using ensemble methods and data balancing techniques

## Features
- **Comprehensive Data Analysis**: Exploratory data analysis of 25,480 H-1B visa applications
- **Advanced Machine Learning Models**: Implementation of 6 classification algorithms:
  - Random Forest Classifier
  - AdaBoost Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
  - Bagging Classifier
  - Decision Tree Classifier
- **Class Imbalance Handling**: SMOTE oversampling and RandomUnderSampler techniques
- **Predictive Modeling**: Accurate prediction of visa approval status with stratified cross-validation
- **Feature Engineering**: One-hot encoding, numerical scaling, feature importance analysis
- **Data Visualization**: Charts and insights to visualize approval patterns and key factors
- **Performance Optimization**: RandomizedSearchCV for hyperparameter tuning

## Complete Architecture

### Classification System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Visa          │    │  Feature         │    │   ML Models     │
│   Application   │───►│  Engineering     │───►│   (Multiple)    │
│   Data          │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Data Balancing │             │
         │              │ (SMOTE/Under-   │             │
         │              │  sampling)      │             │
         │              └────────┬────────┘             │
         │                       │                      │
         │                ┌──────▼──────┐               │
         │                │ Cross-      │               │
         │                │ validation  │               │
         │                └──────┬──────┘               │
         │                       │                      │
         │                ┌──────▼──────┐               │
         │                │  Hyperparam │               │
         │                │  Tuning     │               │
         │                └─────────────┘               │
         │                                              ▼
┌─────────────────┐                               ┌───────────────┐
│   Evaluation    │◄──────────────────────────────│  Deployment   │
│ (Accuracy/F1)   │                               │  (Optional)   │
└─────────────────┘                               └───────────────┘
```

## Dataset
- Source: 25,480 H-1B visa applications with 11 features
- Target: Case status (Certified/Denied)
- Key features: Education, Experience, Prevailing Wage, Job Category, etc.

## Project Structure
```
EasyVisa/
├── data/
│   └── h1b_applications.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## How to Run
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train baseline model
python -m src.model --baseline

# Evaluate and report metrics
python -m src.evaluate
```

## Results & Insights
- Top drivers of approval include higher prevailing wages and advanced education
- Balanced training improves minority class recall without dramatically reducing overall accuracy
- Feature importance from tree-based models provides transparency for stakeholders

## Author
- GitHub: @sy22478
- LinkedIn: https://www.linkedin.com/in/sonu-yadav-a61046245/
