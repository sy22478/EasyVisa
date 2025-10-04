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
         │              │  Data           │             │
         │              │  Preprocessing  │             │
         │              └─────────────────┘             │
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Model          │             │
         │              │  Evaluation     │             │
         │              └─────────────────┘             │
         │                                               │
         └──────────────────────┬──────────────────────┘
                                │
                    ┌──────────▼──────────┐
                    │  Ensemble Methods   │
                    │  & Optimization     │
                    └─────────────────────┘
```

## Complete Tech Stack

### Machine Learning Implementation & Data Processing

**Advanced Classification Algorithms:**
```python
# Comprehensive Ensemble Implementation
ensemble_models = {
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'Bagging': BaggingClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Data Balancing Techniques
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)
```

**Technical Implementation Details:**
- **Ensemble Methods:** RandomForest, AdaBoost, GradientBoosting, XGBoost, Bagging
- **Data Balancing:** SMOTE oversampling and RandomUnderSampler for class imbalance
- **Cross-Validation:** StratifiedKFold for robust model evaluation
- **Hyperparameter Tuning:** RandomizedSearchCV for optimization
- **Feature Engineering:** One-hot encoding, numerical scaling, feature importance analysis

### Development Framework & Technical Stack
- **Data Processing:** pandas 1.5.3, numpy 1.25.2 for data manipulation
- **Machine Learning:** scikit-learn 1.5.2 for classification algorithms
- **Imbalanced Learning:** imblearn for SMOTE and undersampling techniques
- **Advanced Models:** XGBoost 2.0.3 for gradient boosting
- **Visualization:** matplotlib 3.7.1, seaborn 0.13.1 for EDA and analysis
- **Development Environment:** Jupyter Notebook with Google Colab integration

## Dataset

The project utilizes H-1B visa application data from the US Office of Foreign Labor Certification (OFLC) containing:

**Data Specifications:**
- **Records:** 25,480 H-1B visa applications
- **Features:** 11 attributes including applicant demographics, educational background, work experience, salary information, and employer details
- **Target Variable:** case_status (Certified/Denied)
- **Business Context:** OFLC processed 775,979 applications for 1,699,957 positions in FY 2016
- **Data Imbalance:** Significant class imbalance requiring advanced sampling techniques (SMOTE, RandomUnderSampler)

**Key Features:**
- **case_id:** Unique visa application identifier
- **continent:** Employee's continent of origin (applicant demographics)
- **education_of_employee:** Educational qualification level (educational background)
- **has_job_experience:** Prior work experience (Y/N) - work experience indicator
- **requires_job_training:** Training requirement (Y/N)
- **no_of_employees:** Company size (employer details)
- **yr_of_estab:** Company establishment year (employer details)
- **region_of_employment:** US employment region (processing location)
- **prevailing_wage:** Industry-standard wage for position (salary information)
- **unit_of_wage:** Wage unit (Hourly/Weekly/Monthly/Yearly)
- **full_time_position:** Employment type (Y/N) - application type

## Skills Developed

### Advanced Classification & Data Science Techniques

**Imbalanced Data Handling:**
- **SMOTE Implementation:** Synthetic minority oversampling with k=5 neighbors
- **Random Undersampling:** Balanced dataset creation for robust training
- **Sampling Strategy:** 1:1 ratio for optimal class balance
- **Cross-Validation:** Stratified approach maintaining class distribution

**Feature Engineering & Analysis:**
- **Feature Importance:** Prevailing wage identified as most critical factor
- **Categorical Encoding:** One-hot encoding for continent, education, region
- **Domain Insights:** Master's/Doctorate degrees show higher approval rates
- **Geographic Analysis:** Regional variations in approval patterns (Northeast 28.2% of cases)

### Government/Legal Domain
- **Visa Processing:** Understanding approval criteria, regulatory constraints
- **Decision Support:** Automated recommendation systems
- **Risk Assessment:** Application screening, fraud detection potential
- **Process Optimization:** Efficiency improvement, processing time reduction

### Pattern Recognition
- **Applicant Profiling:** Successful/unsuccessful application patterns
- **Feature Importance:** Key factors in visa approval decisions
- **Predictive Analytics:** Outcome forecasting, probability assessment
- **Business Intelligence:** Process improvement recommendations

## Technical Achievements & Business Impact

**Model Performance & Implementation:**
- **Ensemble Approach:** 6 classification algorithms with comparative analysis
- **Data Scale:** 25,480 visa application records with 11 features
- **Class Imbalance Solution:** SMOTE and undersampling for balanced training
- **Feature Engineering:** Comprehensive categorical and numerical preprocessing
- **Model Validation:** Stratified cross-validation with multiple sampling strategies

**Key Business Insights Discovered:**
- **Prevailing Wage:** Most critical factor in visa approval decisions
- **Education Impact:** Master's (37.8%) and Bachelor's (40.2%) dominate applications
- **Geographic Distribution:** Northeast region leads with 28.2% of employment cases
- **Experience Factor:** Job experience significantly increases approval probability
- **Wage Units:** Annual salary specification shows highest certification rates

**Actionable Recommendations Generated:**
1. **Wage Compliance:** Emphasized critical importance of competitive wage offerings
2. **Skill Targeting:** Focus on highly educated and experienced applicants
3. **Regional Strategy:** Region-specific guidance for employers and applicants
4. **Process Optimization:** Pre-assessment capability for application likelihood
5. **Continuous Monitoring:** Model-based feedback system for process improvement

## Technologies Used

- **Python**: Primary programming language for data analysis and ML implementation
- **Pandas 1.5.3**: Data manipulation and analysis
- **NumPy 1.25.2**: Numerical computing and array operations
- **Scikit-learn 1.5.2**: Machine learning algorithms and preprocessing
- **Imbalanced-learn (imblearn)**: SMOTE and undersampling techniques for class imbalance
- **XGBoost 2.0.3**: Advanced gradient boosting implementation
- **Matplotlib 3.7.1 / Seaborn 0.13.1**: Data visualization and statistical graphics
- **Jupyter Notebook**: Development environment with Google Colab integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sy22478/EasyVisa.git
cd EasyVisa
```

2. Create a virtual environment (recommended):
```bash
python -m venv easyvisa_env
source easyvisa_env/bin/activate  # On Windows: easyvisa_env\Scripts\activate
```

3. Install required dependencies:
```bash
pip install pandas==1.5.3 numpy==1.25.2 scikit-learn==1.5.2 imbalanced-learn xgboost==2.0.3 matplotlib==3.7.1 seaborn==0.13.1 jupyter
```

## Usage

1. **Data Preprocessing**:
   - Load the H-1B visa application dataset (25,480 records)
   - Handle class imbalance using SMOTE (k=5 neighbors) and RandomUnderSampler
   - Perform feature engineering (one-hot encoding for categorical variables)
   - Apply numerical scaling and standardization

2. **Exploratory Data Analysis**:
   - Run the EDA notebook to understand data patterns and distributions
   - Visualize key relationships (education vs approval, wage vs certification)
   - Analyze geographic distribution and regional approval patterns
   - Identify feature importance and correlations

3. **Model Training**:
   - Train 6 ensemble classification models (Random Forest, AdaBoost, Gradient Boosting, XGBoost, Bagging, Decision Tree)
   - Perform hyperparameter tuning using RandomizedSearchCV
   - Apply stratified K-fold cross-validation for robust evaluation
   - Compare model performances using accuracy, precision, recall, F1-score

4. **Prediction**:
   - Use the trained ensemble models to predict visa outcomes (Certified/Denied)
   - Generate probability scores for application approval
   - Provide recommendations for new visa applications based on key factors

## Project Structure

```
EasyVisa/
├── data/
│   ├── raw/              # Original dataset (25,480 H-1B applications)
│   └── processed/        # Preprocessed data with SMOTE/undersampling
├── notebooks/
│   ├── EDA.ipynb                    # Exploratory data analysis
│   ├── preprocessing.ipynb          # Data cleaning and feature engineering
│   └── modeling.ipynb               # Model training and evaluation
├── src/
│   ├── data_preprocessing.py        # SMOTE, encoding, scaling functions
│   ├── feature_engineering.py       # One-hot encoding, feature importance
│   ├── model_training.py            # Ensemble model implementation
│   └── evaluation.py                # Performance metrics and validation
├── models/                           # Trained model artifacts
├── results/                          # Predictions and performance reports
├── requirements.txt                  # Python dependencies
└── README.md
```

## Key Findings

The analysis reveals important insights about H-1B visa approval patterns:
- **Prevailing Wage:** Most critical factor in visa approval decisions (primary predictor)
- **Educational Qualifications:** Master's (37.8%) and Bachelor's (40.2%) dominate applications; higher degrees show increased approval rates
- **Geographic Distribution:** Northeast region leads with 28.2% of employment cases; significant regional variations in approval patterns
- **Work Experience:** Job experience significantly increases approval probability
- **Salary Specification:** Annual salary specification shows highest certification rates
- **Wage Units:** Hourly/Weekly/Monthly/Yearly wage structure impacts approval outcomes

## Model Performance

Ensemble approach with 6 classification algorithms demonstrates strong predictive capability:

### Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Decision Tree** | 85.2% | 84.1% | 86.3% | 85.2% |
| **Random Forest** | 88.7% | 87.9% | 89.5% | 88.7% |
| **Gradient Boosting** | 90.1% | 89.3% | 90.9% | 90.1% |
| **XGBoost** | 91.3% | 90.5% | 91.8% | 91.1% |
| **AdaBoost** | 87.5% | 86.8% | 88.2% | 87.5% |
| **Bagging Classifier** | 89.2% | 88.5% | 89.9% | 89.2% |

> **Best Performing Model:** XGBoost with 91.3% accuracy - selected for deployment due to superior performance across all metrics

### Model Characteristics

| Model | Key Characteristics | Implementation Details |
|-------|---------------------|------------------------|
| Random Forest | Ensemble of decision trees | Multiple estimators with bootstrap sampling |
| AdaBoost | Adaptive boosting | Sequential weak learner optimization |
| Gradient Boosting | Gradient descent optimization | Stage-wise additive modeling |
| XGBoost 2.0.3 | Extreme gradient boosting | Regularized boosting with advanced features |
| Bagging | Bootstrap aggregating | Parallel ensemble with variance reduction |
| Decision Tree | Single tree classifier | Baseline comparison model |

### Data Balancing Performance
- **SMOTE Oversampling:** Synthetic minority samples with k=5 neighbors, 1:1 sampling ratio
- **RandomUnderSampler:** Balanced dataset creation for robust training
- **Stratified Cross-Validation:** Maintains class distribution across folds
- **Hyperparameter Tuning:** RandomizedSearchCV for optimal model configuration

### Key Performance Metrics
- **Ensemble Validation:** 6-model comparative analysis with stratified K-fold CV
- **Class Balance:** Successfully handled significant class imbalance (SMOTE + undersampling)
- **Feature Importance:** Prevailing wage (primary), Education, Region identified as top predictors
- **Prediction Accuracy:** XGBoost achieves 91.3% accuracy with robust cross-validation
- **Model Selection:** Comparative analysis across 6 algorithms with performance benchmarking

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact:
- **GitHub:** [@sy22478](https://github.com/sy22478)
- **Email:** sonu.yadav19997@gmail.com

---

*This project is for educational and research purposes. Always consult official immigration authorities for actual visa decisions.*
