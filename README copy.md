# 🏘️ ImmoEliza - Belgian Real Estate Price Prediction 🏡

[![Made with Python](https://forthebadge.com/api/badges/generate?panels=2&primaryLabel=Made+with&secondaryLabel=Python&primaryBGColor=%2383ba7c&primaryTextColor=%23FFFFFF&secondaryBGColor=%23426f3c&secondaryTextColor=%23FFFFFF&primaryFontSize=12&primaryFontWeight=600&primaryLetterSpacing=2&primaryFontFamily=Roboto&primaryTextTransform=uppercase&secondaryFontSize=12&secondaryFontWeight=900&secondaryLetterSpacing=2&secondaryFontFamily=Montserrat&secondaryTextTransform=uppercase&secondaryIcon=python&secondaryIconColor=%23FFFFFF&secondaryIconSize=17&secondaryIconPosition=right)](https://forthebadge.com)

## 🚀 Project Context

The goal of this project is to develop a machine learning model that can reliably predict real estate property prices in Belgium for Immo Eliza. 
After collecting, cleaning, and exploring the data, the next step is to preprocess the dataset and engineer relevant features in order to train a robust and performant predictive model that can support data-driven pricing decisions.

---

## 🧰 Project Pipeline

### 1. Raw Data Audit

Script: `raw_auditor.py`  

This is a first cleaning step where obvious data quality issues are adressed (duplicates, invalid prices, etc.)

- Input: a CSV file containing Belgian real estate data.
- Output: an audited CSV file ready for data cleaning.

Main steps:

- fix swapped latitude/longitude values
- remove rows with invalid or impossible prices
- remove duplicated listings
- remove suspicious duplicate groups
- remove or null impossible values for key numerical fields
- remove top 1% price outliers to keep the prediction scope focused on standard residential properties

---

### 2. Dataset Cleaning

Script: `dataset_cleaner.py`  

This step removes columns that will not be used to train models (metadata, high missingness, etc.)  

- Input: The audited CSV file.
- Output: A clean CSV file ready for feature engineering.

Main steps:

- drop unused or unreliable columns
- extract posting year and month
- keep only features considered relevant for price prediction
- save the clean dataset used for feature engineering

---

### 3. Feature Engineering

Script: `features_engineer.py`

This step makes the final preparations before the dataset is ready for model training.  

Main transformations:

- convert availability into `available_immediately`
- clean and group property categories
- transform EPC labels into energy quality groups
- clean flooding area information
- convert indoor/outdoor parking counts into binary indicators
- group building state values
- create `property_age`
- enrich listings with Statbel commune-level real estate statistics
- split features and target variable

Target variable:

- `price`

---

## 💻 Models

The models trained in this project are:

- Random Forest 
- XGBoost 

### Model Evaluation

Model performance is assessed using the following metrics:

- R² Score to measure how well the model explains the variance in property prices
- Mean Absolute Error (MAE) to quantify the average absolute difference between predicted and actual prices
- Root Mean Squared Error (RMSE) to penalize larger prediction errors more heavily

In addition to these metrics, the model is also analyzed for underfitting and overfitting to ensure it generalizes well to unseen data.

The final selected model is **XGBoost**.

### 🥇 Model Comparison

#### Final Metrics

| Model | Split | MAE (EUR) | RMSE (EUR) | R2 |
|---|---|---:|---:|---:|
| Random Forest | Train | 25,135 | 50,484 | 0.9546 |
| XGBoost | Train | 42,771 | 74,924 | 0.9001 |
| Random Forest | Test | 60,543 | 112,319 | 0.7769 |
| XGBoost | Test | 55,878 | 99,853 | 0.8236 |

#### Overfitting Check

| Model | MAE Gap (EUR) | RMSE Gap (EUR) | R2 Gap |
|---|---:|---:|---:|
| Random Forest | 35,408 | 61,835 | 0.1778 |
| XGBoost | 13,107 | 24,929 | 0.0765 |

XGBoost was selected as the final model because it achieved the best balance between prediction accuracy and generalization.  

#### Metrics Interpretation

The evaluation results show that both models are capable of predicting property prices with good accuracy, but XGBoost demonstrates stronger performance on unseen data.

The lower MAE indicates that XGBoost produces more accurate predictions on average, while the lower RMSE suggests it is also less affected by large prediction errors.

XGBoost achieved a test R² of 0.8236, meaning it explains approximately 82% of the variation in property prices within the test dataset. In comparison, Random Forest achieved a test R² of 0.7769, explaining around 78% of the variance.

Overall, XGBoost provides the best trade-off between predictive accuracy and generalization. It consistently outperforms Random Forest on the test dataset while exhibiting substantially lower overfitting, making it the most reliable model for estimating property prices in this study.

Note: Based on the preliminary model evaluation, XGBoost consistently outperformed Random Forest and thus was selected for further optimization through hyperparameter tuning, while Random Forest was retained using its baseline configuration for comparison.  

---
## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── clean/
├── models/
├── src/
│   ├── config.py
│   ├── raw_auditor.py
│   ├── dataset_cleaner.py
│   ├── features_engineer.py
│   └── model_training/
│       ├── train_random_forest.py
│       └── train_xgboost.py
├── requirements.txt
└── README.md
```
---

## 🛠️ Installation & Usage

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the project root, with the virtual environment activated.

### 1. Audit the raw dataset

```bash
python -m src.raw_auditor
```

This creates the audited raw dataset in `data/raw/`.

### 2. Clean the dataset

```bash
python -m src.dataset_cleaner
```

This creates the cleaned dataset in `data/clean/`.

### 3. Train the Random Forest model

```bash
python -m src.model_training.train_random_forest
```

This trains and saves the Random Forest pipeline.

### 4. Train the XGBoost model

```bash
python -m src.model_training.train_xgboost
```

This trains and saves the XGBoost pipeline.

Saved models are written to:

```text
models/
```

---  
## 🏃 Timeline 🏃

The project was completed over 5 days.

---  

## 🐈‍⬛ Personal Situation 🐈‍⬛

The project was done as part of the AI & Data Science bootcamp at [BeCode](https://becode.org/).
