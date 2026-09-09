# Flight Price Prediction

A machine learning project that predicts flight ticket prices based on flight and journey details.

The project compares different regression models and deploys the final model using Streamlit.

## Live Demo

[Flight Price Prediction - Live App](https://flight-price-prediction-slacdrwa42j5yxgptc5qhg.streamlit.app/)

## Project Workflow

- Data cleaning and preprocessing
- Feature engineering for journey date, time, duration, and stops
- Outlier analysis
- Multiple Linear Regression as a baseline
- Linear Regression residual diagnostics
- Random Forest Regression
- XGBoost Regression
- Model evaluation using MAE, RMSE, and R²
- 5-fold cross-validation
- XGBoost hyperparameter tuning using RandomizedSearchCV
- Streamlit deployment

## Models Compared

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Multiple Linear Regression | ₹1,746.85 | ₹2,552.73 | 0.6978 |
| Random Forest | ₹603.21 | ₹1,414.49 | 0.9072 |
| XGBoost - Before Tuning | ₹838.76 | ₹1,414.92 | 0.9072 |
| Tuned XGBoost | ₹787.61 | ₹1,474.91 | 0.8991 |

Multiple Linear Regression was used as the baseline. Residual diagnostics indicated limitations in the linear model, so nonlinear tree-based models were also evaluated.

Random Forest and XGBoost both showed significant improvement over the baseline.

## XGBoost Hyperparameter Tuning

XGBoost was tuned using `RandomizedSearchCV` with 5-fold cross-validation and RMSE as the optimization metric.

Best parameters:

```text
n_estimators = 400
learning_rate = 0.1
max_depth = 5
min_child_weight = 3
subsample = 0.9
```

Best cross-validated RMSE: **₹1,506.04**

## Final Model

The tuned XGBoost model was used for deployment.

Final test performance:

- **R²:** 0.8991
- **MAE:** ₹787.61
- **RMSE:** ₹1,474.91
- **Average Cross-Validation R²:** 0.8879

## Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- SciPy
- Streamlit
- Joblib

## Project Structure

```text
Flight-Price-Prediction/
├── app.py
├── preprocess.py
├── train_model.py
├── Flight_Price.csv
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

To retrain the model:

```bash
python train_model.py
```
