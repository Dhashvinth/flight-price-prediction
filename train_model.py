import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from preprocess import preprocess_data


# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv("Flight_Price.csv")


print("\nDataset Shape:")
print(df.shape)


print("\nFirst 5 Rows:")
print(df.head())


# ======================================================
# CHECK MISSING VALUES
# ======================================================

print("\nMissing Values:")

print(df.isnull().sum())


# Only one row has missing values in our dataset.
# So we remove the incomplete row.

df = df.dropna()


print("\nShape After Removing Missing Values:")

print(df.shape)


# ======================================================
# OUTLIER ANALYSIS
# ======================================================

Q1 = df["Price"].quantile(0.25)

Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1


lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR


print("\nQ1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower Limit:", lower_limit)

print("Upper Limit:", upper_limit)


outliers = df[

    (df["Price"] < lower_limit)

    |

    (df["Price"] > upper_limit)

]


print("\nNumber of Price Outliers:")

print(len(outliers))


print("\nTop Expensive Outliers:")

print(

    outliers[

        [
            "Airline",
            "Additional_Info",
            "Price"
        ]

    ]

    .sort_values(

        "Price",

        ascending=False

    )

    .head(10)

)


# We are not removing the outliers.
# Some of the high-price observations are valid
# Business Class or premium flights.


# ======================================================
# PREPROCESS DATA
# ======================================================

df = preprocess_data(df)


print("\nData After Preprocessing:")

print(df.head())


# ======================================================
# CHECK DATA TYPES
# ======================================================

print("\nData Types After Preprocessing:")

print(df.dtypes)


# ======================================================
# CHECK OBJECT COLUMNS
# ======================================================

object_cols = df.select_dtypes(

    include=["object"]

).columns


print("\nObject Columns:")


for column in object_cols:

    print("\nColumn:", column)

    print(

        df[column].unique()[:20]

    )


# ======================================================
# SPLIT X AND y
# ======================================================

X = df.drop(

    "Price",

    axis=1

)


y = df["Price"]


# ======================================================
# IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ======================================================

categorical_cols = X.select_dtypes(

    include=["object"]

).columns.tolist()


numerical_cols = X.select_dtypes(

    exclude=["object"]

).columns.tolist()


print("\nCategorical Columns:")

print(categorical_cols)


print("\nNumerical Columns:")

print(numerical_cols)


# ======================================================
# NUMERICAL PIPELINE
# ======================================================

numeric_pipeline = Pipeline([

    (

        "imputer",

        SimpleImputer(

            strategy="median"

        )

    ),

    (

        "scaler",

        StandardScaler()

    )

])


# ======================================================
# CATEGORICAL PIPELINE
# ======================================================

categorical_pipeline = Pipeline([

    (

        "imputer",

        SimpleImputer(

            strategy="most_frequent"

        )

    ),

    (

        "onehot",

        OneHotEncoder(

            handle_unknown="ignore"

        )

    )

])


# ======================================================
# COLUMN TRANSFORMER
# ======================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_pipeline,

            numerical_cols

        ),

        (

            "cat",

            categorical_pipeline,

            categorical_cols

        )

    ]

)


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


print("\nTraining Rows:")

print(len(X_train))


print("\nTesting Rows:")

print(len(X_test))


# ======================================================
# FUNCTION TO EVALUATE MODEL
# ======================================================

def evaluate_model(model_name, model):

    print("\n======================================")

    print(model_name)

    print("======================================")


    # Train model

    model.fit(

        X_train,

        y_train

    )


    # Prediction on training data

    train_prediction = model.predict(

        X_train

    )


    # Prediction on testing data

    test_prediction = model.predict(

        X_test

    )


    # ======================================================
    # TRAINING METRICS
    # ======================================================

    train_mae = mean_absolute_error(

        y_train,

        train_prediction

    )


    train_rmse = np.sqrt(

        mean_squared_error(

            y_train,

            train_prediction

        )

    )


    train_r2 = r2_score(

        y_train,

        train_prediction

    )


    # ======================================================
    # TESTING METRICS
    # ======================================================

    test_mae = mean_absolute_error(

        y_test,

        test_prediction

    )


    test_rmse = np.sqrt(

        mean_squared_error(

            y_test,

            test_prediction

        )

    )


    test_r2 = r2_score(

        y_test,

        test_prediction

    )


    print("\nTraining Results")

    print("MAE :", train_mae)

    print("RMSE:", train_rmse)

    print("R2  :", train_r2)


    print("\nTesting Results")

    print("MAE :", test_mae)

    print("RMSE:", test_rmse)

    print("R2  :", test_r2)


    # ======================================================
    # OVERFITTING CHECK
    # ======================================================

    r2_difference = train_r2 - test_r2


    print("\nTrain-Test R2 Difference:")

    print(r2_difference)


    # ======================================================
    # CROSS VALIDATION
    # ======================================================

    cv_scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="r2"

    )


    print("\nCross Validation R2 Scores:")

    print(cv_scores)


    print("\nAverage Cross Validation R2:")

    print(cv_scores.mean())


    return model


# ======================================================
# MULTIPLE LINEAR REGRESSION
# ======================================================

linear_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        LinearRegression()

    )

])


evaluate_model(

    "MULTIPLE LINEAR REGRESSION",

    linear_model

)


# ======================================================
# LINEAR REGRESSION ASSUMPTION CHECKS
# ======================================================

print("\n======================================")

print("LINEAR REGRESSION ASSUMPTION CHECKS")

print("======================================")


# Predict on unseen test data.

linear_test_prediction = linear_model.predict(

    X_test

)


# Residual = Actual Price - Predicted Price

linear_residuals = (

    y_test

    -

    linear_test_prediction

)


# ======================================================
# 1. LINEARITY AND HOMOSCEDASTICITY
# ======================================================

# Residuals should ideally be randomly scattered
# around zero.
#
# A systematic curved pattern can indicate
# non-linearity.
#
# A funnel-shaped pattern can indicate
# heteroscedasticity.

plt.figure(

    figsize=(8, 5)

)


plt.scatter(

    linear_test_prediction,

    linear_residuals,

    alpha=0.5

)


plt.axhline(

    y=0,

    linestyle="--"

)


plt.xlabel(

    "Predicted Flight Price"

)


plt.ylabel(

    "Residuals"

)


plt.title(

    "Linear Regression: Residuals vs Predicted Values"

)


plt.tight_layout()


plt.show()


# ======================================================
# 2. NORMALITY OF RESIDUALS
# ======================================================

# If residuals are approximately normal,
# most points in the Q-Q plot should follow
# the diagonal reference line.

plt.figure(

    figsize=(8, 5)

)


stats.probplot(

    linear_residuals,

    dist="norm",

    plot=plt

)


plt.title(

    "Linear Regression: Q-Q Plot of Residuals"

)


plt.tight_layout()


plt.show()


# ======================================================
# 3. MULTICOLLINEARITY SCREENING
# ======================================================

# We perform a simple pairwise correlation check
# among the original numerical predictors.
#
# Absolute correlation >= 0.80 is treated as
# a warning of potentially redundant numerical
# information.
#
# This is a screening check rather than a complete
# VIF-based multicollinearity analysis.

numerical_correlation = (

    X_train[numerical_cols]

    .corr()

    .abs()

)


high_correlation_pairs = []


for i in range(

    len(numerical_correlation.columns)

):

    for j in range(i):

        correlation_value = (

            numerical_correlation.iloc[i, j]

        )


        if correlation_value >= 0.80:

            high_correlation_pairs.append(

                (

                    numerical_correlation.columns[i],

                    numerical_correlation.columns[j],

                    correlation_value

                )

            )


print(

    "\nHighly Correlated Numerical Feature Pairs "
    "(absolute correlation >= 0.80):"

)


if len(high_correlation_pairs) == 0:

    print(

        "No highly correlated numerical "
        "feature pairs found."

    )


else:

    for (

        feature_1,

        feature_2,

        correlation_value

    ) in high_correlation_pairs:

        print(

            feature_1,

            "<->",

            feature_2,

            ":",

            round(

                correlation_value,

                3

            )

        )


# ======================================================
# 4. INDEPENDENCE OF OBSERVATIONS
# ======================================================

# The dataset contains individual flight observations
# rather than a conventional time-series sequence.
#
# Therefore we assess independence mainly from the
# dataset structure rather than automatically applying
# a time-series residual test such as Durbin-Watson.

print("\nIndependence Check:")


print(

    "The dataset contains individual flight observations. "
    "No time-series residual test is applied."

)


# ======================================================
# ASSUMPTION CHECK INTERPRETATION
# ======================================================

print(

    "\nHow to interpret the diagnostic results:"

)


print(

    "1. Residuals should be randomly scattered around zero."

)


print(

    "2. A curved pattern in the residual plot "
    "suggests non-linearity."

)


print(

    "3. A funnel-shaped residual plot suggests "
    "non-constant error variance."

)


print(

    "4. Q-Q plot points close to the diagonal suggest "
    "approximately normal residuals."

)


print(

    "5. Highly correlated predictors can indicate "
    "multicollinearity."

)


print(

    "\nThese assumptions apply to the Linear Regression "
    "baseline and are not strict assumptions of XGBoost."

)


# ======================================================
# RANDOM FOREST
# ======================================================

random_forest_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        RandomForestRegressor(

            n_estimators=200,

            random_state=42

        )

    )

])


evaluate_model(

    "RANDOM FOREST",

    random_forest_model

)


# ======================================================
# XGBOOST - INITIAL MODEL
# ======================================================

xgboost_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        XGBRegressor(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=6,

            random_state=42

        )

    )

])


evaluate_model(

    "XGBOOST - BEFORE TUNING",

    xgboost_model

)


# ======================================================
# XGBOOST HYPERPARAMETER TUNING
# ======================================================

print("\n======================================")

print("XGBOOST HYPERPARAMETER TUNING")

print("======================================")


xgb_tuning_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        XGBRegressor(

            random_state=42

        )

    )

])


# ======================================================
# PARAMETERS TO CONTROL OVERFITTING
# ======================================================

parameters = {

    "model__n_estimators": [

        200,

        300,

        400

    ],

    "model__learning_rate": [

        0.03,

        0.05,

        0.1

    ],

    "model__max_depth": [

        3,

        4,

        5

    ],

    "model__min_child_weight": [

        1,

        3,

        5

    ],

    "model__subsample": [

        0.8,

        0.9,

        1.0

    ]

}


# RandomizedSearchCV does not test every possible
# hyperparameter combination.
#
# It randomly evaluates 20 combinations using
# 5-fold cross-validation.
#
# The combination with the lowest average
# cross-validated RMSE is selected.

random_search = RandomizedSearchCV(

    xgb_tuning_model,

    parameters,

    n_iter=20,

    cv=5,

    scoring="neg_root_mean_squared_error",

    random_state=42,

    n_jobs=-1

)


random_search.fit(

    X_train,

    y_train

)


# ======================================================
# BEST CROSS-VALIDATED RMSE
# ======================================================

# Scikit-learn stores RMSE as a negative score
# because higher scores are normally considered better.
#
# Multiplying by -1 converts it back to normal RMSE.

best_cv_rmse = (

    -random_search.best_score_

)


print("\nBest Cross-Validated RMSE:")

print(best_cv_rmse)


# ======================================================
# BEST PARAMETERS
# ======================================================

print("\nBest Parameters:")


print(

    random_search.best_params_

)


# ======================================================
# FINAL MODEL
# ======================================================

# This is the XGBoost pipeline containing the
# hyperparameters selected using cross-validation.

final_model = (

    random_search.best_estimator_

)


# ======================================================
# FINAL MODEL PREDICTIONS
# ======================================================

train_prediction = final_model.predict(

    X_train

)


test_prediction = final_model.predict(

    X_test

)


# ======================================================
# FINAL TRAINING METRICS
# ======================================================

final_train_mae = mean_absolute_error(

    y_train,

    train_prediction

)


final_train_rmse = np.sqrt(

    mean_squared_error(

        y_train,

        train_prediction

    )

)


final_train_r2 = r2_score(

    y_train,

    train_prediction

)


# ======================================================
# FINAL TESTING METRICS
# ======================================================

final_test_mae = mean_absolute_error(

    y_test,

    test_prediction

)


final_test_rmse = np.sqrt(

    mean_squared_error(

        y_test,

        test_prediction

    )

)


final_test_r2 = r2_score(

    y_test,

    test_prediction

)


# ======================================================
# FINAL MODEL RESULTS
# ======================================================

print("\n======================================")

print("FINAL XGBOOST MODEL")

print("======================================")


print("\nTraining MAE:")

print(final_train_mae)


print("\nTraining RMSE:")

print(final_train_rmse)


print("\nTraining R2:")

print(final_train_r2)


print("\nTesting MAE:")

print(final_test_mae)


print("\nTesting RMSE:")

print(final_test_rmse)


print("\nTesting R2:")

print(final_test_r2)


print("\nTrain-Test R2 Difference:")


print(

    final_train_r2

    -

    final_test_r2

)


# ======================================================
# FINAL CROSS VALIDATION
# ======================================================

final_cv_scores = cross_val_score(

    final_model,

    X_train,

    y_train,

    cv=5,

    scoring="r2"

)


print(

    "\nFinal Model Cross Validation R2 Scores:"

)


print(

    final_cv_scores

)


print(

    "\nFinal Average Cross Validation R2:"

)


print(

    final_cv_scores.mean()

)


# ======================================================
# SAVE FINAL MODEL
# ======================================================

model_path = (

    Path(__file__).parent

    /

    "model.pkl"

)


joblib.dump(

    final_model,

    model_path

)


print(

    "\nModel Saved Successfully!"

)


print(

    "Model saved at:",

    model_path

)