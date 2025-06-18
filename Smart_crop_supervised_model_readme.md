# Crop Suitability Multi-Label Prediction Project

## 1. Project Overview

This project aims to build a model to predict the suitability of different crops in various locations based on environmental and geographical factors. It addresses the problem as a multi-label classification task, where each location can be suitable for multiple crops. We explore several machine learning models and evaluate their performance.

## 2. Dataset

The project utilizes a dataset containing various features related to land and climate conditions, along with suitability labels for five specific crops: 'Paddy', 'Wheat', 'Apple', 'Tea', and 'Coconut'.

The dataset is loaded from the file `/content/crop_suitability_multi_label_5 _class.csv`.

## 3. Methodology

The project follows a standard Deep Learning and machine learning workflow:

### 3.1 Data Loading and Initial Inspection

- The dataset is loaded into a pandas DataFrame.
- Initial checks for missing values are performed.

### 3.2 Data Preprocessing and Feature Engineering

- **Data Cleaning:** Boolean crop suitability columns are converted to integers (0/1).
- **Encoding Categorical Features:** 'State' and 'Season' columns are encoded using `LabelEncoder`.
- **Feature Engineering:**
    - Region-based features (`State_Rainfall_Deviation`, `State_Temp_Deviation`) are created by calculating the deviation of local rainfall and temperature from the average for each state.
    - Interaction features (`Rainfall_Temp_Interaction`, `NDVI_pH_Interaction`) are created by multiplying relevant environmental factors.
- **Feature Selection:** A list of relevant features is defined for model training.
- **Target Variable Preparation:** The selected crop suitability columns are defined as the target variables.
- **Feature Scaling:** Continuous features are scaled using `StandardScaler` to normalize their ranges. The scaler is saved for later use in deployment.
- **Train-Test Split:** The data is split into training and testing sets (80% train, 20% test).

### 3.3 Model Development

We explore three different modeling approaches:

#### 3.3.1 Deep Learning Model

- **Class Weights:** Class weights are calculated to handle potential data imbalance for each crop.
- **Custom Weighted Loss Function:** A custom weighted binary cross-entropy loss function is implemented to incorporate the class weights during training.
- **Model Architecture:** A sequential neural network model is built with:
    - Multiple `Dense` layers with `relu` activation.
    - `BatchNormalization` layers to improve training stability.
    - `Dropout` layers for regularization.
    - An output layer with `sigmoid` activation for multi-label prediction.
- **Compilation:** The model is compiled with the 'adam' optimizer and the custom weighted loss function. Metrics such as 'accuracy', 'auc', 'Precision', and 'Recall' are monitored.
- **Callbacks:** `EarlyStopping` and `ReduceLROnPlateau` callbacks are used during training to prevent overfitting and adjust the learning rate.
- **Training:** The model is trained on the training data with a validation split.
- **Saving:** The trained Deep Learning model is saved in HDF5 format (`.h5`).

#### 3.3.2 Random Forest Baseline

- **Model Initialization:** A `RandomForestClassifier` is initialized with `n_estimators=100` and `class_weight='balanced'`.
- **Multi-Output Wrapper:** The `RandomForestClassifier` is wrapped in a `MultiOutputClassifier` to handle the multi-label nature of the problem.
- **Training:** The model is trained on the training data.
- **Saving:** The trained Random Forest model is saved using `joblib`.

#### 3.3.3 XGBoost Model

- **Model Initialization:** An `XGBClassifier` is initialized with `objective='binary:logistic'` and `is_unbalance=True`.
- **Multi-Output Wrapper:** The `XGBClassifier` is wrapped in a `MultiOutputClassifier`.
- **Hyperparameter Tuning:** `RandomizedSearchCV` is used to find the best hyperparameters for the XGBoost model. A dictionary of parameters to sample from is defined.
- **Training:** The `RandomizedSearchCV` is fit to the training data to find the best model.
- **Saving:** The best XGBoost model found by the search is saved using `joblib`.

### 3.4 Feature Importance and Visualization

- Feature importance is visualized for the Random Forest and XGBoost models to understand which features contribute most to the predictions for each crop.
- A sample decision tree from the Random Forest model is visualized.

### 3.5 Evaluation

- A function `evaluate_model` is defined to calculate and print evaluation metrics (Precision, Recall, F1-score) using `classification_report` for each crop and model.
- The Deep Learning, Random Forest, and XGBoost models are evaluated on the test set.
- A DataFrame is created to combine the evaluation metrics from all models.
- Model performance is visualized using bar plots comparing F1-scores and Precision across crops and models.

### 3.6 Deployment Preparation

- A function `predict_crop_suitability` is created to demonstrate how to make predictions using the trained Deep Learning model. This function takes raw feature values as input, performs the necessary feature engineering and scaling, and returns predicted suitability and probabilities for each crop.

## 4. Code Structure

The project code is organized into sections in a Google Colab notebook, with clear steps for data loading, preprocessing, modeling, evaluation, and deployment preparation.

## 5. Results

The evaluation metrics and visualizations provide insights into the performance of each model.

## 6. Future Work

- Exploring additional features or feature engineering techniques.
- Investigating other multi-label classification algorithms.
- Implementing cross-validation for more robust model evaluation.
- Developing a user interface for prediction.
- Incorporating spatial data (if available) for more granular predictions.
- Create a dashboad that can automatically fetch user location, weather and soil features.

## 7. How to Run the Code

1. Open the Google Colab notebook containing the project code.
2. Ensure the dataset file (`crop_suitability_multi_label_5 _class.csv`) is accessible in the Colab environment (e.g., uploaded to your Colab session or mounted from Google Drive).
3. Run each code cell sequentially.
4. The notebook will output the results of data inspection, training progress, evaluation reports, and visualizations.

## 8. Dependencies

The project requires the following Python libraries, which are installed at the beginning of the notebook:

- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `shap`
- `tensorflow`
- `geopandas`
- `pandas`
- `numpy`
- `joblib`

## 9. Author

[Rahul Debnath]
