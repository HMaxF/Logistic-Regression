"""
Demo to show Logistic Regression (classification) 

Demo v1.0
Hariyanto Lim
Last update: 2024-07-23
"""
# interactive mode for Jupyter notebbok in browser (not in command line)
# require: pip install ipympl
#%matplotlib widget
import matplotlib.pylab as plt
import seaborn as sns

import sys # to get arguments (parameter) manually

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay

import pickle # to save/load model to/from file

def save_model(model, filename):
    print('### save_model()')

    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):

    try:
        # load both model
        model = pickle.load(open(filename, 'rb'))

        print(f"### load_model(): '{filename}' loaded")

        return model

    except FileNotFoundError:
        print(f"### load_model(): error '{filename}' not found")

    return None

def predict_data(model, X):
    # predict
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]
    
    

    # visualize_confusion_matrix(model, conf_matrix)
    # visualize_roc_curve(roc_auc, y, y_pred_prob)

    return y_pred, y_pred_prob

def show_result_info(y, y_pred, y_pred_prob):
    """
    LEARNING POINT: after predict then check the accuracy
    """
    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_prob)
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'ROC-AUC: {roc_auc:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    
def create_and_train_model(df):
    """
    Main job: create model from DataFrame
    This function does NOT split dataset, use it fully to build model.
    """

    print(f"### create_and_train_model(DataFrame)")

    # define X ==> data to train (independent) and y ==> the target (dependent, annotated)
    X = df.drop(df.columns[-1], axis=1) # df.columns[-1] == the most right side column
    y = df.take([-1], axis=1)

    """
    Optional check: data to learn (X) should be numerical (int or float), there should NOT be any blank, text, or symbol.    
    """
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    is_all_numeric = not numeric_df.isnull().values.any()
    if is_all_numeric == False:
        print(f"*** ERROR: the data content of CSV to be used as training model has non-numeric value, please fix this!")
        print(df)
        return None
    
    # """
    # LEARNING POINT: A common exercise to split the input data for training and for testing,
    # Reasons:
    # 1. To evaluate the model we should not use data from other source,
    #    we only trust all data points from this same source.
    # 2. To create unseen data to get performance of the model using unseen data later.
    # """
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    # increase max_iter to avoid ConvergenceWarning: lbfgs failed to converge (status=1):
    # STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    model = LogisticRegression(max_iter=200) # default max_iter=100

    # with only values (exclude header text)
    # avoid DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().    
    model.fit(X.values, y.values.ravel())

    print(f"Logistic Regression model is created")

    # show the coefficient and intercept value
    print(f"***\nCoefficients = {model.coef_}\nIntercept = {model.intercept_}\n***")

    # learning point
    # show the model information but predicting the same data
    y_pred, y_pred_prob = predict_data(model, X)
    show_result_info(y, y_pred, y_pred_prob)

    return model

def load_csv_into_dataframe(csv_filename):
    try:
        df = pd.read_csv(csv_filename)

        return df

    except FileNotFoundError:
        print(f"*** load_csv_into_dataframe(): '{csv_filename}' not found")

    return None

# def visualize_data(df):

#     # Histograms of Features,
#     # total 14 columns, remove the last column (it is dependend variable)
#     numerical_features = ['Age', 'Sex', 'CP', 'Trestbps', 'Chol', 'FBS', 'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal']

#     # error: ValueError: num must be an integer with 1 <= num <= 6, not 7

#     # limit to maximum only 6 Features
#     numerical_features = ['Age', 'Sex', 'Trestbps', 'Chol', 'Thalach', 'Oldpeak']

#     plt.figure(figsize=(15, 10))
#     for i, feature in enumerate(numerical_features):
#         plt.subplot(2, 3, i+1)
#         plt.hist(df[feature], bins=20, edgecolor='k')
#         plt.title(f'{feature} Distribution')
#         plt.xlabel(feature)
#         plt.ylabel('Frequency')

#     plt.tight_layout()
#     plt.show()

# def visualize_correlation_matrix(df):
#     # Correlation Matrix, to display all Features including Target
#     corr_matrix = df.corr()

#     plt.figure(figsize=(12, 10))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
#     plt.title('Correlation Matrix')
#     plt.show()

# def visualize_roc_curve(roc_auc, y, y_predicted):
#     # ROC Curve
#     fpr, tpr, thresholds = roc_curve(y, y_predicted)

#     plt.figure()
#     plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.show()

# def visualize_confusion_matrix(model, conf_matrix):
#     # Confusion Matrix
#     disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
#     disp.plot(cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.show()

def show_script_info():
    print(f"Learning Logistic Regression")
    print(f"Using dummy dataset")

if __name__ == "__main__":

    show_script_info()

    input_filename = 'health-insurance-dummy-dataset-to-build-model.csv'

    df = load_csv_into_dataframe(input_filename)
    if df is None:
        print(f"load csv '{input_filename}' failed")
        exit(-3)

    # visualize_data(df)
    # visualize_correlation_matrix(df)
        
    # create model
    model = create_and_train_model(df)
    if model is None:            
        exit(-10)

    # predict
    data_to_predict_filename = 'health-insurance-dummy-dataset-to-predict.csv'
    df_predict = load_csv_into_dataframe(data_to_predict_filename)
    if df_predict is None:
        print(f"load csv '{data_to_predict_filename}' failed")
        exit(-4)

    #print(df_predict)
    #predict_data(model, df_predict.drop(df_predict.columns[-1], axis=1).values, df_predict.take([-1], axis=1).values)
    y_pred, y_pred_prob = predict_data(model, df_predict.values)

    # Combine predictions with new data for output
    df_predict['want_to_get_health_insurance'] = y_pred
    df_predict['probability'] = y_pred_prob

    # show output
    print("***\npredict result:")
    print(df_predict)

    # save predict result
    output_filename = 'health-insurance-dummy-dataset-predict-result.csv'
    df_predict.to_csv(output_filename, index=False)
    print(f"predict result saved to '{output_filename}'")

    print(f"*** app is ended normally ***")