"""
Name: training.py
Summary:
Module for training of different clasiifiers
Author: Mamoutou Fofana
Date: 12/12/2023
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier



import json
import logging



# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/training.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


################### Load config.json and get path variables ###################
logging.info("Loading config.json for getting path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])


def split_dataset(dataset):
    """
    segregate dataset into train set X and test set y
    :param dataset: dataset to into X and y
    :return: X (features to predict) and y (target)
    """

    # target feature
    y = dataset.left
    X = dataset.drop(['left'], axis=1)

    return X, y




################# Function for training the model #################
def train_model(classifier_name):
    """
    Train a logitic regression model for a employee attrition problem
    input: None
    :return: saved the trained model to disk
    """
    logging.info("Reading the training dataset")
    trainingdata_path = os.path.join(dataset_csv_path, 'final_hr_data_comma_sep.csv')
    trainingdata = pd.read_csv(trainingdata_path)

    # split dataset into X and y
    X, y = split_dataset(trainingdata)


    # we split our dataset into 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, shuffle=True, random_state=0)
    
    # Normalize the data
    sc = StandardScaler()
    X_train_normalized = sc.fit_transform(X_train)
    X_test_normalized = sc.transform(X_test)
    

    if classifier_name == 'lg':

        # use this logistic regression for training
        logging.info("Using this logistic regression for training")
        logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, l1_ratio=None, max_iter=1000,
                        multi_class='auto', n_jobs=None, penalty='l2',
                        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                        warm_start=False)
        
        # fit the logistic regression to your data
        model = logit.fit(X_train_normalized, y_train)

        print("score on test: " + str(model.score(X_test_normalized, y_test)))
        print("score on train: "+ str(model.score(X_train_normalized, y_train)))   
    
    # Support Vector machine (SVM)
    elif classifier_name == 'svm':
        svm=LinearSVC(C=0.0001)
        model = svm.fit(X_train_normalized, y_train)

        print("score on test: " + str(model.score(X_test_normalized, y_test)))
        print("score on train: "+ str(model.score(X_train_normalized, y_train)))

    # Gradient Boosting
    elif classifier_name == "xg":
        xg = GradientBoostingClassifier()
        model = xg.fit(X_train, y_train)

        print("score on test: " + str(model.score(X_test_normalized, y_test)))
        print("score on train: "+ str(model.score(X_train_normalized, y_train)))

    # Random Forest
    elif classifier_name == "rf":
        # n_estimators = number of decision trees
        rf = RandomForestClassifier(n_estimators=30, max_depth=9)
        model = rf.fit(X_train_normalized, y_train)

        print("score on test: " + str(model.score(X_test_normalized, y_test)))
        print("score on train: "+ str(model.score(X_train_normalized, y_train)))

    else:
        pass

    logging.info("Saving the model in saving_model_path")
    # path to save model on the disk
    saving_model_path = os.path.join(model_path, 'trainedmodel.pkl')
    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(saving_model_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train_model("rf")
