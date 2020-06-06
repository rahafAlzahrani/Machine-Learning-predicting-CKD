# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import timeit
import warnings
warnings.filterwarnings('ignore') 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('chronic_kidney_disease1.csv')

dataset.info()

# first prepare list of target columns to be converted into numerical columns

num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Prepare a list of categorical columns
# -- You can prepare them manually like the above list or by using the code

# categorical columns - drop the columns (class and num_cols)

cate_cols = dataset.columns.drop('class').drop(num_cols)

# display categorical columns

cate_cols

dataset[num_cols] = dataset[num_cols].apply(pd.to_numeric, errors='coerce')
dataset['dm'] = dataset['dm'].str.strip()



dataset.info()

dataset.replace('?', np.nan, inplace=True)

dataset.head(10)


def missing_values_table(dataset):


    # Total missing values
    mis_val = dataset.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * dataset.isnull().sum() / len(dataset)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    # .iloc[:, 1]!= 0: filter on missing missing values not equal to zero
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(2)  # round(2), keep 2 digits
    
    # Print some summary information

    print("Your slelected dataframe has {} columns.".format(dataset.shape[1]) + '\n' + 
    "There are {} columns that have missing values.".format(mis_val_table_ren_columns.shape[0]))
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


#delete extra spaces from the data


# Take a copy of the original data; 
# For numeric data imputation, we will take a copy of numeric columns 
# 

pf = pd.DataFrame()
pf = dataset[num_cols]
pf.head(30)

## Missing value handler with sklearn.imput library
#Using mean: It work only for numeric data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(pf[num_cols])
pf[num_cols] = imputer.transform(pf[num_cols]) 

pf.head(30)
cf = pd.DataFrame()
cf = dataset[cat_cols]
cf.head(30)



# for each column, get value counts in decreasing order and take the index (value) of most common class
cdate = pd.DataFrame(cf)
cdate = cdate.apply(lambda x: x.fillna(x.value_counts().index[0]))
combineResult = pd.concat([cdate,pf], axis=1)

### Split the data into (1) set of Target and (2) set of features
target = dataset['class']
combineResultWithClass = pd.concat([target,combineResult], axis=1)
features = combineResultWithClass

### convert all string values into numberical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
features['rbc'] = labelEncoder_X.fit_transform(features['rbc'])
features['pc'] = labelEncoder_X.fit_transform(features['pc'])
features['pcc'] = labelEncoder_X.fit_transform(features['pcc'])
features['ba'] = labelEncoder_X.fit_transform(features['ba'])
features['htn'] = labelEncoder_X.fit_transform(features['htn'])
features['dm'] = labelEncoder_X.fit_transform(features['dm'])
features['cad'] = labelEncoder_X.fit_transform(features['cad'])
features['appet'] = labelEncoder_X.fit_transform(features['appet'])
features['pe'] = labelEncoder_X.fit_transform(features['pe'])
features['ane'] = labelEncoder_X.fit_transform(features['ane'])
features['class'] = labelEncoder_X.fit_transform(features['class'])


## Feature Selection
###  Recursievly Eleimination Method
#from sklearn.feature_selection import RFECV
#from sklearn.ensemble import RandomForestClassifier
##from sklearn.model_selection import StratifiedKFold
#rfc = RandomForestClassifier(max_depth= 8, min_samples_leaf= 5, min_samples_split= 2, n_estimators= 100)
#rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
#rfecv.fit(features, target)
#print('Optimal number of features: {}'.format(rfecv.n_features_))
#print(features.columns)
#best_cols = rfecv.support_
#print(best_cols)
#print(rfecv.ranking_)
 
#final_features = ['htn', 'sg', 'al', 'bgr', 'sc', 'hemo', 'pcv', 'rc']
final_features = ['sg', 'al', 'rbc', 'bgr', 'bu', 'sc', 'sod', 'hemo', 'pcv','rc' ,'htn']
#final_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
 
 
optimal_dataset = pd.DataFrame()
optimal_dataset = features[final_features]
### Split the data into (1) set of Target and (2) set of features
target1 = features['class']

#perform training and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(optimal_dataset, target1, test_size=0.3, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))


# SVM  -- it works only with numeric data, before run make sure the features is only numeric
from sklearn.svm import SVR  

clf = SVR().fit(X_train,y_train)
y_pred = clf.predict(X_test).round()
y_pred = abs(y_pred)
# Model Evaluation metrics 
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score 
print('Accuracy Score : '  + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : '    + str(recall_score(y_test,y_pred)))

print('F1 Score : '        + str(f1_score(y_test,y_pred)))  

#Logistic Regression Classifier Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))




#----------------------------------


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn import svm


grid_values ={
    'C': [0.1, 1, 10, 100, 1000],
    #'C': [1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']                            # -- set of values for gamma
}

# set the GridSearchCV

gs = GridSearchCV(
    estimator=SVR(kernel='rbf'), # --- Check SVR with rbf kernel only
    param_grid=grid_values,
    cv=5,               # --- set with Cross-validation=5,
    scoring='accuracy', # --- Check all parameters for high accuracy
    n_jobs=-1)          # --- Parallel processing,i.e., 2= use only 2 CPU, -1= use all available CPU


svc = svm.SVC()
gs = GridSearchCV(svc, grid_values)
gs.fit(X_train, y_train)
y_pred_acc = gs.predict(X_test).round()
y_pred_acc = abs(y_pred_acc)
print("------------------------")
print("Best parameters via GridSearch", gs.best_params_)


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

print('Accuracy Score : '  + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : '    + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : '        + str(f1_score(y_test,y_pred_acc)))

# SVR (Grid Search) Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : ' + str(confusion_matrix(y_test,y_pred_acc)))





