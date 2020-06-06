

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

## Missing value handler with sklearn.imput library
#Using mean: It work only for numeric data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(pf[num_cols])
pf[num_cols] = imputer.transform(pf[num_cols]) 

cf = pd.DataFrame()
cf = dataset[cat_cols]




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
 

#7 features  
final_features = ['sg', 'al', 'bgr', 'sc', 'hemo', 'pcv', 'rc']
#8 features  
#final_features = ['htn', 'sg', 'al', 'bgr', 'sc', 'hemo', 'pcv', 'rc']
#10 features  
#final_features = ['htn', 'dm', 'sg', 'al', 'bgr', 'sc', 'sod', 'hemo', 'pcv', 'rc']
#11 features  
#final_features = ['sg', 'al', 'rbc', 'bgr', 'bu', 'sc', 'sod', 'hemo', 'pcv', 'rc', 'htn']
#14-a features  
#final_features = ['htn', 'dm', 'age', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'hemo', 'pcv', 'wc', 'rc']
#14-b features  
#final_features = ['htn', 'dm', 'age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv','rc']
#15 features  
#final_features = ['htn', 'dm', 'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'hemo', 'pcv', 'wc','rc']
#18 features  
#final_features = ['htn', 'dm', 'appet', 'pe', 'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot','hemo', 'pcv','wc','rc']
#24 features  
#final_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']



  
optimal_dataset = pd.DataFrame()
optimal_dataset = features[final_features]
### Split the data into (1) set of Target and (2) set of features
target1 = features['class']

#perform training and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(optimal_dataset, target1, test_size=0.3, random_state=100)
#print(len(X_train), len(X_test), len(y_train), len(y_test))


# KNN
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=11) 
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test).round()
y_pred = abs(y_pred)

# Model Evaluation metrics 
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score 
print('Accuracy Score : '  + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : '    + str(recall_score(y_test,y_pred)))
print('F1 Score : '        + str(f1_score(y_test,y_pred)))  

# Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))


#find the optimum value of K by using the k-fold cross-validation.

#-----------------------------------------------

print("-----------------------optimization------------------------")

#We’ll use 10-fold cross-validation on our dataset using a generated 
#list of odd K’s ranging from 1–50.
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

# Creating odd list K for KNN
neighbors = list(range(1,50,2))

# empty list that will hold cv scores
cv_scoresAccuracy = [ ]

#perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,X_train,y_train,cv=10 ,scoring = "accuracy")
    cv_scoresAccuracy.append(scores.mean())

#Now you can get the optimal value of K by Calculating the misclassification error
# Changing to mis classification error
mse = [1-x for x in cv_scoresAccuracy]
# determing best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors is {}".format(optimal_k))


knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test).round()
y_pred = abs(y_pred)

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : '  + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : '    + str(recall_score(y_test,y_pred)))
print('F1 Score : '        + str(f1_score(y_test,y_pred)))



# Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))



