"""
Classification problem to predict churn for a telecom dataset.
"""

#Read dataset
import os
import pandas as pd
import numpy as np
import seaborn as sns

os.chdir ("C:/R")
data = pd.read_csv("telecom.csv")

#Check loaded data
data.head()
data.shape
data.info()

#Exploratory Data Analysis
data.isnull().values.any()
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = "viridis")

#Check for imbalanced dataset
data.churn.value_counts()
sns.set_style("whitegrid")
sns.countplot(x = "churn", data = data, palette = "RdBu_r")

#Perform Lable Encoding
from sklearn.preprocessing import LabelEncoder
lblEncoder = LabelEncoder()
data.churn = lblEncoder.fit_transform(data.churn)
data["international plan"] = lblEncoder.fit_transform(data["international plan"])
data["voice mail plan"] = lblEncoder.fit_transform(data["voice mail plan"])

#Perform Mean Encoding on Categorical Variable
data.state.value_counts()

states = data.state.sort_values().unique()
states.shape

mean_churn = data.groupby(["state"])["churn"].mean()
mean_churn.shape

myDict = {}
for index in range( len(states)):
    myDict.update({states[index] : mean_churn[index]})

def meanChurn(val):
    return(myDict[val])

data["state"] = data.state.apply(lambda X : meanChurn(X))

#Drop Phone number variable
data = data.drop("phone number", axis = "columns")

#Divide dataset into dependent and independent sets
X = data.drop("churn", axis = "columns")
Y = data["churn"]

#Handle imbalanced dataset
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 42)
X_res, Y_res = smk.fit_sample(X, Y)
print(X_res.shape, Y_res.shape)

#divide dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, train_size = 0.7, random_state = 0)

#Build Model
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, Y_train)
model

#Hyperparameter Tuning using RandomizedSerchCV
from sklearn.model_selection import RandomizedSearchCV

new_model = XGBClassifier()
new_params = {"learning_rate" : [0.05, 0.10, 0.15, 0.2, 0.25, 0.3],
              "max_depth" : [3, 4, 5, 6, 7, 8, 10, 12, 15],
              "min_child_weight" : [1, 3, 5, 7, 9],
              "gamma" : [0, 0.1, 0.2, 0.3, 0.4],
              "colsample_bytree" : [0.3, 0.4, 0.5, 0.7] }


randomSearch = RandomizedSearchCV(new_model,
                              param_distributions = new_params,
                              n_iter = 5,
                              scoring = "roc_auc",
                              n_jobs = -1,
                              cv = 5,
                              verbose = 3)
randomSearch.fit(X_train, Y_train)
randomSearch.best_estimator_
randomSearch.best_params_

new_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0,
              learning_rate=0.25, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

new_model.fit(X_train, Y_train)

#Check model accuracy
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(new_model, X_res, Y_res, cv = 10)
accuracy.mean()