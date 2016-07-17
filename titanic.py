from __future__ import division
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import KFold 
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

import pandas 
import matplotlib.pyplot as plt
import numpy as np
import operator
import re


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona":10}

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]


titanic = pandas.read_csv("data/train.csv")
titanic_test = pandas.read_csv("data/test.csv")

predictors = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked', 'Title']

# Clean Titanic
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# Clean Titanic Test
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# Combining
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# Add FamilyId and Title
titles = titanic["Name"].apply(get_title)
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic["Title"] = titles
family_ids = titanic.apply(get_family_id, axis=1)
family_ids[titanic["FamilySize"] < 3] = -1
titanic["FamilyId"] = family_ids


titles = titanic_test["Name"].apply(get_title)
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids


# Selecting Best Selector
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
weight = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)), weight)
plt.xticks(range(len(predictors)), predictors, rotation="vertical")
plt.show()


# Linear Regression
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions=[]

for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions < .5] = 0

accuracy = 0
for i, value in enumerate(predictions):
    if value == titanic["Survived"][i]:
        accuracy = accuracy + 1
    
accuracy = accuracy/len(predictions)
print("Linear Regression accuracy is " + str(accuracy))


# Logistic Regression checking
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
alg.fit(titanic[predictors], titanic["Survived"])
predictions = alg.predict(titanic_test[predictors])
print("Logistic Regression accuracy is " + str(scores.mean()) )

# Random Forest
alg = RandomForestClassifier(random_state=1, n_estimators =150, min_samples_split=5, min_samples_leaf=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print("Random Forest accuracy is " + str(scores.mean()) )

# Ensemble KF
algorithms = [
        [
            GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
            ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked', 'Title']
            ],
        [
            RandomForestClassifier(random_state=1, n_estimators =150, min_samples_split=5, min_samples_leaf=1),
            predictors
            ]
        ]
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions=[]
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions=[]
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        test_predictions=alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0]*3+full_test_predictions[1])/4
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)
accuracy = 0
for i, value in enumerate(predictions):
    if value == titanic["Survived"][i]:
        accuracy = accuracy + 1
accuracy = accuracy/len(predictions)
print("Ensemble accuracy is " + str(accuracy))

# Ensemble
full_predictions = []
for alg, predictors in algorithms:
    alg.fit(titanic[predictors], titanic["Survived"])
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
predictions = (full_predictions[0]*3 + full_predictions[1])/4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)

# Submission
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("result.csv", index=False)
