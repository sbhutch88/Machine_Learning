# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:36:03 2016

@author: Steve
"""

#Mission two is a way to improve the performance of our algorithm using better ML techniques.

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

titanic = pd.read_csv("train.csv")

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

###################################Cleaning the data so we can predict
#One way to clean the data and fill in missing values is to replace NA's with
#the median values of the column.
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#let's replace the missing elements with the most common element.
titanic["Embarked"].value_counts()
#notice this is "S" representing "Southampton"
titanic["Embarked"] = titanic["Embarked"].fillna("S")

#now replacing S with 0, C with 1, and Q with 2
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

#Changing all of the "male" to 0 (loc is location)
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0 #(loc is location)
#Changing all of the "female" to 1.
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#################################################################

#Creating a working model

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

#create cross validation scores of the model for prediction of whether or not they survived.
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)

#final printed accuracy.
print(scores.mean())

############## Tuning the model to improve performance
#One option for a better model would simply be more trees
#Another option would be to tweak the splits and the leafs.
#increasing the minimum number of samples for a split and leaf should reduce overfitting because it won't incorporate all of the deep
#level quirks of the training set.

#making these adjustments should improve the model's performance and generalizability.
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

#create cross validation scores of the model for prediction of whether or not they survived.
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())
#Notice performance improved

########################################## Generating new features

# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# The .apply method generates a new series
#The function after the colon creates the information for the new pandas column.
#In this case it is taking the length of the person's name which apparently pertains to how rich they were.
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

#See this using:
print(titanic.head(5))

#We can aslo extract their title (ex. Master, Mrs. etc.)
import re

### A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
    
# Verify that we converted everything.
print(pd.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles

### Creating family ID's since it's likely which family you are in could predict survival

import operator

# A dictionary mapping family name to id
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

# Get the family ids with the apply method
family_ids = titanic.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[titanic["FamilySize"] < 3] = -1

# Print the count of each unique id.
print(pd.value_counts(family_ids))

titanic["FamilyId"] = family_ids

##############################Finding the best features
#We will use univariate feature selection to understand which columns correlate best with target ("Survived")

from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

#New algorithm
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


################################ Ensembling Multiple Models
#Ensembling is a great way to improve performance. Generally the more diverse (using different predictors) the models are,
#the better ensembling will work. One great approach is to ensemble random forests with regression.
#** Note the models must be about the same in terms of accuracy, otherwise the final result will be worse.

#In this example we'll train logistic regression on the most linear predictors (the ones that have a linear ordering, and some correlations to the target)
# and gradient boosted tree on all of the predictors.

#Boosting involves training decision trees one after another, and feeding the errors from one tree into the next. This is easy to overfit, so we will limit the tree count
# to only 25 trees, and keep the depth small in terms of splits and leafs.

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
import numpy as np

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
    
    # Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

#Compares predictions to target
outcomes = np.equal(predictions,titanic["Survived"])

# Likelihood of accuracy
accuracy = sum(outcomes) / len(predictions)
print(accuracy)


##################################### PREDICTING ON THE TEST SET

titanic_test = pd.read_csv("test.csv")

############cleaning just like the training set (addition of Fare because of NAs).
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
#let's replace the missing elements with the most common element.
titanic_test["Embarked"].value_counts()
#notice this is "S" representing "Southampton"
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
#There happens to be some NA values for the "Fare variable in the test set only. Adjust to median
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

#now replacing S with 0, C with 1, and Q with 2
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

#Changing all of the "male" to 0 (loc is location)
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 #(loc is location)
#Changing all of the "female" to 1.
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

############ New Predictor columns just like in the training set.

# First, we'll add titles to the test set.
titles = titanic_test["Name"].apply(get_title)
# Get all the titles and print how often each one occurs.
print(pd.value_counts(titles))
# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
#**** I really can't figure out why but this loop must be entered separately and hit enter.
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
# Check the counts of each unique title.
print(pd.value_counts(titanic_test["Title"]))
# Now, we add the family size column.
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]
# Now we can add family ids.
# We'll use the same ids that we did earlier.
print(family_id_mapping)
family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids

# Generating name length columns.
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))


#################### Making a submissiont to Kaggle
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
# Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

predictions = predictions.astype(int)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)




####Possible next steps suggested by dataquest:
'''
There's still more work you can do in feature engineering:

Try using features related to the cabins.
See if any family size features might help -- do the number of women in a family make the whole family more likely to survive?
Does the national origin of the passenger's name have anything to do with survival?
There's also a lot more we can do on the algorithm side:

Try the random forest classifier in the ensemble.
A support vector machine might work well with this data.
We could try neural networks.
Boosting with a different base classifier might work better.
'''