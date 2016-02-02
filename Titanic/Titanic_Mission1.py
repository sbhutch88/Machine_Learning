# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:55:27 2016

@author: Steve
"""

import pandas as pd
import numpy as np

#creates pandas data frame and assigns it to "titanic"
titanic = pd.read_csv("train.csv")

################Cleaning the Data

#printing first 5 rows of data frame to see what we have.
print(titanic.head(5))

#Code to get some basic stats for each numeric variable.
print(titanic.describe())
#notice age only has a count of 714 meaning missing values
print(titanic["Age"])


#One way to clean the data and fill in missing values is to replace NA's with
#the median values of the column.
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#Notice how the NAs were replaced.
print(titanic["Age"])


#Some of the columns are non-numeric and are not very useful for our model, however
#The sex column may be useful and can be coverted to a numeric variable.

#printing the unique elements of the Sex variable:
print(titanic["Sex"].unique())

#Changing all of the "male" to 0 (loc is location)
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0 #(loc is location)
#Changing all of the "female" to 1.
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#Also can adjust the Embarked variable:
print(titanic["Embarked"].unique())
#notice first there are missing variables

#let's replace the missing elements with the most common element.
titanic["Embarked"].value_counts()
#notice this is "S" representing "Southampton"
titanic["Embarked"] = titanic["Embarked"].fillna("S")

#now replacing S with 0, C with 1, and Q with 2
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


#####################Onto Machine Learning (Linear Regression)
#** Because I'm using XY, I installed with pip from CMD.
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:]) #(iloc is integer location based on indexing)
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
#The above created predictions on our training set. Output should be numpa arrays
    
# The predictions are in three separate numpy arrays (because of the 3 folds).  
# Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
# These adjustments will allow us to compare to actualy binary target ("Survived")
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

#array of booleans of whether prediction equals target
outcomes = np.equal(predictions,titanic["Survived"])

#accuracy varaible taking the sum of outcomes (True = 1) dividing by total num of elements.
accuracy = np.sum(outcomes)/len(predictions)

#Another way to do the same in one line:
#accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

##############################Logistic Regression
#Really because the target is a binary variable, logistic regression is the correct
#way to approach this problem. (because "Survived" is not continuous)

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

##Notice the accuracy is only slightly better using logistic regression than linear regression.


####################Cleaning the Test set.
#in order to do this we must now complete all of the same cleaning steps on our test set, and use the same model.

titanic_test = pd.read_csv("test.csv")

#Changing missing ages to the median age *** USING MEDIAN FROM TRAINING
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
#There happens to be some NA values for the "Fare variable in the test set only. Adjust to median
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

#Changing all of the "male" to 0 (loc is location) and female to 1
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 #(loc is location)
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

#filling in the missing values in the "Embarked" column with "S"
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

#Making "Embark" numeric
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2


###################Creating a Kaggle submission file so our model can be tested.
# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)


