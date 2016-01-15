library(ElemStatLearn)
data(prostate)
str(prostate) #97 observations on 10 variables.

#First on slides the data is looked at (see CE_RegularizedRegression_dataVisualized.r)
#The main point of this display is to show as the number of predictors goes up (if we fit every possible regression model)
#, the linear training set error goes down (it has to) however the error on the testing set will eventually begin to go up
#as the model has over fit the training data.
#Tells us we may not want to include too many predictors in our model.
#Overall a greater complexity is good, only to a certain point.