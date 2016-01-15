#Extensive description of Boosting in notebook.

library(ISLR); data(Wage);
library(ggplot2)
library(caret);

Wage <- subset (Wage, select=-c(logwage)) #removes variable we're trying to predict (logwage)

#Creating training and testing sets.
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

#fit the model
#modeling wage as a function of all remaining variables (using ~ .) to boost with trees (gbm).
#Using verbose = FALSE just limits the extensive output from GBM.
modFit <- train(wage ~ ., method="gbm",data=training, verbose=FALSE)
print(modFit)
#notice there are different numbers of trees and interaciton.depths.

#Plotting our model fit predicting the test set on the x-axis, wage on the y-axis.
qplot(predict(modFit,testing),wage,data=testing)
#notice the generally decent prediction output.
#This took weak classifiers, and averaged them together with weights to get a stronger classifier.
#See notes for how this was done.