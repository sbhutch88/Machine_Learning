#This example comes from the first lecture of week 2 in the Coursera course. Seems to work well, and I beleive each of the subsequent
#lectures will go more in depth into each of these steps? If so I'll comment the name of the new file in here of where I wenth through each one.

library(caret);
library(kernlab);

#Setting up data sets (more on CE_DataSplitting.r)
data(spam) #loads an example data file from kernlab
inTrain <- createDataPartition(y=spam$type,p = .75, list=FALSE) #used so that we have 75% of data to train the model
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)

#Fitting model on training set
set.seed(32343) #this will dissallow the function to regenerate random numbers each time, and will keep the same random number generation.
modelFit <- train(type~., data = training, method = 'glm') #using a glm model, but others can be input as well. For some reason this isnt'
#apparantly the GLM bootstraps by default an corrects for the bias that may come from bootstrapping.
modelFit
modelFit$finalModel #outputs the fitted models

#Predict using testing data set
predictions <- predict(modelFit, newdata=testing)
predictions

#Evaluating the model performance
confusionMatrix(predictions,testing$type) #looking at confusion matrix and stats. (includes sensitivity and specificity)