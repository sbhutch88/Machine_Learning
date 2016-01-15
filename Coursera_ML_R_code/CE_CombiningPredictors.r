#This code shows us how to combine different models on the same data set.

library(ISLR)
library(ggplot2)
library(caret)
Wage <- subset(Wage, select=-c(logwage)) #leave out logwage, since we will use as predictor.

#Create a building data set and validation set

inBuild <- createDataPartition(y=Wage$wage, p=.7, list=FALSE)
validation <- Wage[-inBuild,]; buildData <- Wage[inBuild,]

inTrain <- createDataPartition(y=buildData$wage, p=.7, list=FALSE)
training <- buildData[inTrain,]
testing <- buildData[-inTrain,]

#Looking at sizes of each data set.
dim(training)
dim(testing)
dim(validation)

#building 2 different models
#Linear model built on training data to predict wage.
mod1 <- train(wage ~., method='glm', data=training)
#random forests model to same data set.
mod2 <- train(wage ~., method = "rf", data = training, trControl = trainControl(method="cv"),number=3)

#plot predictions vs. each other.
pred1 <- predict(mod1,testing);
pred2 <- predict(mod2,testing);
qplot(pred1,pred2,colour=wage, data=testing)
#notice they are close, but don't exactly agree.

#Fitting a model that combines predictors
predDF <- data.frame(pred1,pred2,wage=testing$wage)#build new data set based off of both models predictions
combModFit <- train(wage ~., method="gam", data=predDF) #fitting new regression model that relates wage variable to two predicitons. 
combPred <- predict(combModFit,predDF)#predicting from the combined data set on new samples.

#Testing errors of all 3 models:
sqrt(sum((pred1-testing$wage)^2))
sqrt(sum((pred2-testing$wage)^2))
sqrt(sum((combPred-testing$wage)^2))
#Notice error is best for combined predictor

#Since test set was used to blend 2 models together, we will want to test on new validation set.
pred1V <- predict(mod1,validation) #prediction of model 1 on validation set
pred2V <- predict(mod2,validation) #prediction of model 2 on validation set
predVDF <- data.frame(pred1=pred1V,pred2=pred2V) #data frame containing above 2 predictions.
combPredV <- predict(combModFit,predVDF) #predict using combined model on predictions of the validation data set.
#covariates being passed to the model, are the predictions from the two different prior models.

#Testing errors of all 3 models on validation set:
sqrt(sum((pred1V-validation$wage)^2))
sqrt(sum((pred2V-validation$wage)^2))
sqrt(sum((combPredV-validation$wage)^2))
#Notice error is best for combined predictor (actually was a bit higher than model 1 in my case )