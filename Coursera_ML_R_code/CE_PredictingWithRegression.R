#using a data set to analyze geyser eruptions. Overall looking at how waiting time until eruption predicts eruption duration.
library(caret); data(faithful); set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting, p = 0.5, list=FALSE)
trainFaith <- faithful[inTrain,]; testFaith <- faithful[-inTrain,]

head(trainFaith)#looking at training set.

#plot to look at waiting time until eruption by the duration of the eruption.
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")
#by inspection there seems to be a linear relationship.

#using lm command to fit a linear model:
lm1 <- lm(eruptions ~ waiting, data=trainFaith) #eruptions is outcome variable. ~ means as a function of (everything to right).
summary(lm1)
#For the output we want to look at Estimates. Intercept estimate (-1.79) is intercept or b0 from formula in notebook. 
#Estimate of waiting time (.07) is b1 in formula from notebook.
#So new EDi add -1.79 + .07(new waiting time) to predict eruption duration. This is taken from given formula found in notebook and slids.

#Now plotting the fitted values of the linear fit (lm1$fitted):
lines(trainFaith$waiting, lm1$fitted, lwd=3) #Plotting line vs predictor (trainFaith$waiting)

#predicting a new value:
#using coef command to represent the coffeicients (b0 and b1 from formula)
coef(lm1)[1] + coef(lm1)[2]*80 #first term is intercept(beta-hat0). second term is value fit for waiting time (beta-hat1). 80 represents new waiting time to predict eruption time from.

#Instead of doing it manually we can also use the predict function:
newdata <- data.frame(waiting=80)
predict(lm1, newdata) #lm1 is fitted model from training set.
#also outputs prediction for new waiting time value.

#Since we've built the model on the training set we now want to see how it doesn on the test set.
par(mfrow=c(1,2))
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")
lines(trainFaith$waiting, predict(lm1), lwd=3)
plot(testFaith$waiting, testFaith$eruptions, pch=19, col="blue", xlab="Waiting", ylab="Duration")
lines(testFaith$waiting, predict(lm1, newdata=testFaith),lwd=3)

#Next we want to gather the training and test set errors
sqrt(sum((lm1$fitted-trainFaith$eruptions)^2)) #training set error (RMSE, root mean square error, measuring how close the fitted values are to the real values)
#calculating RMSE on test set.
sqrt(sum((predict(lm1, newdata=testFaith)-testFaith$eruptions)^2))#again using lm but now passing test data set and then subtracting actual values(testFaith$eruptions)
#** Test data set error is almost always larger than training set

#Can also calculate prediction intervals
#new predictions for test data set based upon training data set:
pred1 <- predict(lm1, newdata=testFaith, interval="prediction")#addition of prediction interval input
#ordering values for test data set:
ord <- order(testFaith$waiting)
#plotting test waiting times vs eruption times
plot(testFaith$waiting, testFaith$eruptions, pch=19, col="blue")
#adding lines to create a region that we expect the prediction values to land.
matlines(testFaith$waiting[ord], pred1[ord,], type="l",col=c(1,2,2), lty=c(1,1,1), lwd=3)
#represents range of possible predictions you may get.

#this can all be done in the caret package:
#same process but easier, eruptions is the output. Waiting time is the predicton. build on training set. Method is linear model.
modFit <- train(eruptions ~ waiting, data=trainFaith, method="lm")#eruptions is outcome, waiting time is predictor, looked at trained on the training set and method is linear model.
summary(modFit$finalModel)
#notice the estimates should be the same as the long version.
