library(kernlab); data(spam)

#2 steps: 
  #1. Choosing features most useful for prediction.
  #2. Transforming Tidy Covariates EX:
spam$CaptialAveSq <- spam$capitalAve^2

library(ISLR); library(caret); data(Wage);
#Creating training and test sets.
inTrain <- createDataPartition(y=Wage$wage, p=.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

table(training$jobclass)

#Turning qualitative factor variables into quantitative dummy variables
dummies <- dummyVars(wage ~ jobclass, data=training) #wage = outcome, jobclass = predictor
head(predict(dummies, newdata=training)) 
#Noticed the dummy coding as 1's and 0, making them quantitative variables

#Removing zero covariates (nzv = near zero variable)
nsv <-nearZeroVar(training, saveMetrics=TRUE) #caret function to find the predictors with very little variability that won't be useful.
nsv#(nzv = near zero variable)

#If you aren't using linear regression but instead want to fit curvy lines
library(splines)
#bs function will create a polynomial variable
bsBasis <- bs(training$age, df=3) #3rd degree polynomial in this case.
#output 3 variables. First column: age, 2nd column: age^2 (fitting a quadratic fit), 3rd column: age^3 (cubic relationship)
bsBasis
#This can also be done using the gam method in caret

#EX. fitting a linear model
lm1 <- lm(wage ~ bsBasis, data=training)#wage is outcome, ~ telss us what we're predicting with. In this case we're passing the polynomial model (age, age^2, age^3)
#plotting age data vs. wage data
plot(training$age, training$wage, pch=19, cex=0.5)
#because of curvilinear relationship, we can plot age and predicted values of model including curvy polynomial
points(training$age, predict(lm1, newdata=training), col="red", pch=19, cex=0.5)

#Now on the Test set we must predict the same variables
#predicting from created bs variable a new set of values. These new values are plugged into prediction model on test set.
predict(bsBasis, age=testing$age)
