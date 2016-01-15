#Using wages data set from ISLR package.
library(ISLR)
library(ggplot2)
library(caret)

data(Wage);
Wage <- subset(Wage, select=-c(logwage)) #THis changes to a subset of wage of everything except variable we are going to predict (logwage)
summary(Wage)
#Notice the data includes only males from the midatlantic.

#Building training and test sets.
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]; 
testing <- Wage[-inTrain,]
dim(training); dim(testing)

#first looking at how the variables are related to each other.
featurePlot(x=training[,c("age", "education", "jobclass")], y=training$wage, plot="pairs")
#Notice the separate groups formed on the last column. Maybe we can use these variables as predictors.

#plotting age vs. wage
qplot(age, wage, data=training) #trend found but also group of outliers having a high wage.

#plotting age vs. wage and coloring by job class.
qplot(age, wage, colour=jobclass, data=training)
#reveals that most of the top outliers are represented by the information job class, and maybe would be a good predictor

#plotting age vs. wage coloring by education
qplot(age, wage, colour=education, data=training)
#reveals that advanced degree is highly represented in top group
#** Maybe some combination of jobclass and education would be a good predctor of wage.


#fitting a linear model (see notes for formula):
modFit <- train(wage ~age + jobclass + education, method="lm", data=training)#train from caret, wage is outcome, 
#~ says formula on right will be used to predict variable on left. Because jobclass and education are class, the train function 
#creates indicators (as seen in formula from notes) automatically. 
finMod <- modFit$finalModel
print(modFit)
#Notice 10 predictors because of the multiple levels of some predictors.

#looking at some diagnostic plots:
plot(finMod, 1, pch=19, cex=0.5, col="00000010")
#x-axis is model predictions from training set vs. the residuals(amount of variance left over after fitting model)
#fit of residual is difference between our model prediction and real values. Hence a horizantal line at 0 is perfect.
#Still notice a few outliers remain, and are labled in the plot, we may want to see if there are further predictors that could explain them.

#coloring by variables not used in the model (fitted model vs. residuals):
#again optimal is data lies on 0 because it's the difference between fitted values and real values.
qplot(finMod$fitted, finMod$residuals, colour=race, data=training)
#plotting by race reveals maybe some outliers can be explained by race
#This technique can be used after the model fit to identify potential trends.

#Can also plot the fitted residuals vs the index:
#Index is simply which row of the data set you're looking at.
plot(finMod$residuals, pch=19)
#all of the high residuals seem to be appearing at the highest row numbers. Can also see a trend with respect to row number.
#This trend suggests there is another important variable not accounted for. THis is because the order of the data shouldn't matter and the fact that
#it does reveals there is some significance to the order the rows are sorted by (maybe age, time, or something like that.)

#plotting the wage values in test set vs predicted values in test set:
pred <- predict(modFit, testing)
qplot(wage, pred, colour=year, data=testing)
#ideally there would be a perfect relationship (line on 45 degree line)
#This exploration is coloring by year collected to see how the model broke down by collection time.
#** Once you do this exploration you cannot go back and adjust test set. This is a post-hoc indication of whether your model worked or not.

#IF you want to use all covariates:
modFitAll <- train(wage ~., data=training, method="lm") # ~. says predict with all variables in the data set.
pred <- predict(modFitAll, testing)
qplot(wage, pred, data=testing)
#This fit is a bit better, and an option if you don't want to do some model selection in advance.