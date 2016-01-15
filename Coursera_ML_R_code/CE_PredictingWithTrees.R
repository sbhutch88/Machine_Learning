#Example of using decision trees with Iris data set.
#predicting species of flower given the physical properties
data(iris)
library(ggplot2)
library(caret)
library(rattle)
names(iris)

table(iris$Species)
#notice there are 50 of each flower species that we're trying to predict

inTrain <- createDataPartition(y=iris$Species, p=.7, list=FALSE)

training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training);dim(testing)

#petal widths vs. sepal widths colored by flower species
qplot(Petal.Width, Sepal.Width, colour=Species, data=training)
#Notice there are 3 distinct clusters. This indicates a classification model may be better than a linear model.

#using train from caret to fit the model.
modFit <- train(Species ~., method="rpart", data=training)#using all predictors, rpart is r's method for using regression and classification trees.
print(modFit$finalModel)

modFit()
#notice the output gives you: each node and how they're split. Also the probability for being in each class for each split.
#Ex. number 2: for petal length less than 2.45, all outcomes belong to setosa species.

#Plotting the classification tree:
plot(modFit$finalModel, uniform=TRUE, main="Classification Tree")#gives tree with no text
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8) #Adds text for splits.

#Nicer looking plot of the same thing (using rattle package)
fancyRpartPlot(modFit$finalModel)

#Predicting new values
predict(modFit, newdata=testing)
#preiction in class label since tree was built to predict by class



#Looking at stats:
confusionMatrix(testing$Species, predict(modFit, testing))