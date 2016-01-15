#Iris data set has different species of Iris flowers.
data(iris)
library(ggplot2)

#building training and testing data sets.
inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)

training <- iris[inTrain,]
testing <- iris[-inTrain,]

library(caret)
modFit <- train(Species~., data=training, method="rf", prox=TRUE) #outcome is species, and ~. use all other variables as potential predictors.
modFit
#using Prox = TRUE provides a little bit more information for us.
#one variable is the tuning parameter(mtry) or the number of trees the model is going to build.

#looking at a specific tree.
getTree(modFit$finalModel, k=2)#In this case it's the 2nd tree
#each row is a particular split, columns are stats for that particular split (left daughter or split node, right daughter, variable number it's split on, value point of split, (im not sure of status), prediction of model)

#can also look at the center of the class predictions:
irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox) #Class centers finds the center of each class for the training data set. Also using prox which we asked for before.
irisP <- as.data.frame(irisP); #create centers data set
irisP$Species <- rownames(irisP) #create species data set
p <- qplot(Petal.Width, Petal.Length, col=Species, data=training) #Plotting petal length by petal width colored by species
p + geom_point(aes(x=Petal.Width, y=Petal.Length, col=Species), size=5, shape=4, data=irisP) #this plots the centers of the data set for each group

#predicting new values
pred <- predict(modFit, testing);#pass to predict, our model fit and the testing data set.
testing$predRight <- pred==testing$Species #This creates a new variable asking whether or not the prediction was correct.
table(pred, testing$Species) #table of prediction vs. species.
#Notice 2 were wrong, but the model was still highly predictive.

#Taking a look at which points were missed.
qplot(Petal.Width, Petal.Length, colour=predRight, data=testing, main="newdata Predictions")
#Notice the 2 incorrect points were right on the border of the class clouds