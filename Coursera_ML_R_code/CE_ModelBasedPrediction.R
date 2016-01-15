#VERY extensive lession on the slides and notes in my notebook, with this brief simulation.

data(iris);
library(ggplot2)
library(caret)
names(iris)

table(iris$Species) #3 different species of flower we want to predict.

inTrain <- createDataPartition(y=iris$Species,p=.7,list=FALSE)

training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training);dim(testing)

modlda = train(Species ~ .,data=training,method="lda") #linear discriminant model
modnb = train(Species ~ .,data=training, method="nb") #naive bayes classification
plda = predict(modlda,testing); pnb = predict(modnb,testing) #predicting values on the test set
table(plda,pnb)
#Notice each time it runs it fits a bit differently.
#Overall works very well with a few small differences between models.
#Sometimes they will both be off, sometimes they will disagree.

#plotting to compare results
equalPredictions = (plda==pnb)
qplot(Petal.Width,Sepal.Width,colour=equalPredictions,data=testing)
#Notice the red points represent the points that were classified differently by the different models.