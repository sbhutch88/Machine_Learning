

library(caret);
library(kernlab);

#Setting up data sets (more on CE_DataSplitting.r)
data(spam) #loads an example data file from kernlab
inTrain <- createDataPartition(y=spam$type,p = .75, list=FALSE) #used so that we have 75% of data to train the model
training <- spam[inTrain,]
testing <- spam[-inTrain,]

modelFit <- train(type~., data = training, method = 'glm')

#Different options for training the model:
function(x,y, method="rf", preProcess=NULL, ..., weights=NULL,
         metric=ifelse(is.factor(y), "Accuracy", "RMSE"), maximize=ifelse(metric=="RMSE",FALSE, TRUE)
         , trControl=trainControl(), tuneGrid=NULL, tuneLength=3)
NULL
# preProcessing will be talked about later.
#weights can be adjusted, may be useful if training set is unbalanced.
#for categorical variables the default metric is accuracy, whereas for continuous it's root mean squared error
#trControl needs a call to trainControl... examples can be seen with call args(trainControl). Described further on slide 5 of training options lecture.