#This code comes from examples in the data splitting lecture from the coursera course.

data(spam) #loads an example data file from kernlab
inTrain <- createDataPartition(y=spam$type,p = .75, list=FALSE) #used so that we have 75% of data to train the model
training <- spam[inTrain,] #creates a structure of all the randomly chosen data from above for training
testing <- spam[-inTrain,] #creates a structure of all of the left over data to use for testing
dim(training) #gives [#ofcases #ofvariables]

#splitting into k-folds to use for cross-validation
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=TRUE)
#list tells the whether to return each set of indeces of each fold as a list 
#returnTrain tells it whether or not to return the training data set itself. IF FALSE, it will return the test set.
sapply(folds,length) #returns the length of each fold list, to check that it worked. They should be close to the same.

folds[[1]][1:10] #This will tell you which elements appear in the first 10 elements of fold 1.

#resampling
#You can also resample using this code.THis resamples with replacement from the values.
#It's now possible the same elements are repeated in each fold.
folds <- createResample(y=spam$type, times=10, list=TRUE)


#Time slice sampling
#This is similar but now sampling using windows of continuous time slices.
tme <- 1:1000
folds <- createTimeSlices(y=tme, initialWindow=20, horizon=10)
#THis makes the first window the first 20 elements, followed by 10 in each subsequent slice.

names(folds) #This will give you the name of each, notice now that there are slices throughout therea re just 2 (train and test).

folds$test[[1]] #This should give you the elements the inputed slice.