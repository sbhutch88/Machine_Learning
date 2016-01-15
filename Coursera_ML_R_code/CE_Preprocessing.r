library(caret);
library(kernlab);
library(RANN)

#Setting up data sets (more on CE_DataSlicing.r)
data(spam) #loads an example data file from kernlab
inTrain <- createDataPartition(y=spam$type,p = .75, list=FALSE) #used so that we have 75% of data to train the model
training <- spam[inTrain,]
testing <- spam[-inTrain,]

#Histogram of number of capital letters in a row
hist(training$capitalAve, main="", xlab= "ave. capital run length")
#This shows us how very skewed the data is, and may suggest we should preprocess.

#Also look at the mean and SD, revealing how variable the data set is.
mean(training$capitalAve); sd(training$capitalAve)

#standardizing, I believe this is exactly the same as a z-score
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve)) / sd(trainCapAve)
#Now the mean is 0 and SD is 1

#When we want to look at testing set, we must use standardized training set:
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve)) / sd(trainCapAve)
#Notice the mean and SD won't be exactly 0 and 1, but hopefully close.

#Also can use preprocess function from caret:
preObj <- preProcess(training[,-58], method=c("center", "scale"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve #Excludes 58 because it's the outcome we care about
#This centers and scales every variable. This  should do exactly the same thing as above.
mean(trainCapAveS);sd(trainCapAveS) #Again notice the mean is 0 and SD is 1.

#And just like before we can use the training set transformation to predict the testing set.
testCapAveS <- predict(preObj, testing[-58])$capitalAve
mean(testCapAveS);sd(testCapAveS)

#Another option is to send in the preprocessing function as an argument to the model:
modelFit <- train(type ~., data=training, preProcess=c("center","scale"), method="glm")
#The preprocess function here will center and scale all predictors before entering them into the model.

#################Other types of transformations

#Boxcox transformations take continuous data and try to make them look like normal data
#Done using parameters estimating maximum liklihood. This transormation is good for skewed data,
#however because it's meant for continuous data, it doesn't do well with repeated elements. In this case 0 is 
#often repeated and the histogram below should show a fairly normal curve, but does transform the 0s well.
preObj <- preProcess(training[,-58], method=c("BoxCox"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
par(mfrow=c(1,2)); hist(trainCapAveS); qqnorm(trainCapAveS)


#We can also impute data when there are many missing values.
##K nearest neighbors imputation:
set.seed(13343)

#make some values NA
training$capAve <- training$capitalAve #creates a new capave variable
selectNA <- rbinom(dim(training)[1], size=1, prob=0.05)==1
training$capAve[selectNA] <- NA
#now capave is just like the capitalAve variable except it has a subset of variables missing.

#Impute and standardize
#Again using preprocess can impute using k nearest neighbors imputation:
preObj <- preProcess(training[, -58], method="knnImpute")
capAve <- predict(preObj, training[,-58])$capAve
#finds the k (ex.10) nearest data vectors that are most like the data vector with the missing value, and average 
#the values of the variable thats missing, and impute them at that position.
#Now we predict on our training set all the new values including the ones that have beein imputed
#with the k nearest neighbors imputation algorithm

#Standardize true values
#Same standardization method we used before (z-scoring)
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth)) / sd(capAveTruth)

#Looking at the data
quantile(capAve - capAveTruth) #comparison between imputed values and values that were truly there before making them NAs
#This above will tell us how much of a difference the imputing made.
#Because they are all very close to 0, the imputation worked relatively well.

#Looking at the same as above but only
quantile((capAve - capAveTruth)[selectNA])

#Looking at the ones that aren't NA only.
quantile((capAve - capAveTruth)[!selectNA])
#Comparing the above to quantile outputs will show the influence of the NAs