library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)

training <- spam[inTrain,]
testing <- spam[-inTrain,]

#calculating absolute values of correlations of all predictors
M <- abs(cor(training[,-58])) #leaving out the 58'th column which is the outcome to look at all other predictor variables
diag(M) <- 0 #getting rid of all correlations of 1 (because each variable is correlated with itself). Making them 0
which(M > 0.8, arr.ind=T) #Which variables have a high correlation (>.8)?
#2 variables were found to highly correlate with one another (both numebrs so likely a phone number or something.)

#looking at the two columns from the spam data set revealed from the code above.
names(spam)[c(34,32)]
plot(spam[,34],spam[,32])
#plot reveals as expected almost perfect correlation.

#Next we could create new variables and rotate the plot:
X <- 0.71*training$num415 + 0.71*training$num857 #Sum of 2 variables
Y <- 0.71*training$num415 - 0.71*training$num857 #difference of 2 variables
#** Not sure why .71??
#Reveals most of the variability is across the x-axis not hte y-axis. Most have a y value of 0.
#This process tells us that adding the variables together provides more information than subtracting.
#Because of this we may want to use the sum of the two variables as a predictor
plot(X,Y)

#Principal components in R
smallSpam <- spam[,c(34,32)] #These are the 2 highly correlated variables
prComp <- prcomp(smallSpam) #doing principal components on this new small 2 variable data set.
plot(prComp$x[,1],prComp$x[,2])
#Notice how similar this plot looks to the steps taken above.
#The key here is principal components is much easier when trying to reduce a large number of variables.

#can also look at rotation matrix:
prComp$rotation
#** THIS EXPLAINS THE .71 used above.
#Column 1 here is the summation of the two variable above, column to is the difference

#PCA on SPAM data:
typeColor <- ((spam$type=="spam")*1 + 1) #black if not spam, red if spam
prComp <- prcomp(log10(spam[,-58]+1))#calculates principal components on entire data set. Using the log transformation and +1 will make the data look more
#gaussian and display as more sensible.
#PR1 explains the most data, PR2 the second most, PR3 the third most and so forth.
plot(prComp$x[,1],prComp$x[,2], col=typeColor, xlab="PC1", ylab="PC2")
#As can be seen from this plot, the spam messages(red) tend to have higher values of PC1 than non-spam

#This can also be done in caret:
preProc <- preProcess(log10(spam[,-58]+1),method="pca", pcaComp=2) #notice this is preprocess function with "pca" method.Need to input number of principal components to compute.
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1], spamPC[,2], col=typeColor)
#again notice the differences between Principal components 1 and 2.

#can also create training predictions:
preProc <- preProcess(log10(spam[,-58]+1),method="pca", pcaComp=2) 
trainPC <- predict(preProc, log10(training[,-58]+1))
modelFit <- train(training$type ~., method="glm", data=trainPC) #fitting a model that relateds the training variable to the principal component.

#Now using test data set:
testPC <- predict(preProc, log10(testing[,-58]+1)) #notice we must use same PC's calculated for training set
#This can be done by passing training preprocessing variable, and transformation of the new training data.
confusionMatrix(testing$type, predict(modelFit, testPC)) #now predicting using original model fits on the test data.
#again confusion matrix will provide accuracy stats.

#Another option is to not use the predict function separately but build it directly into the training exercise.
modelFit <- train(training$type ~., method="glm", preProcess="pca", data=training)
confusionMatrix(testing$type, predict(modelFit, testing)) #Now just pass testing data set
#notice the results are very similar to other strategy above
#** This is easier to do, the above version was to better understand whate we were doing.