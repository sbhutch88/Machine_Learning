#Bagging is another word for bootstrap aggregating, and a way to smooth your data.
library(ElemStatLearn)
library(caret)
data(ozone, package="ElemStatLearn")
ozone <- ozone[order(ozone$ozone),] #This just orders the data by outcome (ozone variable)
head(ozone) #Data set summary.
#We're going to try and predict temperature as a function of ozone.

#Example:
ll <- matrix(NA,nrow=10, ncol=155) #matrix of 10 rows and 155 columns
for(i in 1:10){ #looping over 10 samples
  ss <- sample(1:dim(ozone)[1], replace=T) #resample data set - sample with replacement from entire data set
  ozone0 <- ozone[ss,] #resampled data set for this particular element of the loop (subset of data set corresponding to our random sample)
  ozone0 <- ozone0[order(ozone0$ozone),] #reorder by ozone variable
  loess0 <- loess(temperature ~ ozone, data=ozone0, span=0.2)#loess is a smooth curve you can fit to the data (similar to spline from modeling with linear regression).
  #above is fitting a smooth curve using temperature as the outcome and ozone as the predictor. Each time usingn the resampled data set with a common span(how smooth that fit will be).
  ll[i,] <- predict(loess0, newdata=data.frame(ozone=1:155))# predicting for every loess curve a new data set with the exact same values(1-155)
  #THe i'th row of the ll object is now the prediction from the loess curve of the i'th resample of the data ozone.
} #overall this code, resamples data 10 times, fits a curve to the data 10 times, and then averages the values.

#Plotting observed ozone values by observed temperature values
plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for(i in 1:10){lines(1:155, ll[i,], col="grey", lwd=2)} #grey lines equal the fit with one resampled data set (these may overcapture the variability)
lines(1:155, apply(ll,2,mean), col="red", lwd=2) #red line is the average of these fits (bagged loess curve)
#bagging will always have lower variability but similar bias.


#Can use caret to bag the data:
#-some models perform bagging for you, in train function consider method options:
  #-bagEarth, -treebag, -bagFDA


#alternatively you can bag any model you choose using the bag function:
predictors = data.frame(ozone=ozone$ozone) #make predictor it's own data frame
temperature = ozone$temperature #outcome variable
treebag <- bag(predictors, temperature, B=10, #predictors, outcome, subsamples
               bagControl = bagControl(fit=ctreeBag$fit, #this tells the function how to fit the model (fit = [could be a call to the train function in caret])
                                       predict=ctreeBag$pred, #Given a particular model fit, how can we predict new values(ex. could be a call to the predict function of a trained model)
                                       aggregate=ctreeBag$aggregate))#aggregate tells how to combine the results for example could be an average of the fits.
#This is advanced and documentation can be found at: http://www.inside-r.org/packages/cran/caret/docs/nbBag

#Custom bagging plots (Ozone vs. Temperature):
plot(ozone$ozone, temperature, col='lightgrey', pch=19)#grey dots represent observed values
points(ozone$ozone, predict(treebag$fits[[1]]$fit, predictors), pch=19, col="red") #red dots represent fit from a single decisional regresstion tree
points(ozone$ozone, predict(treebag, predictors), pch=19, col="blue") #This is fit from the bagged regression that is the average of all 10 fits


#Looking at the different parts of bagging:
ctreeBag$fit #produces the code for fitting (i've pasted below):

function (x, y, ...) #takes in data frame (x) and outcome(y) that we've passed
{
  loadNamespace("party")
  data <- as.data.frame(x)
  data$y <- y
  party::ctree(y ~ ., data = data) #used ctree function to train a conditional regression tree on the data set. This command returns model fit from cTree function.
}
#<environment: namespace:caret>
  
ctreeBag$pred  #command to produce the prediction code (pasted below):

function (object, x) #takes in object from cTree model fit(object) and new data set (x) and get new prediction.
{
  if (!is.data.frame(x)) 
    x <- as.data.frame(x)
  obsLevels <- levels(object@data@get("response")[, 1])
  if (!is.null(obsLevels)) {
    rawProbs <- party::treeresponse(object, x) #calculates each time the outcome(treeresponse) from the object and new data.
    probMatrix <- matrix(unlist(rawProbs), ncol = length(obsLevels), 
                         byrow = TRUE) #calculates teh probability matrix
    out <- data.frame(probMatrix)
    colnames(out) <- obsLevels #produces either the observed levels that it predicts OR (see below)
    rownames(out) <- NULL
  }
  else out <- unlist(party::treeresponse(object, x)) # OR outputs the predicted response from the variable.
  out
}
#<environment: namespace:caret>

ctreeBag$aggregate #command to produce aggregation code (often averaging; pasted  (below):
function (x, type = "class") 
{
  if (is.matrix(x[[1]]) | is.data.frame(x[[1]])) {
    pooled <- x[[1]] & NA
    classes <- colnames(pooled)
    for (i in 1:ncol(pooled)) {
      tmp <- lapply(x, function(y, col) y[, col], col = i) #getting the prediction from every single one of the model fits
      tmp <- do.call("rbind", tmp) #binds predictions together into one data matrix (each row equals prediction from one of the model predictions)
      pooled[, i] <- apply(tmp, 2, median) #then takes the median at every value (takes the median prediction from each of the different model fits across all the bootstrapped samples)
    }
    if (type == "class") {
      out <- factor(classes[apply(pooled, 1, which.max)], 
                    levels = classes)
    }
    else out <- as.data.frame(pooled)
  }
  else {
    x <- matrix(unlist(x), ncol = length(x))
    out <- apply(x, 1, median)
  }
  out
}
#<environment: namespace:caret>
