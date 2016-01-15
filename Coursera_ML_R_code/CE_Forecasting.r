#beginning by looking at some google data.
#*** Having some issues with code: see below. Overall this is only a few examples and extremely course
#I don't think I could do much from just this code, must see more resources from notes and slides.

library(quantmod)
library(zoo)

#Taking a look at some google stock data.
from.dat <- as.Date("01/01/08", format="%m/%d/%y")
to.dat <- as.Date("12/31/13", format="%m/%d/%y")
getSymbols("GOOG", src="google", from = from.dat, to = to.dat)
head(GOOG)

#Plotting
mGoog <- to.monthly(GOOG) #changes it to a monthly time series
googOpen <- Op(mGoog) #just takes opening information
ts1 <- ts(googOpen, frequency=12) #creates a time series
plot(ts1,xlab = "Years+1", ylab = "GOOG")
#I'm really not sure why the to.monthly function won't work however this just plots the data into a line graph.

plot(decompose(ts1), xlab="Years+1")
#Because of issues with to.montly I cant get here either, however this breaks it down by:
#Observed data, trend(line), seasonal, and random time series

#training and test sets
ts1Train <- window(ts1,start=1,end=5)
ts1Test <- window(ts1,start=5, end=(7-0.01))
#notice data sets must be consecutive data

#One option is Simple Moving Average
#This averages up all of the values for a particular time point... see notebook
plot(ts1Train)
lines(ma(ts1Train,order=3),col="red")

#Exponential Smoothing (example)
ets1 <- ets(ts1Train, model="MMM")
fcast <- forecast(ets1)
plot(fcast); lines(ts1Test,col="red")
#Creates prediction and possible bounds for prediction

#Getting accuracy
accuracy(fcast,ts1Test)
#will give RMSE and other metrics
