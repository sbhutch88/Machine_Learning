#To run this script a CSV file named train.csv from the SF crime kaggle competition must be uploaded and labeled
#crime_training. Also the test data must be loaded.

tot.time <- proc.time()
#################____Libraries and loading data_____

library(caret)
library(ggplot2)
library(plyr)
library(lubridate) # for date handling
#Combine from randomForests was masked.

#crime_training <- read.csv("train.csv", header = TRUE)

#set.seed(1410) #This should keep the same subset each time.
#using only a random subset number of cases for speed.

set.seed(1094)
crime_training <- crime_training[sample(nrow(crime_training),500000),]


################# ______ Preprocessing
crime_training <- crime_training[-c(3, 7)] #removing description and address for now since they are unstructured and hard to use.

##Probably a good idea to also include the severity of the case variable I created in python code.

## Creating some new date and time variables that will be easier to use. 
# convert `Dates` variable from factor to date type
crime_training$Dates = ymd_hms(crime_training$Dates)
# create new variable `Year`
crime_training$Year = as.factor(year(crime_training$Dates))
# create new variable `Month`
crime_training$Month = as.factor(month(crime_training$Dates))
# create new variable `Day`
crime_training$Day = as.factor(day(crime_training$Dates))
# create new variable `Month`
crime_training$Hour = as.factor(hour(crime_training$Dates))
# get rid the variable: `Dates`
crime_training = crime_training[-1]

#Recoding the resolution into a binary variable of severity. This could be thought of as liklihood of punishment.
#**** The test set from kaggle doesn't have Resulution so I can't use punish in my model.
#crime_training$Punish <- NA
#crime_training$Punish[crime_training$Resolution == "NONE"] <- "No"
#crime_training$Punish[crime_training$Resolution == "ARREST, BOOKED"] <- "Yes"
#crime_training$Punish[crime_training$Resolution == "ARREST, CITED"] <- "Yes"
#crime_training$Punish[crime_training$Resolution == "LOCATED"] <- "No"
#crime_training$Punish[crime_training$Resolution == "PSYCHOPATHIC CASE"] <- "Yes"
#crime_training$Punish[crime_training$Resolution == "UNFOUNDED"] <- "No"
#crime_training$Punish[crime_training$Resolution == "JUVENILE BOOKED"] <- "Yes"
#crime_training$Punish[crime_training$Resolution == "COMPLAINANT REFUSES TO PROSECUTE"] <- "No"
#crime_training$Punish[crime_training$Resolution == "DISTRICT ATTORNEY REFUSES TO PROSECUTE"] <- "No"
#crime_training$Punish[crime_training$Resolution == "NOT PROSECUTED"] <- "No"
#crime_training$Punish[crime_training$Resolution == "JUVENILE CITED"] <- "Yes"
#crime_training$Punish[crime_training$Resolution == "PROSECUTED BY OUTSIDE AGENCY"] <- "Yes"
#crime_training$Punish[crime_training$Resolution == "EXCEPTIONAL CLEARANCE"] <- "No"
#crime_training$Punish[crime_training$Resolution == "JUVENILE ADMONISHED"] <- "Yes"
#crime_training$Punish[crime_training$Resolution == "JUVENILE DIVERTED"] <- "No"
#crime_training$Punish[crime_training$Resolution == "CLEARED-CONTACT JUVENILE FOR MORE INFO"] <- "No"
#crime_training$Punish[crime_training$Resolution == "PROSECUTED FOR LESSER OFFENSE"] <- "Yes"

#Dropping resolution now that I made Punish.
crime_training = crime_training[-4]
crime_training$Hour <- as.integer(crime_training$Hour)

#Time of Day creation
whenHour <- function(m) {
  if(m >5 & m < 12){
    TOD <- 'Morning'
  } else if(m >= 12 & m < 18){
      TOD <- 'Afternoon'
  } else if(m >= 18 & m < 22){
      TOD <- 'Evening'
  } else{
      TOD <- 'Night'
  }
  return(TOD)
}    
  
crime_training$TOD <- NA
#Probably could do without a for loop but this way works.
#for (i in 1:length(crime_training$Hour)){
#  crime_training$TOD[i] <- whenHour(as.numeric(crime_training$Hour[i])-1) #Need -1 because factor starts at 0 and numeric as 1.
#}

crime_training[10] <- apply(crime_training[9],MARGIN = 1,whenHour)

###Creating Weekday vs. weekend instead of day of the week.
#I'll start it this way, but looking at the counts, sunday is the lowest and Friday is the highest.Maybe Just Friday, saturday will be best but I'll see.

#Code to create the best representation of a weekend, this will be tested with GLM below.
#crime_training$weekday = "Weekday"
#weekend <- data.frame(crime_training$weekday)
#weekend$fridaySaturday <- crime_training$weekday
#weekend$fridaySaturdaySunday <- crime_training$weekday
#weekend$thursdayFridaySaturday <- crime_training$weekday
#weekend <- weekend[-c(weekend$crime_training.weekday)]

#weekend$fridaySaturdaySunday[crime_training$DayOfWeek== "Saturday" | 
#                   crime_training$DayOfWeek== "Sunday" | 
#                   crime_training$DayOfWeek== "Friday" ] = "Weekend"
#weekend$fridaySaturday[crime_training$DayOfWeek== "Saturday" | 
#                 crime_training$DayOfWeek== "Friday" ] = "Weekend"

#weekend$thursdayFridaySaturday[crime_training$DayOfWeek== "Saturday" | 
#                         crime_training$DayOfWeek== "Thursday" | 
#                         crime_training$DayOfWeek== "Friday" ] = "Weekend"
#weekend <- cbind(crime_training$Category,weekend)
#names(weekend)[1] <- "Category"

###Weekend winner
#crime_training$weekday = "Weekday"
#crime_training$weekday[crime_training$DayOfWeek== "Saturday" | 
#                   crime_training$DayOfWeek== "Sunday" | 
#                   crime_training$DayOfWeek== "Friday" ] = "Weekend"

##** The model actually predicts better without the weekday/weekend variable. I will need a better assessment method, but 
#for now I've been running it with a subset of the data.

#crime_training <- crime_training[-2] #Getting rid of day of week now that I have weekend/weekday
##Other ideas:
#Create more categories by time of day, and month.
#What if I just remove day of week altogether? ** Didn't help

################ ______ creating training and testing sets

#Removing unused levels of Category. I may have to change my methods later when testing model.
crime_training$Category<- factor(crime_training$Category)

set.seed(4901)
intrain <- createDataPartition(y=crime_training$Category, p = .7, list = FALSE)
training <- crime_training[intrain,]
testing <- crime_training[-intrain,]


################ ______ Quick and dirty exploration_______
#qplot(X,Y,colour = PdDistrict, data = training,xlab = "Longitude", ylab = "Latitude")
#qplot(X,Y,colour = Punish, data = training,xlab = "Longitude", ylab = "Latitude")
#qplot(X,Y,colour = TOD, data = training,xlab = "Longitude", ylab = "Latitude")

#If there are outliers.
#outliers <- training[training$X == -120.5,]#2 clear outliers
#training <- training[-c(679644,674044),]

#dayOfWeekCounts <- summary(crime_training$DayOfWeek)
#barchart(dayOfWeekCounts)

################## ____________ Analyzing Predictors (AIC)
#full.model = glm(Category ~ ., data = training, family = binomial)
#null.model = glm(Category ~ 1, data = training, family = binomial) #1 is category

#Stepwise AIC assessment step(initial modeld, range of models)
#variable.selection = step(null.model, formula(full.model), direction = "forward")
#variable.selection[1]
#I believe 'none' is the intercept. Notice some of the top predictors are my created variables and lat and long.
#Also notice how much better TOD is than hour, and that Month is also very good.


#analyzing the best weekend classifier
#weekend_full.model = glm(Category ~ ., data = weekend, family = binomial)
#weekend_null.model = glm(Category ~ 1, data = weekend, family = binomial) #1 is category
#variable.selection = step(weekend_null.model, formula(weekend_full.model), direction = "forward")
#Results reveal the Friday-Saturday-Sunday as a representation for weekend is the best predictor with 100,000 data points.

################## ____________ Logistic Regression Model

#library(nnet) #I like caret but I'll try this out.

# run logistic regression model
#log.time <- proc.time() #determines the length of time in seconds the process is taking
#log.model <- multinom(formula = Category ~ DayOfWeek + X + Y + Month + TOD + Punish , data = training)
#log.time <- proc.time() - log.time

#Testing the model
#log.result <- predict(log.model, testing[, -1]) # prediction on test data
#log.accuracy <- sum(log.result == t(testing[, 1])) # checking for out-of-sample performance
#percentCorrect <- round(log.accuracy/length(testing$Category), digits = 3)
#cat("The model took ", log.time[3], " seconds to generate\n",
#    "Out of ", dim(testing)[1], " test cases, it got", log.accuracy ," right, or",percentCorrect, "percent correct")


############# __________ Random Forests Model

#Changing the category names to be valid names of a new dataframe when creating probability outputs.
training$Category <- make.names(training$Category, unique = FALSE, allow_ = TRUE)

#getting rid of continuous latitude and longitude
rf.train = training[-c(4,5)]
rf.test = training[-c(4,5)]

rf.grid <- expand.grid(.mtry = c(3, 6, 9)) #creating cross validation sets
rf.control <- trainControl(method="cv", number=10, repeats=3,classProbs = TRUE)


#rf.cv.time <- proc.time()
rf.cv <- train(Category ~ ., data = rf.train,
               method = "rf",
               maxit = 1000,
               tuneGrid = rf.grid,
               trace = TRUE,
               ntree = 100,
               trControl = rf.control
)
#rf.cv.time <- proc.time() - rf.cv.time

#Testing the Model
rf.cv.result <- predict(rf.cv, rf.test[, -1],"prob")
rf.cv.accuracy <- sum(rf.cv.result == t(rf.test[, 1]))
rf.cv.probs <- predict(rf.cv, rf.test[, -1],"prob") #Used for kaggle submission
#percentCorrect <- round(rf.cv.accuracy/length(rf.test$Category), digits = 3)
#cat("The model took ", rf.cv.time[3], " seconds to generate.\n",
#    "Out of ", dim(rf.test)[1], " test cases, it got", rf.cv.accuracy ," right, or", 
#    percentCorrect, " percent correct")



################ ______________ Kaggle submission

#example kaggle submission so I can see what it looks like.
example <- read.csv("sampleSubmission.csv", header = TRUE)
kaggleTest <- read.csv("test.csv", header=T)

#Used for testing.
#kaggleTest <- kaggleTest[sample(nrow(kaggleTest),5000),]


###Preprocessing Test set.
kaggleTest <- kaggleTest[-c(5)] #Removing Address

# convert `Dates` variable from factor to date type
kaggleTest$Dates = ymd_hms(kaggleTest$Dates)
# create new variable `Year`
kaggleTest$Year = as.factor(year(kaggleTest$Dates))
# create new variable `Month`
kaggleTest$Month = as.factor(month(kaggleTest$Dates))
# create new variable `Day`
kaggleTest$Day = as.factor(day(kaggleTest$Dates))
# create new variable `Month`
kaggleTest$Hour = as.factor(hour(kaggleTest$Dates))
# get rid the variable: `Dates`
kaggleTest = kaggleTest[-2]


print("Creating TOD variable for Test Data")
kaggleTest$TOD <- NA
kaggleTest$Hour <- as.integer(kaggleTest$Hour)
#Probably could do without a for loop but this way works.
#Takes FOREVER, gotta be a faster way using apply.
#for (i in 1:length(kaggleTest$Hour)){
#kaggleTest$TOD[i] <- whenHour(as.numeric(kaggleTest$Hour[i])-1) #Need -1 because factor starts at 0 and numeric as 1.
#}

kaggleTest[10] <- apply(kaggleTest[9],MARGIN = 1,whenHour)


 #Used for kaggle submission
#rf.kaggleTest.result <- predict(rf.cv, kaggleTest)
rf.kaggleTest.probs <- predict(rf.cv, kaggleTest,"prob") #Used for kaggle submission

#rf.kaggleTest.probs <- extractProb(rf.kaggleTest.result)



#This whole chunk of code will add an ID and TREA column to make the data frame match the example submission.
kaggleSubmission <- rf.kaggleTest.probs
a <- data.frame(example$Id) #used for my test example only.
names(a)[1] <- "Id"
kaggleSubmission <-cbind(a,kaggleSubmission)

#If a TREA column doesn't exist because of a limited data set.
#b <- data.frame(example$TREA)
#names(b)[1] <- "TREA"
#kaggleSubmission <- cbind(kaggleSubmission[,1:34],b,kaggleSubmission[,35:ncol(kaggleSubmission)])

#Changing example data frame to data frame of all 0s.
#kaggleSubmission <- replace(example,c(39),0)


#Filling in the data frame with my results
#for(i in 1:length(rf.kaggleTest.result)){
#kaggleSubmission[i,createKaggleCol(rf.kaggleTest.result[i])] = 1
#}


##########_______ Saving Kaggle submission csv file

#Not sure why the sameple enters with a . instead of space for 2 word headers 
#but they need to be changed back for the submission.


names(kaggleSubmission)[4] <- as.character("BAD CHECKS")
names(kaggleSubmission)[7] <- as.character("DISORDERLY CONDUCT")
names(kaggleSubmission)[8] <- as.character("DRIVING UNDER THE INFLUENCE")
names(kaggleSubmission)[13] <- as.character("FAMILY OFFENSES")
names(kaggleSubmission)[19] <- as.character("LIQUOR LAWS")
names(kaggleSubmission)[21] <- as.character("MISSING PERSON")
names(kaggleSubmission)[23] <- as.character("OTHER OFFENSES")
names(kaggleSubmission)[26] <- as.character("RECOVERED VEHICLE")
names(kaggleSubmission)[29] <- as.character("SECONDARY CODES")
names(kaggleSubmission)[30] <- as.character("SEX OFFENSES FORCIBLE")
names(kaggleSubmission)[31] <- as.character("SEX OFFENSES NON FORCIBLE")
names(kaggleSubmission)[32] <- as.character("STOLEN PROPERTY")
names(kaggleSubmission)[34] <- as.character("SUSPICIOUS OCC")
names(kaggleSubmission)[38] <- as.character("VEHICLE THEFT")
names(kaggleSubmission)[40] <- as.character("WEAPON LAWS")

names(kaggleSubmission)[9] <- as.character("DRUG/NARCOTIC")
names(kaggleSubmission)[14] <- as.character("FORGERY/COUNTERFEITING")
names(kaggleSubmission)[18] <- as.character("LARCENY/THEFT")
names(kaggleSubmission)[24] <- as.character("PORNOGRAPHY/OBSCENE MAT")

names(kaggleSubmission)[22] <- as.character("NON-CRIMINAL")

names(kaggleSubmission) <- as.character(names(kaggleSubmission))


write.csv(kaggleSubmission,"kaggleSubmission4.csv",row.names = FALSE)

#First submission wasn't great.
#I trid simply doubling the ntrees and it was only barely better.
#I think my training set is far too small and I need to really push my computing power.
tot.time <- proc.time() - tot.time
print(tot.time)