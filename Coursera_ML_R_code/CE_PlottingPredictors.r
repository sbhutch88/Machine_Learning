library(ISLR); 
library(ggplot2); 
library(caret);
library(Hmisc);
library(gridExtra)
data(Wage) #loading a data set from ISLR
summary(Wage) #code to look at some simple stats for the Wage variables.

#building training and testing sets.
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)

#feature plot of training set from caret package
featurePlot(x=training[,c("age","education","jobclass")], y=training$wage, plot="pairs")
#outcome is wage, using predicting variables: age education and jobclass.
#This is one way to look at data of all chosen comparisons

#just a plot of the data of a potential interesting trend from above.
qplot(age,wage,data=training)
#Notice there are some points on top that don't fit with the rest of the data.

#Exploring the above issue:
#Plot to split another preidictor by color
qplot(age,wage,colour=jobclass,data=training)
#Notice most of the outlier type chunk comes from informational based jobs rather that industrial based jobs.

#New plot splitting by education and fitting a linear model to each education class.
qq <- qplot(age,wage,colour=education,data=training)
qq + geom_smooth(method='lm', formula=y~x)

#Sometimes we will want to split by different categories:
cutWage <- cut2(training$wage,g=3) #function from Hmisc package, breaks into 3 groups based on quantiles
table(cutWage)
#Then we can plot them:
p1 <- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot"))
#And we can add the points on top of the plot:
p2 <- qplot(cutWage,age, data=training, fill=cutWage, geom=c("boxplot","jitter")) #creates boxplots with overlayed data points.
#This lets us know how well the boxes represent the data. They could be misleading
grid.arrange(p1,p2,ncol=2) #puts 2 plots side by side (from gridExtra package)

#can also cut groups to create tables:
#using the cutWage from above to compare by jobclass.
t1 <- table(cutWage,training$jobclass)
#can also look at proportions
prop.table(t1,1) #set up for proportions by row, but a 2 would switch to column.

#Density plots (used for continuous predictors)
qplot(wage, colour=education, data=training, geom="density")
#looking at wage split by education