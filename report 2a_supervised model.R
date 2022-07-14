#Team 2: Mckinlytic 
#BUS 212
#Report 2a: Code


#Part 0. Preparing Data for Further Analysis  ---------------------------------
housing.df <- read.csv("Housing.csv")


#price	               #price of the house
#area	               #area of the house
#bedrooms	           #no. of bedrooms
#bathrooms	           #no. of bathrooms
#stories	             #no. of stories/floors in the house
#mainroad	           #whether it is connected to the main road
#guestroom	           #Whether has a guest room
#basement	           #Whether has a basement
#hotwaterheating	     #Whether has a water heater
#airconditioning	     #Whether has AC
#parking	             #Number of House Parkings
#prefarea	           #if the house is in preferred area 
#furnishingstatus     #Furnishing status of the House


library(car)
library(caret)
# library(FNN)
library(class)


set.seed(123)
train.index <- sample(c(1:dim(housing.df)[1]), dim(housing.df)[1]*0.6)  
train.df <- housing.df[train.index, ]
valid.df <- housing.df[-train.index, ]


#Convert all dummy variables to 1 and 0
train.df$mainroad <- ifelse(train.df$mainroad=="yes", 1, 0)
train.df$guestroom <- ifelse(train.df$guestroom=="yes", 1, 0)
train.df$basement <- ifelse(train.df$basement=="yes", 1, 0)
train.df$hotwaterheating <- ifelse(train.df$hotwaterheating=="yes", 1, 0)
train.df$airconditioning <- ifelse(train.df$airconditioning=="yes", 1, 0)
train.df$prefarea <- ifelse(train.df$prefarea=="yes", 1, 0)

valid.df$mainroad <- ifelse(valid.df$mainroad=="yes", 1, 0)
valid.df$guestroom <- ifelse(valid.df$guestroom=="yes", 1, 0)
valid.df$basement <- ifelse(valid.df$basement=="yes", 1, 0)
valid.df$hotwaterheating <- ifelse(valid.df$hotwaterheating=="yes", 1, 0)
valid.df$airconditioning <- ifelse(valid.df$airconditioning=="yes", 1, 0)
valid.df$prefarea <- ifelse(valid.df$prefarea=="yes", 1, 0)


train.df$price_level <- ifelse(train.df$price>=mean(train.df$price), 1, 0)
valid.df$price_level <- ifelse(valid.df$price>=mean(train.df$price), 1, 0)
housing.df$price_level <- ifelse(housing.df$price>=mean(train.df$price), 1, 0)





#Part 1. Descriptive, Exploratory Analysis  ------------------------------------
#Box-and-whisker plots for each variable
boxplot(price ~ area, data=housing.df)  
boxplot(price ~ bedrooms, data=housing.df)
boxplot(price ~ bathrooms, data=housing.df)
boxplot(price ~ stories, data=housing.df)
boxplot(price ~ mainroad, data=housing.df)
boxplot(price ~ guestroom, data=housing.df)
boxplot(price ~ basement, data=housing.df)
boxplot(price ~ hotwaterheating, data=housing.df)
boxplot(price ~ airconditioning, data=housing.df)
boxplot(price ~ parking, data=housing.df)
boxplot(price ~ prefarea, data=housing.df)
boxplot(price ~ furnishingstatus, data=housing.df)


#Statistics of important numeric variables
summary(housing.df$price)
summary(housing.df$area)
summary(housing.df$bedrooms)
summary(housing.df$bathrooms)
summary(housing.df$stories)
summary(housing.df$parking)


#Scatterplot among variables
ggplot(housing.df, aes(x=area, y=price)) + geom_point()
ggplot(housing.df, aes(x=bedrooms, y=price)) + geom_point()
ggplot(housing.df, aes(x=furnishingstatus, y=price)) + geom_point()
ggplot(housing.df, aes(x=mainroad, y=price)) + geom_point()





#Part 2. Classification (Categorical Target) -----------------------------------
#Run logistic regression on price_level, the categorical target variable
#initial logit regression, with no interaction or polynomial term
logit.reg1 <- glm(price_level ~ . -price, data = train.df, family = binomial(link = "logit")) 
options(scipen=999)
summary(logit.reg1)
#3 insignificant variables: guestroom, and two furnishing status 


#Create a second model, with one interaction term guestroom*area 
logit.reg2 <- glm(price_level ~ . + guestroom*area -price, data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg2)
#Since the interaction term guestroom*area is statistically significant, we should keep guestroom in our model even if it's not significant by itself


#Create a third model, with two interaction terms
logit.reg3 <- glm(price_level ~ . + guestroom*area + furnishingstatus*area
                    -price, data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg3)
#Since the furnishingstatus is still insignificant even after adding the interaction term, we should drop them in our model for better estimation


#Create a fourth model, with one interaction term and furnishingstatus dropped
logit.reg4 <- glm(price_level ~ . + guestroom*area -furnishingstatus -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg4)


#Interpretation
data.frame(odds = exp(coef(logit.reg4))) 
#Among all predictors, guestroom and hotwaterheating have the strongest predictive power
#For every house with guestroom, it's 37 times more likely to have a price level of 1, i.e., having a price above the mean price 
#Similarly, houses with hot water heating are 10 times more likely to have a price level above the average. 


#Check for outliers
plot(logit.reg4)   #we have 3 outliers here: row 215, 230, and 250
train.df <- train.df[-c(215,230,250),]  #remove outliers from the train.df
valid.df <- valid.df[-c(215,230,250),]






###knn model-----------------------------
##normalize df
train.norm.df <- train.df
valid.norm.df <- valid.df

norm.values <- preProcess(train.df[1:12], method=c("center", "scale"))
train.norm.df[,1:12] <- predict(norm.values, train.df[,1:12])
valid.norm.df[,1:12] <- predict(norm.values, valid.df[,1:12])



#Check which k is the best
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))
for(i in 1:14) {          
  knn.pred <- knn(train = train.norm.df[, 1:12], cl = train.norm.df[, 13], 
                  test = valid.norm.df[, 1:12], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, factor(valid.norm.df[, 13]))$overall[1]  #$overall gives the overall accuracy value of the cm
  #note:no need to set an ifelse like in logit regression model, since knn.pred already creates 1 and 0 classification groups for valid df based on prediction
}


accuracy.df
#k=6,9,10,14


for(i in 6:6) {          
  knn.pred <- knn(train = train.norm.df[, 1:12], cl = train.norm.df[, 13], 
                  test = valid.norm.df[, 1:12], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, factor(valid.norm.df[, 13]))$overall[1] 
  
}

confusionMatrix(knn.pred, factor(valid.norm.df[, 13]))
#accuracy=0.8761, sens=0.9291, spec=0.8022, F1=xxx


for(i in 9:9) {          
  knn.pred <- knn(train = train.norm.df[, 1:12], cl = train.norm.df[, 13], 
                  test = valid.norm.df[, 1:12], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, factor(valid.norm.df[, 13]))$overall[1] 
  
}

confusionMatrix(knn.pred, factor(valid.norm.df[, 13]))

#etc. for i=10, 14







#Part 3. Regression (Numeric Target)  ------------------------------------------
rm(list=ls())


#Load packages and read the dataset
library(tidyverse)
library(forecast)
library(leaps)
library(ggplot2)
library(car)
housing.df = read.csv('housing.csv')
housing.df <- as_tibble(housing.df)  


# set up log price and visual inspect the price and log price
housing.df$logprice = log(housing.df$price)
hist(housing.df$logprice)
hist(housing.df$price)

# partition data
selected.var <- c(1:14)
set.seed(123)

train.index <- sample(c(1:dim(housing.df)[1]), dim(housing.df)[1]*0.6)  
train.df <- housing.df[train.index,selected.var ]
valid.df <- housing.df[-train.index,selected.var ]



#### Exclude outliers
#train.df <- train.df[-c(40,150,147),]
valid.df <- housing.df[-train.index,selected.var]


#lm functions for log price with interactions/variables
#housing.lm2 <- lm(logprice ~ . -price, data = train.df)
#housing.lm <- lm(logprice ~ . -price -bedrooms - basement -factor(furnishingstatus,exclude = c('furnished')) ,
#data = train.df)

housing.lm <- lm(logprice ~ . + I(area^2)-price -bedrooms - basement -factor(furnishingstatus,exclude = c('furnished')) ,
                 data = train.df)

#housing.lm <- lm(logprice ~ .  - stories -guestroom -basement -hotwaterheating -airconditioning -parking
#-prefarea, data = train.df)

#  summarize lm train
options(scipen = 999)
summary(housing.lm)

#check for Variance Inflation Factor (VIF); must be < 10; should be less than 5
vif(housing.lm)

## additional diagnostics to check for outliers/leverage points
par(mfrow=c(2,2))
plot(housing.lm)
accuracy(housing.lm)

#plot residual on histogram
housing.res = resid(housing.lm)
hist(housing.res)


#check if residual is normal
library(e1071)      
kurtosis(housing.res)
skewness(housing.res)


# as we dropped the factor level semi furnished, we will just set it to the same as the control level(semi-furnished)

#valid.df$furnishingstatus = factor(valid.df$furnishingstatus, exclude = "semi-furnished")
valid.df$furnishingstatus[valid.df$furnishingstatus == 'furnished'] = 'semi-furnished'

# use predict() to make predictions on a new set. 
housing.lm.pred <- predict(housing.lm, valid.df)
options(scipen=999, digits = 0)
some.residuals <- valid.df$logprice[1:20] - housing.lm.pred[1:20]
data.frame("Predicted" =housing.lm.pred[1:20], "Actual" = valid.df$logprice[1:20],
           "Residual" = some.residuals)

options(scipen=99, digits = 10)

# use accuracy() to compute common accuracy measures.
accuracy(housing.lm.pred, valid.df$logprice)



all.residuals <- valid.df$logprice - housing.lm.pred
hist(all.residuals, breaks = 25, xlab = "Residuals", main = "Residuals of prediction",xlim=c(-0.5,0.5))
kurtosis(all.residuals)
skewness(all.residuals)


# use regsubsets() in package leaps to run an exhaustive search. 
# unlike with lm, categorical predictors must be turned into dummies manually.

search.train <- regsubsets(logprice ~ .-price , data = train.df, nbest = 1, nvmax = dim(train.df)[2],
                           method = "exhaustive")
sum <- summary(search.train)

search.valid <- regsubsets(logprice ~ . -price, data = valid.df, nbest = 1, nvmax = dim(valid.df)[2],
                           method = "exhaustive")
sum <- summary(search.valid)

# show models
sum$which

# show metrics
sum$rsq
sum$adjr2



# use step() to run stepwise regression.
housing.lm.step <- step(housing.lm2, direction = "backward") #run lm2 in previous section to make this work
summary(housing.lm.step)  # Which variables were dropped?
housing.lm.step.pred <- predict(housing.lm.step, valid.df)
accuracy(housing.lm.step.pred, valid.df$logprice)

# create model with no predictors
housing.lm.null <- lm(logprice~1, data = valid.df)
# use step() to run forward regression.
housing.lm.step <- step(housing.lm.null, scope=list(lower=housing.lm.null, upper=housing.lm), direction = "forward")
summary(housing.lm.step)  # Which variables were added?
housing.lm.step.pred <- predict(housing.lm.step, valid.df)
accuracy(housing.lm.step.pred, valid.df$logprice)
vif(housing.lm.step)

# use step() to run stepwise regression.
housing.lm.step <- step(housing.lm, direction = "both")
summary(housing.lm.step)  # Which variables were dropped/added?
housing.lm.step.pred <- predict(housing.lm.step, valid.df)
accuracy(housing.lm.step.pred, valid.df$logprice)
vif(housing.lm.step)



