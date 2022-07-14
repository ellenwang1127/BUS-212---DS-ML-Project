#Team 2: Mckinlytic 
#BUS 212
#Report 2b: Code


##########################################################
#Part 0. Data Preparation

rm(list=ls())

#Load packages and read the dataset
library(tidyverse)
library(forecast)
library(leaps)
library(ggplot2)
library(car)

housing.df = read.csv('housing.csv')
housing.df <- as_tibble(housing.df)  

housing.df$mainroad = as.factor(housing.df$mainroad)
housing.df$guestroom = as.factor(housing.df$guestroom)
housing.df$basement = as.factor(housing.df$basement)
housing.df$hotwaterheating = as.factor(housing.df$hotwaterheating)
housing.df$airconditioning = as.factor(housing.df$airconditioning)
housing.df$prefarea = as.factor(housing.df$prefarea)
housing.df$furnishingstatus = as.factor(housing.df$furnishingstatus)

housing.df$mainroad = as.numeric(housing.df$mainroad)
housing.df$guestroom = as.numeric(housing.df$guestroom)
housing.df$basement = as.numeric(housing.df$basement)
housing.df$hotwaterheating = as.numeric(housing.df$hotwaterheating)
housing.df$airconditioning = as.numeric(housing.df$airconditioning)
housing.df$prefarea = as.numeric(housing.df$prefarea)
housing.df$furnishingstatus = as.numeric(housing.df$furnishingstatus)




##########################################################
#Part 1. Descriptive, Exploratory Analysis 


#Box-and-whisker plots for each variable
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

sd(housing.df$price)
sd(housing.df$area)


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

pairs(~ price+area+bedrooms+bathrooms+stories+mainroad+guestroom+
        basement+hotwaterheating+airconditioning+
        parking+prefarea+furnishingstatus,
      data=housing.df, main="Scatterplot Matrix")



##########################################################
#Part 2. Discover Clusters


# Hierarchical Clustering --------------------------------------

# compute Euclidean distance
# (to compute other distance measures, change the value in method = )
set.seed(2)
d <- dist(housing.df, method = "euclidean")



# normalize input variables
housing.df.norm <- sapply(housing.df, scale)


# compute normalized DISTANCE parameter based on all 8 variables
d.norm <- dist(housing.df.norm, method = "euclidean") 
d.norm2 <- dist(housing.df.norm, method = "maximum") 
d.norm3 <- dist(housing.df.norm, method = "manhattan") 
d.norm4 <- dist(housing.df.norm, method = "canberra") 
d.norm5 <- dist(housing.df.norm, method = "binary") 
d.norm6 <- dist(housing.df.norm, method = "minkowski") 



# in hclust() set LINKAGE parameter to "ward.D", "single", "complete", "average", "median", or "centroid"
hc1 <- hclust(d.norm, method = "single")
plot(hc1, hang = -1, ann = FALSE) 
# Note: hang -1 means do not hang labels off the leaves; make them level; ann is for plot annotation

hc2 <- hclust(d.norm, method = "average")
plot(hc2, hang = -1, ann = FALSE)

hc3 <- hclust(d.norm, method = "median")
plot(hc3, hang = -1, ann = FALSE)

hc4 <- hclust(d.norm, method = "complete")
plot(hc4, hang = -1, ann = FALSE)

hc5 <- hclust(d.norm, method = "centroid")
plot(hc5, hang = -1, ann = FALSE)

hc6 <- hclust(d.norm, method = "ward.D")
plot(hc6, hang = -1, ann = FALSE)

hc62 <- hclust(d.norm2, method = "ward.D")
plot(hc62, hang = -1, ann = FALSE)

hc63 <- hclust(d.norm3, method = "ward.D")
plot(hc63, hang = -1, ann = FALSE)

hc64 <- hclust(d.norm4, method = "ward.D")
plot(hc64, hang = -1, ann = FALSE)

hc65 <- hclust(d.norm5, method = "ward.D")
plot(hc65, hang = -1, ann = FALSE)

hc66 <- hclust(d.norm6, method = "ward.D")
plot(hc66, hang = -1, ann = FALSE)

# Use map_dbl to run many models with varying value of k (centers)
tot_withinss <- map_dbl(1:30,  function(k){
  model <- kmeans(x = housing.df.norm, centers = k)
  model$tot.withinss
})

# Generate a data frame containing both k and tot_withinss
elbow_df <- data.frame(
  k = 1:30,
  tot_withinss = tot_withinss
)


# Plot the elbow plot
ggplot(elbow_df, aes(x = k, y = tot_withinss)) +
  geom_line() + geom_point()+
  scale_x_continuous(breaks = 1:30)
#elbow point at k=8, so go with 8 clusters



#### Table 15.6

memb1 <- cutree(hc1, k = 8) #single
memb1
hist(memb1)

memb2 <- cutree(hc2, k = 8) #average
memb2
hist(memb2)

memb3 <- cutree(hc3, k = 8) #median
memb3
hist(memb3)

memb4 <- cutree(hc4, k = 8) #complete
memb4
hist(memb4)

memb5 <- cutree(hc5, k = 8) #centroid
memb5
hist(memb5)

# ward.D
memb6 <- cutree(hc6, k = 8) #ward.D
memb6
hist(memb6)

memb62 <- cutree(hc62, k = 8) #ward.D
memb62
hist(memb62)

memb63 <- cutree(hc63, k = 8) #ward.D
memb63
hist(memb63)

memb64 <- cutree(hc64, k = 8) #ward.D
memb64
hist(memb64)

memb65 <- cutree(hc65, k = 8) #ward.D
memb65
hist(memb65)

memb66 <- cutree(hc66, k = 8) #ward.D
memb66
hist(memb66)

hist(memb6)
hist(memb66)

# plot heatmap 
# rev() reverses the color mapping to large = dark
heatmap(as.matrix(housing.df.norm), Colv = NA, hclustfun = hclust, 
        col=rev(paste("grey",1:99,sep="")))






# K-means Clusering -------------------------------------------------------------

# Run k-means algorithm if you know what k should be; you do not have to choose distance or linkage parameters.
set.seed(2)
km <- kmeans(housing.df.norm, 8)   # k = 8 

# With what utilities is Boston utility clustering?

# show cluster membership
km$cluster

# How can we get just the cluster numbers (no rownames, no line numbers in output)?
cat(km$cluster)

#### Table 15.10
# centroids
km$centers



#### Figure 15.5

# plot an empty scatter plot
plot(c(0), xaxt = 'n', ylab = "", type = "l", 
     ylim = c(min(km$centers), max(km$centers)), xlim = c(0, 13))

# label x-axes
axis(1, at = c(1:13), labels = colnames(housing.df.norm))

# plot centroids
for (i in c(1:8))
  lines(km$centers[i,], lty = i, lwd = 2, col = switch(i, "black", "red", 
                                                       "green", "purple","yellow","pink","grey","blue"))  

# name clusters
text(x = 0.1, y = km$centers[, 1], labels = paste("Cluster", c(1:8)))


dist(km$centers)




########################################################################
#Part 3. Revisit Models From Report 2a, 

#adding cluster variables as additional predictors
housing.df$hclust <- memb6
housing.df$kmeans <- km$cluster




# Regression model (Numeric Target)  ------------------------------------------
# set up log price and visual inspect the price and log price
housing.df$logprice <- log(housing.df$price)
hist(housing.df$logprice)
hist(housing.df$price)

# partition data
selected.var <- c(1:16)
set.seed(2)

train.index <- sample(c(1:dim(housing.df)[1]), dim(housing.df)[1]*0.6)  
train.df <- housing.df[train.index,selected.var ]
valid.df <- housing.df[-train.index,selected.var ]



#### Exclude outliers
#train.df <- train.df[-c(166,239),] # a total of 5 outliers were removed in the iterative process
valid.df <- housing.df[-train.index,selected.var]


#lm functions for log price with interactions/variables
housing.lm <- lm(logprice ~ . + I(area^2)-price -bedrooms - basement - furnishingstatus -mainroad -guestroom, 
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


#check if residual is close to normal
library(e1071)      
kurtosis(housing.res)
skewness(housing.res)


# use predict() to make predictions on a new set. 
housing.lm.pred <- predict(housing.lm, valid.df)
options(scipen=99, digits = 10)

# use accuracy() to compute common accuracy measures.
accuracy(housing.lm.pred, valid.df$logprice)

#check if prediction residual is close to normal
all.residuals <- valid.df$logprice - housing.lm.pred
hist(all.residuals, breaks = 25, xlab = "Residuals", main = "Residuals of prediction",xlim=c(-0.5,0.5))
kurtosis(all.residuals)
skewness(all.residuals)





# Classification Model: Logit Regression and KNN ------------------------------------
set.seed(2)
housing.df <- read.csv("Housing.csv")
housing.df <- as_tibble(housing.df)  


housing.df$mainroad = as.factor(housing.df$mainroad)
housing.df$guestroom = as.factor(housing.df$guestroom)
housing.df$basement = as.factor(housing.df$basement)
housing.df$hotwaterheating = as.factor(housing.df$hotwaterheating)
housing.df$airconditioning = as.factor(housing.df$airconditioning)
housing.df$prefarea = as.factor(housing.df$prefarea)
housing.df$furnishingstatus = as.factor(housing.df$furnishingstatus)

housing.df$mainroad = as.numeric(housing.df$mainroad)
housing.df$guestroom = as.numeric(housing.df$guestroom)
housing.df$basement = as.numeric(housing.df$basement)
housing.df$hotwaterheating = as.numeric(housing.df$hotwaterheating)
housing.df$airconditioning = as.numeric(housing.df$airconditioning)
housing.df$prefarea = as.numeric(housing.df$prefarea)
housing.df$furnishingstatus = as.numeric(housing.df$furnishingstatus)


#add cluster variables to the dataset
housing.df$hclust <- memb6
housing.df$kmeans <- km$cluster


# split the dataset into training set and testing set
train.index <- sample(c(1:dim(housing.df)[1]), dim(housing.df)[1]*0.6)  
train.df <- housing.df[train.index, ]
valid.df <- housing.df[-train.index, ]

train.df$price_level <- ifelse(train.df$price>=mean(train.df$price), 1, 0)
valid.df$price_level <- ifelse(valid.df$price>=mean(train.df$price), 1, 0)



# Logit regression model

#try different logit regression models, adding or dropping variables
logit.reg <- glm(price_level ~ . -price, 
                 data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg)


logit.reg1 <- glm(price_level ~ . +area*bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg1)


logit.reg2 <- glm(price_level ~ . +area*mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg2)


logit.reg3 <- glm(price_level ~ . -basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg3)


logit.reg4 <- glm(price_level ~ . +area*furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg4)


logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)



#check for outliers
#1st time removing outliers
par(mfrow=c(2,2))
plot(logit.reg5)
train.df <- train.df[-c(319,249,292),]  
logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)


#2nd time removing outliers
plot(logit.reg5)
train.df <- train.df[-c(273,143,256),]
logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)


#3rd time removing outliers
plot(logit.reg5)
train.df <- train.df[-c(281,208,53),]
logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)


#4th time removing outliers
plot(logit.reg5)
train.df <- train.df[-c(148,5,171),]
logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)


#5th time removing outliers
plot(logit.reg5)
train.df <- train.df[-c(184,229,88),]
logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)


#6th time removing outliers
plot(logit.reg5)
train.df <- train.df[-c(296,6,126),]
logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)


#7th time removing outliers
plot(logit.reg5)
train.df <- train.df[-c(124),]
logit.reg5 <- glm(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, 
                  data = train.df, family = binomial(link = "logit"))
options(scipen=999)
summary(logit.reg5)

#stop removing outliers, as the new outliers are almost on the horizontal level in the residuals vs fitted plot
#and aligning with the diagonal in the normal Q-Q plot



#plot residual on histogram
housing.res <- resid(logit.reg5)
hist(housing.res)


#check if residual is normal
library(e1071)      
kurtosis(housing.res)
skewness(housing.res)


# interpretation of logistic model
data.frame(odds = exp(coef(logit.reg5))) 


library(caret)
#check accuracy, specificity, sensitivity from Confusion Matrix (interpretation of CM)
confusionMatrix(as.factor(ifelse(logit.reg5$fitted > 0.5, 1, 0)), as.factor(as.data.frame(train.df)[,16]))
#accuracy=0.9221, sens=0.9451, spec=0.8889, F1=0.9161


# area under AUC curve
library(pROC)
auc(as.data.frame(train.df)[,16], logit.reg5$fitted)
#area=.9831




# KNN model

set.seed(2)
train.norm.df <- train.df
valid.norm.df <- valid.df

norm.values <- preProcess(train.df[1:15], method=c("center", "scale"))
train.norm.df[,1:15] <- predict(norm.values, train.df[,1:15])
valid.norm.df[,1:15] <- predict(norm.values, valid.df[,1:15])


#Check which k is the best
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))
for(i in 1:14) {          
  knn.pred <- knn(train = as.data.frame(train.norm.df)[, 1:15], cl = as.data.frame(train.norm.df)[,16], 
                  test = as.data.frame(valid.norm.df)[, 1:15], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, factor(as.data.frame(valid.norm.df)[, 16]))$overall[1]  #$overall gives the overall accuracy value of the cm
}

accuracy.df
# k=13



for(i in 13:13) {          
  knn.pred <- knn(train = as.data.frame(train.norm.df)[, 1:15], cl = as.data.frame(train.norm.df)[,16], 
                  test = as.data.frame(valid.norm.df)[, 1:15], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, factor(as.data.frame(valid.norm.df)[, 16]))$overall[1]  #$overall gives the overall accuracy value of the cm
  #note:no need to set an ifelse like in logit regression model, since knn.pred already creates 1 and 0 classification groups for valid df based on prediction
}

confusionMatrix(knn.pred, factor(as.data.frame(valid.norm.df)[, 16]))
#accuracy=0.8716, sens=.9291, spec=0.7912, F1=0.8546









