#Team 2: Mckinlytic 
#BUS 212
#Report 3: Code


##########################################################
#Part 0. Data Preparation

rm(list=ls())

#Load packages and read the dataset
library(tidyverse)
library(forecast)
library(leaps)
library(ggplot2)
library(car)

housing.df <- read.csv('housing.csv')
#housing.df <- as_tibble(housing.df)  

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
#Part 1: Classification Tree

library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
library(caret)

# split the dataset into training set and testing set
set.seed(2)

train.index <- sample(c(1:dim(housing.df)[1]), dim(housing.df)[1]*0.6)  
train.df <- housing.df[train.index, ]
valid.df <- housing.df[-train.index, ]

train.df$price_level <- ifelse(train.df$price>=mean(train.df$price), 1, 0)
valid.df$price_level <- ifelse(valid.df$price>=mean(train.df$price), 1, 0)



# Grid Search to try many combinations of cp and minsize, the two parameters.

curr_F1 <- 0  
best_cost_penalty <- 0
best_min_leaf_to_split <- 2

for( cost_penalty in seq(from=0, to=0.1, by=0.01)) {
  for( min_leaf_to_split in seq(from=1, to=100, by=1)) {
    
    # train the tree
    trained_tree <- rpart(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, method = "class", 
                          cp = cost_penalty, minsplit = min_leaf_to_split)
    
    # predict with the trained tree
    train.results <- predict( trained_tree, train.df, type = "class" )
    valid.results <- predict( trained_tree, valid.df, type = "class" )  
    
    # generate the confusion matrix to compare the prediction with the actual value 
    # to calculate the sensitivity and specificity
    results <- confusionMatrix( valid.results, as.factor(valid.df$price_level) )
    
    # calculate F1 from results
    Sensitivity <- results$byClass[1] # where did this come from?
    Specificity <- results$byClass[2] 
    F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
    
    # Is this F1 the best we have so far? If so, store the current values:
    if( F1 > curr_F1 ) {
      curr_F1 <- F1
      best_cost_penalty <- cost_penalty
      best_min_leaf_to_split <- min_leaf_to_split
    }
  }
}
cat("best F1=" , curr_F1, "; best best_cost_penalty=", best_cost_penalty, "; best_min_leaf_to_split=", best_min_leaf_to_split)


# retrain the tree to match the best parameters we found  
trained_tree <- rpart(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, method = "class", 
                      cp = .01 , minsplit = 11 )  
valid.results <- predict( trained_tree, valid.df, type = "class" )  
results <- confusionMatrix( valid.results, as.factor(valid.df$price_level) )
results
#accuracy=.8532, sensitivity=.8976, specificity=.7912, F1=.8410


# print that best tree 
prp(trained_tree, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(trained_tree$frame$var == "<leaf>", 'gray', 'white'))  





##########################################################
#Part 2: Ensemble Models


library(rpart)
library(rpart.plot)
library(adabag)
library(caret)
library(randomForest)

train.df$price_level = as.factor(train.df$price_level)
valid.df$price_level = as.factor(valid.df$price_level)



# bagging

set.seed(2)

mfinal_1 <- 0
maxdepth_1 <- 0
minsplit_1 <- 0


current_F1 <- 0
best_F1 <- 0
best_mfinal <- 0
best_maxdepth <- 0
best_minsplit <- 0


for (mfinal_1 in seq(1, 10, 1)) {
  for (maxdepth_1 in seq(1, 10, 1)) {
    for (minsplit_1 in seq(1, 10, 1)) {
      bag <- bagging(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, mfinal = mfinal_1, control = rpart.control(maxdepth=maxdepth_1, minsplit=minsplit_1))
      pred <- predict(bag, valid.df)
      cm <- confusionMatrix(as.factor(pred$class), as.factor(valid.df$price_level))
      cm
      
      accuracy <- cm$overall[1]
      Sensitivity <- cm$byClass[1] # where did this come from?
      Specificity <- cm$byClass[2] 
      current_F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
      # cat("Boosting trees accuracy=", accuracy, " and ", " F1=", F1)
      
      
      if (current_F1 > best_F1) {
        best_F1 <- current_F1
        best_mfinal <- mfinal_1
        best_maxdepth <- maxdepth_1
        best_minsplit <- minsplit_1
      }}}}
cat("best F1=" , current_F1, "; best mfinal=", best_mfinal, "; best maxdepth=", best_maxdepth, "; best minsplit=",best_minsplit)
#best F1= 0.8251828 ; best mfinal= 7 ; best maxdepth= 7 ; best minsplit= 3


bag <- bagging(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, mfinal = 3, control = rpart.control(maxdepth=7, minsplit=3))
pred <- predict(bag, valid.df)
cm <- confusionMatrix(as.factor(pred$class), as.factor(valid.df$price_level))
cm
2*0.8661*.7912/(0.8661+.7912)
#accuracy=.8303, sensitivity=.8740, specificity=.7692, F1=.8183





## random forest 

set.seed(2)

ntree_1 <- 0
mtry_1 <- 0
nodesize_1 <- 0
importance_1 <- TRUE

current_F1 <- 0
best_F1 <- 0
best_ntree <- 0
best_mtry <- 0
best_nodesize <- 0
best_importance <- TRUE


for (ntree_1 in seq(500, 510, 1)) {
  for (mtry_1 in seq(1, 10, 1)) {
    for (nodesize_1 in seq(1, 10, 1)) {
      for (importance_1 in c(TRUE,FALSE)) {
        rf <- randomForest(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, ntree = ntree_1, mtry = mtry_1, nodesize = nodesize_1, importance = importance_1)
        pred <- predict(rf, valid.df)
        cm <- confusionMatrix(as.factor(pred), as.factor(valid.df$price_level))
        cm
        
        accuracy <- cm$overall[1]
        Sensitivity <- cm$byClass[1] # where did this come from?
        Specificity <- cm$byClass[2] 
        current_F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
        # cat("Boosting trees accuracy=", accuracy, " and ", " F1=", F1)
        
        
        if (current_F1 > best_F1) {
          best_F1 <- current_F1
          best_ntree <- ntree_1
          best_mtry <- mtry_1
          best_nodesize <- nodesize_1
          best_importance <- importance_1
      }}}}}
cat("best F1=" , current_F1, "; best ntree=", best_ntree, "; best mtry=", best_mtry, "; best nodesize=",best_nodesize, "; best importance=",best_importance)
#best F1= 0.8533279 ; best ntree= 501 ; best mtry= 6 ; best nodesize= 7 ; best importance= TRUE



rf <- randomForest(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, ntree = 501, mtry = 6, nodesize = 7, importance = TRUE)
pred <- predict(rf, valid.df)
cm <- confusionMatrix(as.factor(pred), as.factor(valid.df$price_level))
cm
#accuracy=.8716, sensitivity=.9055, specificity=.8242, F1=.8629


## variable importance plot
varImpPlot(rf, type = 1)
#area turns out to be the most important variable, with the highest mean decrease accuracy






# boosting 
set.seed(2)

boos_1 <- TRUE
mfinal_1 <- 10
coeflearn_1 <- 'Breiman'


current_F1 <- 0
best_F1 <- 0
best_boos <- FALSE
best_mfinal <- 0
best_coeflearn <- 'Breiman'


for (boos_1 in c(TRUE,FALSE)) {
  for (mfinal_1 in seq(1, 10, 1)) {
    for (coeflearn_1 in c('Breiman','Zhu','Freund')) {
      boost <- boosting(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, boos = boos_1, mfinal = mfinal_1, coeflearn = coeflearn_1)
      pred <- predict(boost, valid.df)
      cm <- confusionMatrix(as.factor(pred$class), as.factor(valid.df$price_level))
      cm
      
      accuracy <- cm$overall[1]
      Sensitivity <- cm$byClass[1] # where did this come from?
      Specificity <- cm$byClass[2] 
      current_F1 <- (2 * Sensitivity * Specificity) / (Sensitivity + Specificity)
      # cat("Boosting trees accuracy=", accuracy, " and ", " F1=", F1)
      
      
      if (current_F1 > best_F1) {
        best_F1 <- current_F1
        best_boos <- boos_1
        best_mfinal <- mfinal_1
        best_coeflearn <- coeflearn_1
      }}}}
cat("best F1=" , current_F1, "; best boos=", best_boos, "; best mfinal=", best_mfinal, "; best_min_leaf_to_split=",best_coeflearn)
#best F1= 0.8279452 ; best boos= FALSE ; best mfinal= 8 ; best_min_leaf_to_split= Breiman


boost <- boosting(price_level ~ . -furnishingstatus-basement-parking-mainroad-bedrooms -price, data = train.df, boos = FALSE, mfinal = 8, coeflearn = 'Breiman')
pred <- predict(boost, valid.df)
cm <- confusionMatrix(as.factor(pred$class), as.factor(valid.df$price_level))
cm
#accuracy=.8716, sensitivity=.8976, specificity=.8352, F1=.8653



