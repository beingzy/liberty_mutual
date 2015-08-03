## ############################################ ##
## Liberty Mutual Kaggle Competition            ##
## Regression Modeling to Predict the existing  ##
## Damage number for given house property       ##
##                                              ##
## The below code is bachmark code to develop   ##
## a random forest models                       ##
##                                              ##
## ############################################ ##
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(stringr)
library(caret)
library(glmnet)
library(e1071)

source("SumModelGini.R")
## ############################################ ##
## set up environment ----------------------------
## ############################################ ##
repo        <- list()
repo$root   <- getwd()
repo$data   <- str_c(repo$root, "/data/")
repo$output <- str_c(repo$root, "/output/")

set.seed(1)

cat("Reading data\n")
train <- read.csv(str_c(repo$data, "train.csv"))
test  <- read.csv(str_c(repo$data, "test.csv"))

# We'll convert all the characters to factors so we can train a randomForest model on them
extractFeatures <- function(data) {
    col_dtypes <- sapply(train, class)
    character_cols <- colnames(col_dtypes == "character")
    for (col in character_cols) {
        data[, col] <- as.factor(data[, col])
    }
    return(data)
}
cat("Data Preprocessing before modeling...")
trainFea <- extractFeatures(train)
testFea  <- extractFeatures(test)

# cat("Training model\n")
# rf <- randomForest(trainFea[, 3:34], trainFea$Hazard, ntree=1000, imp=TRUE, sampsize=10000, do.trace=TRUE)

# cat("Making predictions\n")
# submission <- data.frame(Id=test$Id)
# submission$Hazard <- predict(rf, extractFeatures(testFea[,2:33]))
# write_csv(submission, str_c("output", "1_random_forest_benchmark.csv"))

# cat("Plotting variable importance\n")
# imp <- importance(rf, type=1)
# featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

## ################################################ ##
## Penlaized logistic regression                    ##
## ################################################ ##
# kfolds     <- createMultiFolds(trainFea$Hazard, k=2, times=2)
# fitControl <- trainControl(index=kfolds)
#glmnetGrid <- expand.grid(alpha=c(0, 0.5, 1), lambda=exp(seq(-10, 5, 0.5)))
# glmnetGrid <- expand.grid(alpha=c(0, 1), lambda=c(0.001, 0.5, 1))
# cv_glm     <- train(Hazard~., data=trainFea[, -1], method="glmnet"
#                     , verbose=TRUE, tuneGrid=glmnetGrid)
train_idx  <- createDataPartition(trainFea$Hazard, times=1, p=.8)
train_data <- trainFea[train_idx$Resample1, -1]
valid_data <- trainFea[-train_idx$Resample1, -1]

# train svm models
# svm_linear_fit    <- svm(Hazard~., train_data, kernel="linear")
# lm_fit     <- lm(Hazard~., train_data)
# prediction <- predict(lm_fit, valid_data)

