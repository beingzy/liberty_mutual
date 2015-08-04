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

# create dummy variable to encode the 
# 1) create dummy variable transformer
factor_cols  <- colnames(trainFea)[sapply(trainFea, class) == "factor"]
dummies_proc <- dummyVars(Hazard~., data=trainFea[, c("Hazard", factor_cols)])

# 2) transform trian data
trainFea_dummies <- predict(dummies_proc, trainFea)
trainFea_numeric <- trainFea[, -which(colnames(trainFea) %in% factor_cols)]
trainFea         <- data.frame(cbind(trainFea_numeric, trainFea_dummies))

# 3) transform test data
testFea_dummies <- predict(dummies_proc, data.frame(Hazard=rep(1, times=nrow(testFea)), testFea[, c(factor_cols)]))
testFea_numeric <- testFea[, -which(colnames(testFea) %in% factor_cols)]
testFea         <- data.frame(cbind(testFea_numeric, testFea_dummies))

## export data for subsequent analysis
runInfo <- list()
runInfo$train_data <- trainFea
runInfo$new_data   <- testFea

save(runInfo, file="runInfo.RData")
