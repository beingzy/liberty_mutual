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
library(randomForest)
library(readr)
library(stringr)

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

trainFea <- extractFeatures(train)
testFea  <- extractFeatures(test)

cat("Training model\n")
rf <- randomForest(trainFea[, 3:34], trainFea$Hazard, ntree=1000, imp=TRUE, sampsize=10000, do.trace=TRUE)

cat("Making predictions\n")
submission <- data.frame(Id=test$Id)
submission$Hazard <- predict(rf, extractFeatures(testFea[,2:33]))
write_csv(submission, str_c("output", "1_random_forest_benchmark.csv"))

cat("Plotting variable importance\n")
imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
    geom_bar(stat="identity", fill="#53cfff") +
    coord_flip() +
    theme_light(base_size=20) +
    xlab("Importance") +
    ylab("") +
    ggtitle("Random Forest Feature Importance\n") +
    theme(plot.title=element_text(size=18))

ggsave("2_feature_importance.png", p, height=12, width=8, units="in")


## ################################################ ##
## LOAD DUMMY VARAIBLE encoded data for training    ##
## ################################################ ##
load("runInfo.RData")

trainFea <- runInfo$train_data
newFea   <- runInfo$new_data

cat("Training randomforest with data which is encoded with dummy variables\n")
rf2 <- randomForest(trainFea[, 3:ncol(trainFea)], trainFea$Hazard
                    , ntree=1000, imp=TRUE, sampsize=10000
                    , do.trace=TRUE)

cat("Making predictions\n")
submission <- data.frame(Id=newFea$Id)
submission$Hazard <- predict(rf2, newFea[, 2:ncol(newFea)])
write_csv(submission, str_c("output/", "20150804_random_forest_benchmark.csv"))

cat("Plotting variable importance\n")
imp <- importance(rf2, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() +
  theme_light(base_size=20) +
  xlab("Importance") +
  ylab("") +
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=10))

ggsave("20150804_feature_importance.png", p, height=12, width=8, units="in")

## export trained model
fit_models <- list()
fit_models$rf_n1k_dummies <- rf2

save(fit_models, file="fit_models.RData")