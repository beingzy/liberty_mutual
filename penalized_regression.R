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
repo$image  <- str_c(repo$root, "/images/")

## ############################################ ##
## load with transformed data --------------------
## ############################################ ##
load("runInfo.RData")

train_data <- runInfo$train_data
new_data   <- runInfo$new_data

cat("Cross-validation for Parameter tuning")
cv_lasso <- cv.glmnet(x = as.matrix(train_data[, 3:ncol(train_data)])
                       , y=train_data$Hazard, alpha=1)
# export cv curve
png(filename=str_c(repo$image, "cv_lasso.png"))
plot(cv_lasso, main = str("lambda=", str(cv_lasso$lambda.min)))
dev.off()

cat("Training lasso\n")
lasso_fit <- glmnet(x=as.matrix(train_data[, 3:ncol(train_data)])
                    , y=train_data$Hazard
                    , alpha=1
                    , lambda=cv_lasso$lambda.min)

cat("Making predictions\n")
submission <- data.frame(Id=new_data$Id)
submission$Hazard <- predict(lasso_fit, as.matrix(new_data[, 2:ncol(new_data)]))
write_csv(submission, str_c("output/", "20150804_lasso.csv"))

# laod fit_model
source("fit_models.RData")
fit_models$lasso <- lasso_fit

