# install.packages("glmnet")

# Importing libraries
library(caret)
library(glmnet)
library(rpart)
library(randomForest)
library(e1071)
library(ggplot2)

# Defining directories
ADCTLtrain_dir <- "./data/train/ADCTLtrain.csv"
ADMCItrain_dir <- "./data/train/ADMCItrain.csv"
MCICTLtrain_dir <- "./data/train/MCICTLtrain.csv"

ADCTLtest_dir <- "./data/test/ADCTLtrain.csv"
ADMCItest_dir <- "./data/test/ADMCItrain.csv"
MCICTLtest_dir <- "./data/test/MCICTLtrain.csv"

# Reading data
train_data <- read.csv(ADCTLtrain_dir, header = TRUE)
test_data <- read.csv(ADCTLtrain_dir, header = TRUE)

# Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data$Label, p = 0.7, list = FALSE)
train <- train_data[trainIndex,]
valid_test <- train_data[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

# Define the features and labels
train_features <- train[,2:430]
train_labels <- train[,431]
validation_features <- validation[,2:430]
validation_labels <- validation[,431]
test_features <- test[,2:430]
test_labels <- test[,431]

# Convert labels to factor with levels "AD" and "CTL"
train_labels_factor <- factor(train_labels, levels = c("AD", "CTL"))
validation_labels_factor <- factor(validation_labels, levels = c("AD", "CTL"))
test_labels_factor <- factor(test_labels, levels = c("AD", "CTL"))

# Train an SVM classifier using the training set
svm_model <- svm(train_features, train_labels_factor)

# Tune the SVM hyperparameters using the validation set
tuned_svm_model <- tune.svm(train_features, train_labels_factor,
                            gamma = 10^(-6:-1), cost = 10^(0:3),
                            kernel = "radial", tunecontrol = tune.control(sampling = "cross"))

# Use the tuned model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features)

# Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c("AD", "CTL"))

# Evaluate the model's performance on the test data
confusionMatrix(test_pred_factor, test_labels_factor)

