# install.packages("usdm")

library(caret)
library(glmnet)
library(rpart)
library(randomForest)
library(e1071)
library(ggplot2)
library(usdm)

# Defining directories
ADCTLtrain_dir <- "./data/train/ADCTLtrain.csv"
ADMCItrain_dir <- "./data/train/ADMCItrain.csv"
MCICTLtrain_dir <- "./data/train/MCICTLtrain.csv"

ADCTLtest_dir <- "./data/test/ADCTLtest.csv"
ADMCItest_dir <- "./data/test/ADMCItest.csv"
MCICTLtest_dir <- "./data/test/MCICTLtest.csv"

# Reading data
train_data <- read.csv(ADCTLtrain_dir, header = TRUE)
test_data <- read.csv(ADCTLtest_dir, header = TRUE)

# dropping the id col
train_data <- train_data[, -1] 
test_data <- test_data[, -1]

# encoding label col in training data
train_data$Label <- ifelse(train_data$Label == "AD", 1, 0)

# Checking if the dataset has the non-numeric columns
non_numeric_cols <- names(train_data)[!sapply(train_data, is.numeric)]
cat(non_numeric_cols)

# Compute the correlation matrix between the variables in the dataset
cor_matrix <- cor(train_data)

# Identify pairs of features with a correlation coefficient above 0.7
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.7)

# Remove the problematic features from the dataset
train_data_clean <- train_data[,-highly_correlated]

# Convert the labels to a factor
train_data_clean$Label <- as.factor(train_data_clean$Label)

# Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data_clean$Label, p = 0.7, list = FALSE)
train <- train_data_clean[trainIndex,]
valid_test <- train_data_clean[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

##### Lasso
# # Set up the Lasso regression model
# train$Label <- as.numeric(train$Label)
# lasso_model <- cv.glmnet(as.matrix(train[,1:ncol(train)-1]), train$Label, type.measure = "class", nfolds = 10)
# 
# # Extract the selected features for each dataset
# train_selected <- predict(lasso_model, newx = as.matrix(train[,1:ncol(train)-1]), s = "lambda.min")
# validation_selected <- predict(lasso_model, newx = as.matrix(validation[,1:ncol(validation)-1]), s = "lambda.min")
# test_selected <- predict(lasso_model, newx = as.matrix(test[,1:ncol(test)-1]), s = "lambda.min")

##### RFE
# # Define the control parameters for the RFE algorithm
# control <- rfeControl(functions = rfFuncs,
#                       method = "cv",
#                       number = 10)
# 
# # Define the model to be used by RFE
# model <- train(Label ~ ., data = train, method = "rf")
# 
# # Use RFE to perform feature selection
# results <- rfe(train[, -1], train$Label, sizes = c(1:10), rfeControl = control, method = "rf")
# 
# # Print the results
# print(results)
# 
# # Plot the results
# plot(results)

##### SVM

# Define the features and labels
train_features <- train[,2:ncol(train)-1]
train_labels <- train[,ncol(train)]
validation_features <- validation[,2:ncol(validation)-1]
validation_labels <- validation[,ncol(validation)]
test_features <- test[,2:ncol(test)-1]
test_labels <- test[,ncol(test)]

# Convert labels to factor with levels "AD" and "CTL"
train_labels_factor <- factor(train_labels, levels = c(1, 0))
validation_labels_factor <- factor(validation_labels, levels = c(1, 0))
test_labels_factor <- factor(test_labels, levels = c(1, 0))

# Train an SVM classifier using the training set
svm_model <- svm(train_features, train_labels_factor)

# Tune the SVM hyperparameters using the validation set
tuned_svm_model <- tune.svm(train_features, train_labels_factor,
                            gamma = 10^(-6:-1), cost = 10^(0:3),
                            kernel = "radial", tunecontrol = tune.control(sampling = "cross"),
                            validation.x = validation_features, validation.y = validation_labels_factor)

# Use the tuned model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features)

# Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0))

# Evaluate the model's performance on the test data
confusionMatrix(test_pred_factor, test_labels_factor)

# Predicting on test data
# Preprocess the test data, MUST be same as training preprocessing!!
test_features_prediction <- test_data[,-highly_correlated]

# Use the trained SVM model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features_prediction)

# Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0))

# Print the predicted labels for the test data
print(test_pred_factor)


