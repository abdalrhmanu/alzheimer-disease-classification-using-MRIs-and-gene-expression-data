####  0.0 Installing necessary packages
list.packages <- c("caret", "glmnet", "rpart", "randomForest", "e1071", "ggplot2", "usdm", "Boruta", "gbm", "pROC")
new.packages <- list.packages[!(list.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(caret)
library(glmnet)
library(rpart)
library(randomForest)
library(e1071)
library(ggplot2)
library(usdm)
library(Boruta)
library(gbm)
library(pROC)  # Load the pROC package for AUC calculation


####  0.1 Defining directories
ADCTL_train_dir <- "./data/train/ADCTLtrain.csv"
ADMCI_train_dir <- "./data/train/ADMCItrain.csv"
MCICTL_train_dir <- "./data/train/MCICTLtrain.csv"

ADCTL_test_dir <- "./data/test/ADCTLtest.csv"
ADMCI_test_dir <- "./data/test/ADMCItest.csv"
MCICTL_test_dir <- "./data/test/MCICTLtest.csv"


#### Defining functions
calculate_metrics <- function(predicted_labels, actual_labels, true_labels_value, false_labels_value) {
  library(pROC)  # Load the pROC package for AUC calculation
  
  # Convert factor labels to numeric
  predicted_numeric <- as.numeric(predicted_labels) - 1
  actual_numeric <- as.numeric(actual_labels) - 1
  
  # Calculate AUC
  roc_obj <- roc(actual_numeric, predicted_numeric)
  auc <- auc(roc_obj)
  
  # Calculate MCC
  TP <- sum(predicted_labels == true_labels_value & actual_labels == true_labels_value)
  TN <- sum(predicted_labels == false_labels_value & actual_labels == false_labels_value)
  FP <- sum(predicted_labels == true_labels_value & actual_labels == false_labels_value)
  FN <- sum(predicted_labels == false_labels_value & actual_labels == true_labels_value)
  
  # Check for division by zero
  if (TP + FP == 0 || TP + FN == 0 || TN + FP == 0 || TN + FN == 0) {
    mcc <- NaN
  } else {
    mcc <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  }
  
  # Calculate precision
  precision <- TP / (TP + FP)
  
  # Calculate recall
  recall <- TP / (TP + FN)
  
  # Calculate F1 score
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Calculate sensitivity
  sensitivity <- TP / (TP + FN)
  
  # Calculate specificity
  specificity <- TN / (TN + FP)
  
  # Calculate Balanced Accuracy
  ba <- (sensitivity + specificity) / 2
  
  
  # Print the calculated metrics
  print(paste("AUC:", auc))
  print(paste("MCC:", mcc))
  print(paste("Precision:", precision))
  print(paste("F1 Score:", f1_score))
  print(paste("Balanced Accuracy:", ba))
  
  
  # Return the calculated metrics
  return(list(AUC = auc, MCC = mcc, Precision = precision, Recall = recall, F1_Score = f1_score, Balanced_Accuracy = ba))
}

# 
# 
# ####  0.2 Reading data
# train_data_ADCTL <- read.csv(ADCTL_train_dir, header = TRUE)
# test_data_ADCTL <- read.csv(ADCTL_test_dir, header = TRUE)
# 
# #############################################
# #                   ADCTL
# ######### DATASET 1 EXPERIMENTAL ############
# #############################################
# 
# ####  3.0 Pre-processing the data
# #     3.1 Dropping the id col in all datasets
# ####  3.0 Pre-processing the data
# train_data_ADCTL <- train_data_ADCTL[, -1] 
# test_data_ADCTL <- test_data_ADCTL[, -1]
# 
# dim(train_data_ADCTL)
# dim(test_data_ADCTL)
# 
# #     3.2 Encoding label col in training data
# train_data_ADCTL$Label <- ifelse(train_data_ADCTL$Label == "AD", 1, 0)
# 
# #     3.3 Checking if the data set has the non-numeric columns
# non_numeric_cols <- names(train_data_ADCTL)[!sapply(train_data_ADCTL, is.numeric)]
# print(non_numeric_cols)
# 
# #     4.0 Feature selection using correlation on ADCTL
# #     4.1 Compute the correlation matrix between the variables in the dataset
# cor_matrix <- cor(train_data_ADCTL)
# 
# #     4.2 Identify pairs of features with a correlation coefficient above 0.7
# highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.7)
# 
# #     4.3 Remove the problematic features from the dataset
# train_data_ADCTL_clean <- train_data_ADCTL[,-highly_correlated]
# test_data_ADCTL_clean <- test_data_ADCTL[,-highly_correlated]
# 
# #     5.0 Convert the labels to a factor
# train_data_ADCTL_clean$Label <- as.factor(train_data_ADCTL_clean$Label)
# 
# #     6.0 Recursive Feature Elimination
# #     6.1 Perform Recursive Feature Elimination (RFE) using caret
# ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)  # Use random forest as the underlying model
# result <- rfe(train_data_ADCTL_clean[, -ncol(train_data_ADCTL_clean)], train_data_ADCTL_clean$Label, sizes = c(1:ncol(train_data_ADCTL_clean)-1), rfeControl = ctrl)
# 
# # Extract the top 5 variables from the RFE results
# selected_features <- result$optVariables[1:5]
# 
# #     6.2 Get the selected features
# # selected_features <- result$optVariables
# 
# #     6.3 Subset the training and test sets with the selected features
# train_data_ADCTL_clean <- train_data_ADCTL_clean[, c(selected_features, "Label")]
# test_data_ADCTL_clean <- test_data_ADCTL_clean[, selected_features]
# 
# dim(train_data_ADCTL_clean)
# dim(test_data_ADCTL_clean)
# 
# #     7.0 Convert the labels to a factor
# train_data_ADCTL_clean$Label <- as.factor(train_data_ADCTL_clean$Label)
# 
# ################################################################################
# ####  6.0 Data Splitting
# #     6.1 Split the data into training, validation, and test sets
# set.seed(123)
# trainIndex <- createDataPartition(train_data_ADCTL_clean$Label, 
#                                   p = 0.7, list = FALSE)
# train <- train_data_ADCTL_clean[trainIndex,]
# valid_test <- train_data_ADCTL_clean[-trainIndex,]
# 
# validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
# validation <- valid_test[validIndex,]
# test <- valid_test[-validIndex,]
# 
# # Convert the labels to factors for all splits
# train$Label <- as.factor(train$Label)
# validation$Label <- as.factor(validation$Label)
# test$Label <- as.factor(test$Label)
# 
# ################################################################################
# ####  7.0 Training dataset ADCTL Using SVM
# #     7.1 Train an SVM classifier using the training set
# svm_model <- svm(train[, -ncol(train)], train$Label,
#                  trControl = ctrl)
# 
# ####  7.2 Tune the SVM hyper-parameters using the validation set
# tuned_svm_model <- tune.svm(train[, -ncol(train)], train$Label,
#                             gamma = 10^(-6:-1), cost = 10^(0:3),
#                             kernel = "radial", tunecontrol = tune.control(sampling = "cross"),
#                             validation.x = validation[, -ncol(validation)], validation.y = validation$Label)
# 
# # Extract the best model from the tuning results
# best_svm_model <- tuned_svm_model$best.model
# 
# #     7.3 Use the best model to predict labels for the test data
# test_pred <- predict(best_svm_model, test[, -ncol(test)])
# 
# #     7.4 Convert test_pred to factor with levels "AD" and "CTL"
# test_pred_factor <- factor(test_pred, levels = c(1, 0))
# 
# #     7.5 Evaluate the model's performance on the test data
# confusionMatrix(test_pred_factor, test$Label)
# 
# #     7.4 Convert test_pred to factor with levels "AD" and "CTL"
# test_labels_factor <- factor(test$Label, levels = c(0, 1), labels = c("CTL", "AD"))
# test_pred_factor <- factor(test_pred, levels = c(0, 1), labels = c("CTL", "AD"))
# 
# #     7.5 Evaluate the model's performance on the test data
# metrics <- calculate_metrics(test_pred_factor, test_labels_factor)
# auc <- metrics$AUC
# mcc <- metrics$MCC
# 
# # Print the calculated metrics
# print(paste("AUC:", auc))
# print(paste("MCC:", mcc))
# 
# ####  8.0 Predicting on  test data
# #     8.1 Preprocess the test data, MUST be same as training preprocessing!!
# test_features_prediction <- test_data_ADCTL_clean
# 
# #     8.2 Use the trained SVM model to predict labels for the test data
# test_pred <- predict(tuned_svm_model$best.model, test_features_prediction)
# 
# #     8.3 Convert test_pred to factor with levels "AD" and "CTL"
# test_pred_factor <- factor(test_pred, levels = c(1, 0), labels = c("AD", "CTL"))
# 
# #     8.4 Print the predicted labels for the test data
# print(test_pred_factor)
# 
# 
# 
# 




# #############################################
# #                   MCICTL
# ######### DATASET 3 EXPERIMENTAL ############
# #############################################
# ####  0.2 Reading data
# train_data_MCICTL <- read.csv(MCICTL_train_dir, header = TRUE)
# test_data_MCICTL <- read.csv(MCICTL_test_dir, header = TRUE)
# dim(train_data_MCICTL)
# dim(test_data_MCICTL)
# 
# #     3.1 Dropping the id col in all datasets
# train_data_MCICTL <- train_data_MCICTL[, -1] 
# test_data_MCICTL <- test_data_MCICTL[, -1]
# 
# #     3.2 Encoding label col in training data
# train_data_MCICTL$Label <- ifelse(train_data_MCICTL$Label == "MCI", 1, 0)
# 
# # Separate the features and labels
# features <- train_data_MCICTL[, -ncol(train_data_MCICTL)]
# labels <- train_data_MCICTL$Label
# 
# # Convert the labels to a numeric binary format (0 and 1)
# labels <- as.numeric(as.factor(labels)) - 1
# 
# # Perform feature selection using Lasso regularization
# lasso_model <- cv.glmnet(as.matrix(features), labels, family = "binomial", alpha = 1)
# 
# # Extract the selected features
# selected_features <- as.matrix(features)[, which(coef(lasso_model, s = "lambda.min") != 0)]
# selected_features <- selected_features[, 1:10]
# 
# # Perform dimensional reduction using PCA
# reduced_data <- prcomp(selected_features, center = TRUE, scale. = TRUE)$x
# 
# # Split the data back into training and testing sets
# train_data_MCICTL <- reduced_data[1:nrow(train_data_MCICTL), ]
# dim(train_data_MCICTL)
# 
# # Combine the splitting of the reduced data
# trainIndex <- createDataPartition(labels, p = 0.7, list = FALSE)
# train <- reduced_data[trainIndex, ]
# valid_test <- reduced_data[-trainIndex, ]
# 
# validIndex <- createDataPartition(1:nrow(valid_test), p = 0.5, list = FALSE)
# validation <- valid_test[validIndex, ]
# validation_labels <- labels[-trainIndex][validIndex]  # Extract labels using row indices
# 
# test <- valid_test[-validIndex, ]
# 
# # Convert the data frame to matrix for the training data
# train_matrix <- as.matrix(train)
# train_labels <- labels[trainIndex]
# 
# # Set up the control parameters for 10-fold cross-validation
# ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
# 
# # Convert the train_labels to a factor with two levels
# train_labels <- factor(train_labels, levels = c(0, 1))
# 
# 
# # Train the GBM model
# gbm_model <- train(train_matrix, train_labels,
#                    method = "gbm",
#                    trControl = ctrl,
#                    verbose = FALSE)
# 
# 
# # Perform predictions on the validation data
# validation_predictions <- predict(gbm_model, validation)
# 
# # Convert the validation_labels to a factor with two levels
# validation_labels <- factor(validation_labels, levels = c(0, 1))
# 
# # Evaluate the model on the validation data
# validation_results <- confusionMatrix(validation_predictions, validation_labels)
# print(validation_results)
# 
# #     7.4 Convert test_pred to factor with levels "MCI" and "CTL"
# test_labels_factor <- factor(validation_predictions, levels = c(0, 1), labels = c("MCI", "CTL"))
# test_pred_factor <- factor(validation_labels, levels = c(0, 1), labels = c("MCI", "CTL"))
# 
# #     7.5 Evaluate the model's performance on the test data
# calculate_metrics(test_pred_factor, test_labels_factor, true_labels_value='MCI', false_labels_value='CTL')
# 
# 
# # Perform the same feature selection using the Lasso model
# selected_test_features <- as.matrix(test_data_MCICTL)[, which(coef(lasso_model, s = "lambda.min") != 0)]
# 
# # Perform the same dimensional reduction using PCA
# reduced_test_data <- predict(prcomp(selected_features, center = TRUE, scale. = TRUE), newdata = selected_test_features)
# dim(reduced_test_data)
# 
# # Perform predictions on the test data
# test_predictions_prob <- predict(gbm_model, reduced_test_data, type='prob')
# test_predictions <- predict(gbm_model, reduced_test_data)
# 
# # Convert the test predictions to a factor with two levels
# # test_predictions <- factor(test_predictions, levels = c(0, 1))
# test_pred_factor <- factor(test_predictions, levels = c(1, 0), labels = c("MCI", "CTL"))
# 
# # Print the test predictions
# print(test_predictions)
# print(test_predictions_prob)









#############################################
#                   MCICTL
######### DATASET 3 EXPERIMENTAL ############
#############################################
####  0.2 Reading data
train_data_MCICTL <- read.csv(MCICTL_train_dir, header = TRUE)
test_data_MCICTL <- read.csv(MCICTL_test_dir, header = TRUE)
dim(train_data_MCICTL)
dim(test_data_MCICTL)

#     3.1 Dropping the id col in all datasets
train_data_MCICTL <- train_data_MCICTL[, -1]
test_data_MCICTL <- test_data_MCICTL[, -1]

#     3.2 Encoding label col in training data
train_data_MCICTL$Label <- ifelse(train_data_MCICTL$Label == "MCI", 1, 0)

#     3.0 Feature selection using correlation on ADCTL
#     3.1 Compute the correlation matrix between the variables in the dataset
cor_matrix <- cor(train_data_MCICTL)

#     3.2 Identify pairs of features with a correlation coefficient above 0.7
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.7)

#     3.3 Remove the problematic features from the dataset
train_data_MCICTL_clean <- train_data_MCICTL[,-highly_correlated]
test_data_MCICTL_clean <- test_data_MCICTL[,-highly_correlated]

#     3.4 Convert the labels to a factor
train_data_MCICTL_clean$Label <- as.factor(train_data_MCICTL_clean$Label)

#     4.0 Recursive Feature Elimination
#     4.1 Perform Recursive Feature Elimination (RFE) using caret
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)  # Use random forest as the underlying model
result <- rfe(train_data_MCICTL_clean[, -ncol(train_data_MCICTL_clean)], train_data_MCICTL_clean$Label, sizes = c(1:ncol(train_data_MCICTL_clean)-1), rfeControl = ctrl)

#     4.2 Extract the top variables from the RFE results
selected_features <- result$optVariables[1:7]

#     4.3 Subset the training and test sets with the selected features
train_data_MCICTL_clean <- train_data_MCICTL_clean[, c(selected_features, "Label")]
test_data_MCICTL_clean <- test_data_MCICTL_clean[, selected_features]

dim(train_data_MCICTL_clean)
dim(test_data_MCICTL_clean)

#     4.4 Convert the labels to a factor
train_data_MCICTL_clean$Label <- as.factor(train_data_MCICTL_clean$Label)

# # Separate the features and labels
# features <- train_data_MCICTL[, -ncol(train_data_MCICTL)]
# labels <- train_data_MCICTL$Label
#
# # Convert the labels to a numeric binary format (0 and 1)
# labels <- as.numeric(as.factor(labels)) - 1

# # Perform feature selection using Lasso regularization
# lasso_model <- cv.glmnet(as.matrix(features), labels, family = "binomial", alpha = 1)
#
# # Extract the selected features
# selected_features <- as.matrix(features)[, which(coef(lasso_model, s = "lambda.min") != 0)]

# Perform dimensional reduction using PCA
# reduced_data <- prcomp(selected_features, center = TRUE, scale. = TRUE)$x

# Split the data back into training and testing sets
# train_data_MCICTL <- reduced_data[1:nrow(train_data_MCICTL), ]
# dim(train_data_MCICTL)

# Combine the splitting of the reduced data
trainIndex <- createDataPartition(train_data_MCICTL_clean$Label, p = 0.7, list = FALSE)
train <- train_data_MCICTL_clean[trainIndex, ]
valid_test <- train_data_MCICTL_clean[-trainIndex, ]

validIndex <- createDataPartition(1:nrow(valid_test), p = 0.5, list = FALSE)
validation <- valid_test[validIndex, ]

# validation_labels <- labels[-trainIndex][validIndex]  # Extract labels using row indices
test <- valid_test[-validIndex, ]

#     5.2 Convert the labels to factors for all splits
train$Label <- as.factor(train$Label)
validation$Label <- as.factor(validation$Label)
test$Label <- as.factor(test$Label)


# Convert the data frame to matrix for the training data
# train_matrix <- as.matrix(train)
# train_labels <- labels[trainIndex]

# Set up the control parameters for 10-fold cross-validation
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Convert the train_labels to a factor with two levels
# train_labels <- factor(train_labels, levels = c(0, 1))

# Train the GBM model
# gbm_model <- train(train_matrix, train_labels,
#                    method = "gbm",
#                    trControl = ctrl,
#                    verbose = FALSE)
#     6.0 Training dataset ADCTL
#     6.1 Train an SVM classifier using the training set
svm_model <- svm(train[, -ncol(train)], train$Label,
                 trControl = ctrl, probability = TRUE)

####  6.2 Tune the SVM hyper-parameters using the validation set
tuned_svm_model <- tune.svm(train[, -ncol(train)], train$Label,
                            gamma = 10^(-6:-1), cost = 10^(0:3),
                            kernel = "radial", tunecontrol = tune.control(sampling = "cross"),
                            validation.x = validation[, -ncol(validation)], validation.y = validation$Label,
                            probability = TRUE)

#     6.3 Extract the best model from the tuning results
best_svm_model <- tuned_svm_model$best.model


#     7.0 Evaluation on test set
#     7.1 Use the best model to predict labels for the test set of the training data
test_pred <- predict(best_svm_model, test[, -ncol(test)], probability = TRUE)

#     7.2 Convert test_pred to factor with levels "MCI" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0), labels = c("MCI", "CTL"))

#     7.3 Get the predicted probabilities for each class
predicted_probabilities <- attr(test_pred, "probabilities")

print(data.frame(Label = test_pred_factor,
                 MCI_Probability = predicted_probabilities[, 1],
                 CTL_Probability = predicted_probabilities[, 0]))


#     7.4 Convert the labels to factors with levels "AD" and "CTL"
test_labels_factor <- factor(test$Label, levels = c(1, 0), labels = c("MCI", "CTL"))
confusionMatrix(test_pred_factor, test_labels_factor)

#     7.5 Evaluate the model's performance on the test set
calculate_metrics(test_pred_factor, test_labels_factor, true_labels_value='MCI', false_labels_value='CTL')



####  8.0 Predicting on  test data
#     8.1 Preprocess the test data, MUST be same as training preprocessing!!
test_features_prediction <- test_data_MCICTL_clean

#     8.2 Use the trained SVM model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features_prediction, probability = TRUE)

#     8.3 Convert test_pred to factor with levels "MCI" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0), labels = c("MCI", "CTL"))

#     8.4 Print the predicted labels for the test data
print(test_pred_factor)

#     8.4 Get the predicted probabilities for each class
predicted_probabilities <- attr(test_pred, "probabilities")

print(data.frame(Label = test_pred_factor,
                 MCI_Probability = predicted_probabilities[, 1],
                 CTL_Probability = predicted_probabilities[, 0]))







