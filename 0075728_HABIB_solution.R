####  0.0 Installing necessary packages
list.packages <- c("caret", "glmnet", "rpart", "randomForest", "e1071", "ggplot2", "usdm", "Boruta", "gbm", "pROC")
new.packages <- list.packages[!(list.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# Install and load the necessary package
library(caret)
library(glmnet)
library(rpart)
library(randomForest)
library(e1071)
library(ggplot2)
library(usdm)
library(Boruta)
library(gbm)
library(pROC)  



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


#############################################
#                   ADCTL
######### DATASET 1 EXPERIMENTAL ############
#############################################

####  1.0 Reading data
train_data_ADCTL <- read.csv(ADCTL_train_dir, header = TRUE)
test_data_ADCTL <- read.csv(ADCTL_test_dir, header = TRUE)

####  2.0 Pre-processing the data
#     2.1 Dropping the id col in all datasets
train_data_ADCTL <- train_data_ADCTL[, -1] 
test_data_ADCTL <- test_data_ADCTL[, -1]

dim(train_data_ADCTL)
dim(test_data_ADCTL)

#     2.2 Encoding label col in training data
train_data_ADCTL$Label <- ifelse(train_data_ADCTL$Label == "AD", 1, 0)

#     2.3 Checking if the data set has the non-numeric columns
non_numeric_cols <- names(train_data_ADCTL)[!sapply(train_data_ADCTL, is.numeric)]
print(non_numeric_cols)

#     3.0 Feature selection using correlation on ADCTL
#     3.1 Compute the correlation matrix between the variables in the dataset
cor_matrix <- cor(train_data_ADCTL)

#     3.2 Identify pairs of features with a correlation coefficient above 0.7
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.7)

#     3.3 Remove the problematic features from the dataset
train_data_ADCTL_clean <- train_data_ADCTL[,-highly_correlated]
test_data_ADCTL_clean <- test_data_ADCTL[,-highly_correlated]

#     3.4 Convert the labels to a factor
train_data_ADCTL_clean$Label <- as.factor(train_data_ADCTL_clean$Label)

#     4.0 Recursive Feature Elimination
#     4.1 Perform Recursive Feature Elimination (RFE) using caret
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)  # Use random forest as the underlying model
result <- rfe(train_data_ADCTL_clean[, -ncol(train_data_ADCTL_clean)], train_data_ADCTL_clean$Label, sizes = c(1:ncol(train_data_ADCTL_clean)-1), rfeControl = ctrl)

#     4.2 Extract the top 5 variables from the RFE results
selected_features <- result$optVariables[1:5]

#     4.3 Subset the training and test sets with the selected features
train_data_ADCTL_clean <- train_data_ADCTL_clean[, c(selected_features, "Label")]
test_data_ADCTL_clean <- test_data_ADCTL_clean[, selected_features]

dim(train_data_ADCTL_clean)
dim(test_data_ADCTL_clean)

#     4.4 Convert the labels to a factor
train_data_ADCTL_clean$Label <- as.factor(train_data_ADCTL_clean$Label)


#     5.0 Data Splitting
#     5.1 Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data_ADCTL_clean$Label, 
                                  p = 0.7, list = FALSE)
train <- train_data_ADCTL_clean[trainIndex,]
valid_test <- train_data_ADCTL_clean[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

#     5.2 Convert the labels to factors for all splits
train$Label <- as.factor(train$Label)
validation$Label <- as.factor(validation$Label)
test$Label <- as.factor(test$Label)

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

#     7.2 Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0), labels = c("AD", "CTL"))

#     7.3 Get the predicted probabilities for each class
predicted_probabilities <- attr(test_pred, "probabilities")

print(data.frame(Label = test_pred_factor,
                 AD_Probability = predicted_probabilities[, 1],
                 CTL_Probability = predicted_probabilities[, 0]))

#     7.4 Convert the labels to factors with levels "AD" and "CTL"
test_labels_factor <- factor(test$Label, levels = c(1, 0), labels = c("AD", "CTL"))
confusionMatrix(test_pred_factor, test_labels_factor)

#     7.5 Evaluate the model's performance on the test set
calculate_metrics(test_pred_factor, test_labels_factor, true_labels_value='AD', false_labels_value='CTL')


####  8.0 Predicting on  test data
#     8.1 Preprocess the test data, MUST be same as training preprocessing!!
test_features_prediction <- test_data_ADCTL_clean

#     8.2 Use the trained SVM model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features_prediction, probability = TRUE)

#     8.3 Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0), labels = c("AD", "CTL"))

#     8.4 Print the predicted labels for the test data
print(test_pred_factor)

#     8.4 Get the predicted probabilities for each class
predicted_probabilities <- attr(test_pred, "probabilities")

print(data.frame(Label = test_pred_factor,
                 AD_Probability = predicted_probabilities[, 1],
                 CTL_Probability = predicted_probabilities[, 0]))


#############################################
#                   ADMCI
######### DATASET 2 EXPERIMENTAL ############
#############################################
####  1.0 Reading data
train_data_ADMCI <- read.csv(ADMCI_train_dir, header = TRUE)
test_data_ADMCI <- read.csv(ADMCI_test_dir, header = TRUE)

####  2.0 Pre-processing the data
#     2.1 Dropping the id col in all datasets
train_data_ADMCI <- train_data_ADMCI[, -1] 
test_data_ADMCI <- test_data_ADMCI[, -1]

#     2.2 Encoding label col in training data
train_data_ADMCI$Label <- ifelse(train_data_ADMCI$Label == "AD", 1, 0)

#     2.3 Checking if the data set has the non-numeric columns
non_numeric_cols <- names(train_data_ADMCI)[!sapply(train_data_ADMCI, is.numeric)]
cat(non_numeric_cols)

####  3.0 Data Splitting
#     3.1 Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data_ADMCI$Label, 
                                  p = 0.7, list = FALSE)
train <- train_data_ADMCI[trainIndex,]
valid_test <- train_data_ADMCI[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

#     3.2 Convert the data frame to matrix
train_matrix <- as.matrix(train[, -ncol(train)])
train_labels <- train$Label

# Perform feature selection using the Boruta algorithm
# boruta_result <- Boruta(train_matrix, train_labels)

# Print the selected features
# selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)

#     4.1 Perform feature selection using Recursive Feature Elimination with Cross-Validation
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfe_result <- rfe(x = train_matrix, y = train_labels, sizes = c(1:ncol(train_matrix)-1), rfeControl = ctrl)

#     4.2 Extract the top 10 variables from the RFE results
selected_features <- rfe_result$optVariables[1:10]
print(selected_features)

#     4.3 Select the features from the training and testing datasets
train_data_selected <- train[, c(selected_features, "Label")]
valid_data_selected <- validation[, c(selected_features, "Label")]
test_data_selected <- test[, c(selected_features, "Label")]

#     4.4 Convert the selected features to a matrix
train_matrix <- as.matrix(train_data_selected[, -ncol(train_data_selected)])
train_labels <- train_data_selected$Label

#     5.0 Training dataset ADMCI
#     5.1 Train the random forest model using cross-validation:
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

#     5.2 Train the random forest model with hyperparameter tuning
rf_model <- train(
  x = train_matrix,
  y = as.factor(train_labels),
  method = "rf",
  trControl = ctrl,
  tuneParams = list(
    mtry = c(2, 4, 6),  # Range of values for mtry
    ntree = c(100, 200, 300)  # Range of values for ntree
  ),
  preProcess = c("center", "scale"),  # Apply centering and scaling preprocessing
  metric = "Accuracy"  # Use Accuracy as the evaluation metric
)

####  6.0 Predicting on  test data
#     6.1 Convert the selected features to a matrix
valid_matrix <- as.matrix(valid_data_selected[, -ncol(valid_data_selected)])
valid_labels <- valid_data_selected$Label

#     6.2 Validate the model on the validation data
valid_predictions <- predict(rf_model, newdata = valid_matrix, type = "prob")
valid_predictions_binary <- ifelse(valid_predictions[, 2] > 0.5, 1, 0)

#     6.3 Calculate the accuracy on the validation data
valid_accuracy <- mean(valid_predictions_binary == valid_labels)
cat(valid_accuracy)

#     6.4 Convert the selected features to a matrix
test_matrix <- as.matrix(test_data_selected[, -ncol(test_data_selected)])
test_labels <- test_data_selected$Label

#     6.5 Make predictions on the test set
test_predictions <- predict(rf_model, newdata = test_matrix, type = "prob")
test_predictions_binary <- ifelse(test_predictions[, 2] > 0.5, 1, 0)

#     6.6 Calculate the accuracy on the test data
test_accuracy <- mean(test_predictions_binary == test_labels)
cat(test_accuracy)

#     6.7 Convert the predictions and labels to factors with the same levels
test_predictions_factor <- factor(test_predictions_binary, levels = c(0, 1), labels = c("MCI", "AD"))
test_labels_factor <- factor(test_labels, levels = c(0, 1), labels = c("MCI", "AD"))

#     6.8 Create the confusion matrix
confusionMatrix(test_predictions_factor, test_labels_factor)

#     6.9 Convert test_pred to factor with levels "MCI" and "AD"
# test_labels_factor <- factor(test_labels_factor, levels = c(0, 1), labels = c("AD", "MCI"))
# test_pred_factor <- factor(test_predictions_factor, levels = c(0, 1), labels = c("MCI", "AD"))

#     6.10 Evaluate the model's performance on the test set
# calculate_metrics(test_pred_factor, test_labels_factor)
calculate_metrics(test_predictions_factor, test_labels_factor, true_labels_value='MCI', false_labels_value='AD')

####  7.0 Predicting on  test data
#     7.1 Select the features from the test data using the same selected features
test_data_selected <- test_data_ADMCI[, selected_features]

#     7.2 Convert the selected features to a matrix
test_matrix <- as.matrix(test_data_selected)

#     7.3 Make predictions on the test data
test_predictions <- predict(rf_model, newdata = test_matrix, type = "prob")
test_predictions_binary <- ifelse(test_predictions[, 2] > 0.5, 1, 0)

#     7.4 Print the test predictions
print(test_predictions_binary)

test_predictions_binary <- factor(test_predictions_binary, levels = c(0, 1), labels = c("MCI", "AD"))

print(data.frame(
  # ID = test_data_selected$ID,
  Prediction = test_predictions_binary,
  MCI_Probability = test_predictions[, 1],
  AD_Probability = test_predictions[, 2]
))




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

ID_column <- test_data_MCICTL[, 1]
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

print(data.frame(
  ID = ID_column,
  Label = test_pred_factor,
  CTL_Probability = predicted_probabilities[, 1],
  MCI_Probability = predicted_probabilities[, 2]))


