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

calculate_metrics <- function(predicted_labels, actual_labels) {
  library(pROC)  # Load the pROC package for AUC calculation
  
  # Convert factor labels to numeric
  predicted_numeric <- as.numeric(predicted_labels) - 1
  actual_numeric <- as.numeric(actual_labels) - 1
  
  # Calculate AUC
  roc_obj <- roc(actual_numeric, predicted_numeric)
  auc <- auc(roc_obj)
  
  # Calculate MCC
  TP <- sum(predicted_labels == "AD" & actual_labels == "AD")
  TN <- sum(predicted_labels == "CTL" & actual_labels == "CTL")
  FP <- sum(predicted_labels == "AD" & actual_labels == "CTL")
  FN <- sum(predicted_labels == "CTL" & actual_labels == "AD")
  
  # Check for division by zero
  if (TP + FP == 0 || TP + FN == 0 || TN + FP == 0 || TN + FN == 0) {
    mcc <- NaN
    print("HERE")
  } else {
    mcc <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  }
  
  # Return the calculated metrics
  return(list(AUC = auc, MCC = mcc))
}



####  0.2 Reading data
train_data_ADCTL <- read.csv(ADCTL_train_dir, header = TRUE)
test_data_ADCTL <- read.csv(ADCTL_test_dir, header = TRUE)

#############################################
#                   ADCTL
######### DATASET 1 EXPERIMENTAL ############
#############################################

####  3.0 Pre-processing the data
#     3.1 Dropping the id col in all datasets
####  3.0 Pre-processing the data
train_data_ADCTL <- train_data_ADCTL[, -1] 
test_data_ADCTL <- test_data_ADCTL[, -1]

dim(train_data_ADCTL)
dim(test_data_ADCTL)

#     3.2 Encoding label col in training data
train_data_ADCTL$Label <- ifelse(train_data_ADCTL$Label == "AD", 1, 0)

#     3.3 Checking if the data set has the non-numeric columns
non_numeric_cols <- names(train_data_ADCTL)[!sapply(train_data_ADCTL, is.numeric)]
print(non_numeric_cols)

#     4.0 Feature selection using correlation on ADCTL
#     4.1 Compute the correlation matrix between the variables in the dataset
cor_matrix <- cor(train_data_ADCTL)

#     4.2 Identify pairs of features with a correlation coefficient above 0.7
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.7)

#     4.3 Remove the problematic features from the dataset
train_data_ADCTL_clean <- train_data_ADCTL[,-highly_correlated]
test_data_ADCTL_clean <- test_data_ADCTL[,-highly_correlated]

#     5.0 Convert the labels to a factor
train_data_ADCTL_clean$Label <- as.factor(train_data_ADCTL_clean$Label)

#     6.0 Recursive Feature Elimination
#     6.1 Perform Recursive Feature Elimination (RFE) using caret
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)  # Use random forest as the underlying model
result <- rfe(train_data_ADCTL_clean[, -ncol(train_data_ADCTL_clean)], train_data_ADCTL_clean$Label, sizes = c(1:ncol(train_data_ADCTL_clean)-1), rfeControl = ctrl)

# Extract the top 5 variables from the RFE results
selected_features <- result$optVariables[1:5]

#     6.2 Get the selected features
# selected_features <- result$optVariables

#     6.3 Subset the training and test sets with the selected features
train_data_ADCTL_clean <- train_data_ADCTL_clean[, c(selected_features, "Label")]
test_data_ADCTL_clean <- test_data_ADCTL_clean[, selected_features]

dim(train_data_ADCTL_clean)
dim(test_data_ADCTL_clean)

#     7.0 Convert the labels to a factor
train_data_ADCTL_clean$Label <- as.factor(train_data_ADCTL_clean$Label)

################################################################################
####  6.0 Data Splitting
#     6.1 Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data_ADCTL_clean$Label, 
                                  p = 0.7, list = FALSE)
train <- train_data_ADCTL_clean[trainIndex,]
valid_test <- train_data_ADCTL_clean[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

# Convert the labels to factors for all splits
train$Label <- as.factor(train$Label)
validation$Label <- as.factor(validation$Label)
test$Label <- as.factor(test$Label)

################################################################################
####  7.0 Training dataset ADCTL Using SVM
#     7.1 Train an SVM classifier using the training set
svm_model <- svm(train[, -ncol(train)], train$Label,
                 trControl = ctrl)

####  7.2 Tune the SVM hyper-parameters using the validation set
tuned_svm_model <- tune.svm(train[, -ncol(train)], train$Label,
                            gamma = 10^(-6:-1), cost = 10^(0:3),
                            kernel = "radial", tunecontrol = tune.control(sampling = "cross"),
                            validation.x = validation[, -ncol(validation)], validation.y = validation$Label)

# Extract the best model from the tuning results
best_svm_model <- tuned_svm_model$best.model

#     7.3 Use the best model to predict labels for the test data
test_pred <- predict(best_svm_model, test[, -ncol(test)])

#     7.4 Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0))

#     7.5 Evaluate the model's performance on the test data
confusionMatrix(test_pred_factor, test$Label)

#     7.4 Convert test_pred to factor with levels "AD" and "CTL"
test_labels_factor <- factor(test$Label, levels = c(0, 1), labels = c("CTL", "AD"))
test_pred_factor <- factor(test_pred, levels = c(0, 1), labels = c("CTL", "AD"))

#     7.5 Evaluate the model's performance on the test data
metrics <- calculate_metrics(test_pred_factor, test_labels_factor)
auc <- metrics$AUC
mcc <- metrics$MCC

# Print the calculated metrics
print(paste("AUC:", auc))
print(paste("MCC:", mcc))

####  8.0 Predicting on  test data
#     8.1 Preprocess the test data, MUST be same as training preprocessing!!
test_features_prediction <- test_data_ADCTL_clean

#     8.2 Use the trained SVM model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features_prediction)

#     8.3 Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0), labels = c("AD", "CTL"))

#     8.4 Print the predicted labels for the test data
print(test_pred_factor)




