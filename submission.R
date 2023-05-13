####  0.0 Installing necessary packages
list.packages <- c("caret", "glmnet", "rpart", "randomForest", "e1071", "ggplot2", "usdm", "Boruta", "gbm")
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


####  0.1 Defining directories
ADCTL_train_dir <- "./data/train/ADCTLtrain.csv"
ADMCI_train_dir <- "./data/train/ADMCItrain.csv"
MCICTL_train_dir <- "./data/train/MCICTLtrain.csv"

ADCTL_test_dir <- "./data/test/ADCTLtest.csv"
ADMCI_test_dir <- "./data/test/ADMCItest.csv"
MCICTL_test_dir <- "./data/test/MCICTLtest.csv"

####  0.2 Reading data
train_data_ADCTL <- read.csv(ADCTL_train_dir, header = TRUE)
test_data_ADCTL <- read.csv(ADCTL_test_dir, header = TRUE)

#############################################
#                   ADCTL
######### DATASET 1 EXPERIMENTAL ############
#############################################
####  3.0 Pre-processing the data
#     3.1 Dropping the id col in all datasets
train_data_ADCTL <- train_data_ADCTL[, -1] 
test_data_ADCTL <- test_data_ADCTL[, -1]

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

#     5.0 Convert the labels to a factor
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

####  6.0 Recursive Feature Elimination
#     6.1 Perform Recursive Feature Elimination (RFE) using caret
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)  # Use random forest as the underlying model
result <- rfe(train[, -ncol(train)], train$Label, sizes = c(1:ncol(train)-1), rfeControl = ctrl)

#     6.2 Get the selected features
selected_features <- result$optVariables

#     6.3 Subset the training, validation, and test sets with the selected features
train_features <- train[, selected_features]
validation_features <- validation[, selected_features]
test_features <- test[, selected_features]

#     6.4 Define the features and labels
train_labels <- train$Label
validation_labels <- validation$Label
test_labels <- test$Label

#     6.5 Convert labels to factor with levels "AD" and "CTL"
train_labels_factor <- factor(train_labels, levels = c(1, 0))
validation_labels_factor <- factor(validation_labels, levels = c(1, 0))
test_labels_factor <- factor(test_labels, levels = c(1, 0))

####  7.0 Training dataset ADCTL Using SVM
#     7.1 Train an SVM classifier using the training set
svm_model <- svm(train_features, train_labels_factor,
                 trControl = ctrl)

#     7.2 Tune the SVM hyper-parameters using the validation set
tuned_svm_model <- tune.svm(train_features, train_labels_factor,
                            gamma = 10^(-6:-1), cost = 10^(0:3),
                            kernel = "radial", tunecontrol = tune.control(sampling = "cross"),
                            validation.x = validation_features, validation.y = validation_labels_factor)

#     7.3 Use the tuned model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features)

#     7.4 Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0))

#     7.5 Evaluate the model's performance on the test data
confusionMatrix(test_pred_factor, test_labels_factor)

####  8.0 Predicting on  test data
#     8.1 Preprocess the test data, MUST be same as training preprocessing!!
test_features_prediction <- test_data_ADCTL[,-highly_correlated]
test_features_prediction <- test_features_prediction[selected_features]

#     8.2 Use the trained SVM model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features_prediction)

#     8.3 Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0))

#     8.4 Print the predicted labels for the test data
print(test_pred_factor)



#############################################
#                   ADMCI
######### DATASET 2 EXPERIMENTAL ############
#############################################
####  0.2 Reading data
train_data_ADMCI <- read.csv(ADMCI_train_dir, header = TRUE)
test_data_ADMCI <- read.csv(ADMCI_test_dir, header = TRUE)

#     3.1 Dropping the id col in all datasets
train_data_ADMCI <- train_data_ADMCI[, -1] 
test_data_ADMCI <- test_data_ADMCI[, -1]


#     3.2 Encoding label col in training data
train_data_ADMCI$Label <- ifelse(train_data_ADMCI$Label == "AD", 1, 0)

#     3.3 Checking if the data set has the non-numeric columns
non_numeric_cols <- names(train_data_ADMCI)[!sapply(train_data_ADMCI, is.numeric)]
cat(non_numeric_cols)

####  6.0 Data Splitting
#     6.1 Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data_ADMCI$Label, 
                                  p = 0.7, list = FALSE)
train <- train_data_ADMCI[trainIndex,]
valid_test <- train_data_ADMCI[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

# Convert the data frame to matrix
train_matrix <- as.matrix(train[, -ncol(train)])
train_labels <- train$Label

# Perform feature selection using the Boruta algorithm
boruta_result <- Boruta(train_matrix, train_labels)

# Print the selected features
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)

# Select the features from the training and testing datasets
train_data_selected <- train[, c(selected_features, "Label")]
valid_data_selected <- validation[, c(selected_features, "Label")]
test_data_selected <- test[, c(selected_features, "Label")]

# Convert the selected features to a matrix
train_matrix <- as.matrix(train_data_selected[, -ncol(train_data_selected)])
train_labels <- train_data_selected$Label

# Train the random forest model using cross-validation:
ctrl <- trainControl(method = "cv", number = 10)

# Train the random forest model with hyperparameter tuning
rf_model <- train(
  x = train_matrix,
  y = as.factor(train_labels),
  method = "rf",
  trControl = ctrl,
  tuneParams = list(
    mtry = c(2, 4, 6),  # Range of values for mtry
    ntree = c(100, 200, 300)  # Range of values for ntree
  )
)


# Convert the selected features to a matrix
valid_matrix <- as.matrix(valid_data_selected[, -ncol(valid_data_selected)])
valid_labels <- valid_data_selected$Label

# Validate the model on the validation data
valid_predictions <- predict(rf_model, newdata = valid_matrix, type = "prob")
valid_predictions_binary <- ifelse(valid_predictions[, 2] > 0.5, 1, 0)

# Calculate the accuracy on the validation data
valid_accuracy <- mean(valid_predictions_binary == valid_labels)
cat(valid_accuracy)

# Convert the selected features to a matrix
test_matrix <- as.matrix(test_data_selected[, -ncol(test_data_selected)])
test_labels <- test_data_selected$Label

# Make predictions on the test data
test_predictions <- predict(rf_model, newdata = test_matrix, type = "prob")
test_predictions_binary <- ifelse(test_predictions[, 2] > 0.5, 1, 0)

# Calculate the accuracy on the test data
test_accuracy <- mean(test_predictions_binary == test_labels)
cat(test_accuracy)

# Convert the predictions and labels to factors with the same levels
test_predictions_factor <- as.factor(test_predictions_binary)
test_labels_factor <- as.factor(test_labels)

# Create the confusion matrix
confusionMatrix(test_predictions_factor, test_labels_factor)

# Select the features from the test data using the same selected features
test_data_selected <- test_data_ADMCI[, selected_features]

# Convert the selected features to a matrix
test_matrix <- as.matrix(test_data_selected)

# Make predictions on the test data
test_predictions <- predict(rf_model, newdata = test_matrix, type = "prob")
test_predictions_binary <- ifelse(test_predictions[, 2] > 0.5, 1, 0)

# Print the test predictions
print(test_predictions_binary)


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

# Separate the features and labels
features <- train_data_MCICTL[, -ncol(train_data_MCICTL)]
labels <- train_data_MCICTL$Label

# Convert the labels to a numeric binary format (0 and 1)
labels <- as.numeric(as.factor(labels)) - 1

# Perform feature selection using Lasso regularization
lasso_model <- cv.glmnet(as.matrix(features), labels, family = "binomial", alpha = 1)

# Extract the selected features
selected_features <- as.matrix(features)[, which(coef(lasso_model, s = "lambda.min") != 0)]

# Perform dimensional reduction using PCA
reduced_data <- prcomp(selected_features, center = TRUE, scale. = TRUE)$x

# Split the data back into training and testing sets
train_data_MCICTL <- reduced_data[1:nrow(train_data_MCICTL), ]
dim(train_data_MCICTL)

# Combine the splitting of the reduced data
trainIndex <- createDataPartition(labels, p = 0.7, list = FALSE)
train <- reduced_data[trainIndex, ]
valid_test <- reduced_data[-trainIndex, ]

validIndex <- createDataPartition(1:nrow(valid_test), p = 0.5, list = FALSE)
validation <- valid_test[validIndex, ]
validation_labels <- labels[-trainIndex][validIndex]  # Extract labels using row indices

test <- valid_test[-validIndex, ]

# Convert the data frame to matrix for the training data
train_matrix <- as.matrix(train)
train_labels <- labels[trainIndex]

# Set up the control parameters for 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Convert the train_labels to a factor with two levels
train_labels <- factor(train_labels, levels = c(0, 1))

# Train the GBM model
gbm_model <- train(train_matrix, train_labels,
                   method = "gbm",
                   trControl = ctrl,
                   verbose = FALSE)


# Perform predictions on the validation data
validation_predictions <- predict(gbm_model, validation)

# Convert the validation_labels to a factor with two levels
validation_labels <- factor(validation_labels, levels = c(0, 1))

# Evaluate the model on the validation data
validation_results <- confusionMatrix(validation_predictions, validation_labels)
print(validation_results)

# Perform the same feature selection using the Lasso model
selected_test_features <- as.matrix(test_data_MCICTL)[, which(coef(lasso_model, s = "lambda.min") != 0)]

# Perform the same dimensional reduction using PCA
reduced_test_data <- predict(prcomp(selected_features, center = TRUE, scale. = TRUE), newdata = selected_test_features)
dim(reduced_test_data)

# Perform predictions on the test data
test_predictions <- predict(gbm_model, reduced_test_data)

# Convert the test predictions to a factor with two levels
test_predictions <- factor(test_predictions, levels = c(0, 1))

# Print the test predictions
print(test_predictions)

