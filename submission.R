list.packages <- c("caret", "glmnet", "rpart", "randomForest", "e1071", "ggplot2", "usdm")
new.packages <- list.packages[!(list.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

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
train_data_ADCTL <- read.csv(ADCTLtrain_dir, header = TRUE)
test_data_ADCTL <- read.csv(ADCTLtest_dir, header = TRUE)

# dropping the id col
train_data_ADCTL <- train_data_ADCTL[, -1] 
test_data_ADCTL <- test_data_ADCTL[, -1]

# encoding label col in training data
train_data_ADCTL$Label <- ifelse(train_data_ADCTL$Label == "AD", 1, 0)

# Checking if the dataset has the non-numeric columns
non_numeric_cols <- names(train_data_ADCTL)[!sapply(train_data_ADCTL, is.numeric)]
cat(non_numeric_cols)

# Compute the correlation matrix between the variables in the dataset
cor_matrix <- cor(train_data_ADCTL)

# Identify pairs of features with a correlation coefficient above 0.7
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.7)

# Remove the problematic features from the dataset
train_data_ADCTL_clean <- train_data_ADCTL[,-highly_correlated]

# Convert the labels to a factor
train_data_ADCTL_clean$Label <- as.factor(train_data_ADCTL_clean$Label)


# Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data_ADCTL_clean$Label, p = 0.7, list = FALSE)
train <- train_data_ADCTL_clean[trainIndex,]
valid_test <- train_data_ADCTL_clean[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

# Perform Recursive Feature Elimination (RFE) using caret
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)  # Use random forest as the underlying model
result <- rfe(train[, -ncol(train)], train$Label, sizes = c(1:ncol(train)-1), rfeControl = ctrl)

# Get the selected features
selected_features <- result$optVariables

# Subset the training, validation, and test sets with the selected features
train_features <- train[, selected_features]
validation_features <- validation[, selected_features]
test_features <- test[, selected_features]

# Define the features and labels
train_labels <- train$Label
validation_labels <- validation$Label
test_labels <- test$Label

# Convert labels to factor with levels "AD" and "CTL"
train_labels_factor <- factor(train_labels, levels = c(1, 0))
validation_labels_factor <- factor(validation_labels, levels = c(1, 0))
test_labels_factor <- factor(test_labels, levels = c(1, 0))

######### SVM Model ######### 
# Train an SVM classifier using the training set
svm_model <- svm(train_features, train_labels_factor,
                 trControl = ctrl)

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
test_features_prediction <- test_data_ADCTL[,-highly_correlated]
test_features_prediction <- test_features_prediction[selected_features]

# Use the trained SVM model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features_prediction)

# Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c(1, 0))

# Print the predicted labels for the test data
print(test_pred_factor)


