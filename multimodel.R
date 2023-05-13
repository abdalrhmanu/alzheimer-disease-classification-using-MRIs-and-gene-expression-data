# install.packages("reshape") 

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

ADCTLtest_dir <- "./data/test/ADCTLtest.csv"
ADMCItest_dir <- "./data/test/ADMCItest.csv"
MCICTLtest_dir <- "./data/test/MCICTLtest.csv"

# Reading data
train_data <- read.csv(ADCTLtrain_dir, header = TRUE)
test_data <- read.csv(ADCTLtest_dir, header = TRUE)

# Checking if the dataset has the non-numeric columns
non_numeric_cols <- names(train_data)[!sapply(train_data, is.numeric)]
print(non_numeric_cols)

# Converting non-numeric cols to numeric
train_data <- train_data[, -1] # dropping the id col
test_data <- test_data[, -1] # dropping the id col
# train_data$Label <- ifelse(train_data$Label == "AD", 1, 0)

all(sapply(train_data, is.numeric))

# Checking if there are any missing values in the training data --> No missing values
all(sapply(train_data, is.na))

# Scale the numeric columns of the training data
num_cols <- sapply(train_data, is.numeric)
train_data[, num_cols] <- scale(train_data[, num_cols])

# Split the data into training, validation, and test sets
set.seed(123)
trainIndex <- createDataPartition(train_data$Label, p = 0.7, list = FALSE)
train <- train_data[trainIndex,]
valid_test <- train_data[-trainIndex,]

validIndex <- createDataPartition(valid_test$Label, p = 0.5, list = FALSE)
validation <- valid_test[validIndex,]
test <- valid_test[-validIndex,]

# Define the features and labels
train_features <- train[,2:429]
train_labels <- train[,430]
validation_features <- validation[,2:429]
validation_labels <- validation[,430]
test_features <- test[,2:429]
test_labels <- test[,430]

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

# Predicting on test data
# Preprocess the test data
test_features_prediction <- test_data[,2:429]

# Use the trained SVM model to predict labels for the test data
test_pred <- predict(tuned_svm_model$best.model, test_features_prediction)

# Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c("AD", "CTL"))

# Print the predicted labels for the test data
print(test_pred_factor)




