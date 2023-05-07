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

#data <- unique(data)
#sum(is.na(data))
#summary(data)
#table(data$Label)

# Checking if the dataset has missing values (NA)
# Seems there are no missing values
sum(is.na(train_data))

# Split the data into training and testing sets
# set.seed(123)
# trainIndex <- createDataPartition(train_data$Label, p = 0.8, list = FALSE)
# train <- train_data[trainIndex,]
# test <- train_data[-trainIndex,]

features <- train_data[,2:430]
labels <- train_data[,431]

# Convert features to matrix
features_matrix <- as.matrix(features)

# Convert labels to factor with levels "AD" and "CTL"
labels_factor <- factor(labels, levels = c("AD", "CTL"))

fit <- glmnet(features_matrix, labels_factor, family="binomial", alpha = 1)
cv.fit <- cv.glmnet(features_matrix, labels_factor, family="binomial", alpha = 1)

plot(cv.fit)
coef(cv.fit, s = "lambda.min")

selected_features <- which(coef(cv.fit, s = "lambda.min") != 0)[-1]
selected_features

# Train a random forest classifier using the selected features
rf <- randomForest(labels_factor ~ ., data = train_data[, c(selected_features, 431)])

# Get the test features and labels
test_features <- test_data[, 2:430]
test_labels <- test_data[, 431]

# Predict labels for the test data using the trained model
test_pred <- predict(rf, newdata = test_data[, c(selected_features, 431)])

# Convert test_labels to factor with levels "AD" and "CTL"
test_labels_factor <- factor(test_labels, levels = c("AD", "CTL"))

# Convert test_pred to factor with levels "AD" and "CTL"
test_pred_factor <- factor(test_pred, levels = c("AD", "CTL"))

# Evaluate the model's performance on the test data
confusionMatrix(test_pred_factor, test_labels_factor)

