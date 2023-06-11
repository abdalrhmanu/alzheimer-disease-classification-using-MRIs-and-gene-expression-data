####  0.0 Installing necessary packages
list.packages <- c("caret", "readxl","glmnet", "rpart", "randomForest", "e1071", "ggplot2", "usdm", "Boruta", "gbm", "pROC")
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
library(readxl)


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

#### With label directories
ADCTL_wl <- "./test/ADCTLtest_wl.csv"
ADMCI_wl <- "./test/ADMCItest_wl.csv"
MCICTL_wl <- "./test/MCICTLtest_wl.csv"

#### prediction results directories
ADCTL_res <- "./submission/0075728_HABIB_ADCTLres.csv.xlsx"
ADMCI_res <- "./submission/0075728_HABIB_ADMCIres.xlsx"
MCICTL_res <- "./submission/0075728_HABIB_MCICTLres.xlsx"

####  1.0 Reading data ADCTL
ADCTL_wl_labels <- read.csv(ADCTL_wl, header = TRUE)
ADCTL_res_res <- read_excel(ADCTL_res)

ADCL_wl_y <- ADCTL_wl_labels$Label
ADCTL_res_y <- ADCTL_res_res$`Predicted Labels`

ADCTL_test_pred_factor <- as.factor(ADCTL_res_y)
ADCTL_test_wl_factor <- as.factor(ADCL_wl_y)

calculate_metrics(predicted_labels=ADCTL_test_pred_factor, actual_labels=ADCTL_test_wl_factor, true_labels_value='AD', false_labels_value='CTL')

####  2.0 Reading data ADMCI
ADMCI_wl_labels <- read.csv(ADMCI_wl, header = TRUE)
ADMCI_res_res <- read_excel(ADMCI_res)

ADMCI_wl_y <- ADMCI_wl_labels$Label
ADMCI_res_y <- ADMCI_res_res$`Predicted Labels`

ADCMI_test_pred_factor <- as.factor(ADMCI_res_y)
ADMCI_test_wl_factor <- as.factor(ADMCI_wl_y)

calculate_metrics(predicted_labels=ADCMI_test_pred_factor, actual_labels=ADMCI_test_wl_factor, true_labels_value='MCI', false_labels_value='AD')

####  3.0 Reading data MCICTL
MCICTL_wl_labels <- read.csv(MCICTL_wl, header = TRUE)
MCICTL_res_res <- read_excel(MCICTL_res)

MCICTL_wl_y <- MCICTL_wl_labels$Label
MCICTL_res_y <- MCICTL_res_res$`Predicted Labels`

MCICTL_test_pred_factor <- as.factor(MCICTL_res_y)
MCICTL_test_wl_factor <- as.factor(MCICTL_wl_y)

calculate_metrics(predicted_labels=MCICTL_test_pred_factor, actual_labels=MCICTL_test_wl_factor, true_labels_value='MCI', false_labels_value='CTL')

