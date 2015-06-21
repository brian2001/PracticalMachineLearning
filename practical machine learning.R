
# Practical Machine Learning Assignment  
# This submission consist of a Github repo with R markdown and compiled HTML file describing the analysis.
# The machine learning algorithm was applied to the 20 test cases available in the test data with 100% success.

# loading relevant libraries
library(gdata)
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart)
library(rpart.plot)
library(doParallel)

# cleaning up the memory
rm(list=ls())

# make a cluster to train in //
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# loading the training set and test set
TotalTrainingData = read.csv("pml-training.csv", header = TRUE)
TotalTestingData = read.csv("pml-testing.csv", header = TRUE)


# preprocessing training data
# we decide here to filter out columns wihout data or with NAs
# as well let's delete the non relevant data, timestamps, user names, num window..
TotalTrainingData_noNA <- TotalTrainingData[ , ! is.na(TotalTrainingData[1,])  ]
TotalTrainingData_noNAnoZero <- TotalTrainingData_noNA[ , !(TotalTrainingData_noNA[1, ] == "")]
TotalTrainingData_Clean <- TotalTrainingData_noNAnoZero[,8:ncol(TotalTrainingData_noNAnoZero)]

# We proceed to the same preprocessing on test data
TotalTestData_noNA <- TotalTestingData[ , ! is.na(TotalTestingData[1,])  ]
TotalTestData_noNAnoZero <- TotalTestData_noNA[ , !(TotalTestData_noNA[1, ] == "")]
TotalTestData_Clean <- TotalTestData_noNAnoZero[,8:ncol(TotalTestData_noNAnoZero)]


# the final total training set consist of 19622 observations and 53 variables (52 predictors and one class variable to predict)
# let's first set the seed to insure that someone else can reproduce exactly our results
set.seed(337733)

# We want to evaluate our model on out of sample data
# Hence we will split our training set in training and a validation set.
# We choose a 3-Fold set of data for training and cross validation of the model
# it is important to have in each of the fold roughly the same number of each class to predict 
# this is why we set y to TotalTrainingData_Clean$classe
my3folds <- createFolds(y =TotalTrainingData_Clean$classe, k = 3 )

# we check the size of the 3 folds
sapply(my3folds,length)


# we now generate the 3 subtrainingsets and subvalidationsets
# 3-fold will authorize us to train and validate 3 models.
# we take on purpose small training sets and large validation set to avoid overfitting and to speed up the training time.
subTrainingSet1 = TotalTrainingData_Clean[ my3folds[[1]] , ]
subValidationSet1 = TotalTrainingData_Clean[ -my3folds[[1]] , ]
subTrainingSet2 = TotalTrainingData_Clean[ my3folds[[2]] , ]
subValidationSet2 = TotalTrainingData_Clean[ -my3folds[[2]] , ]
subTrainingSet3 = TotalTrainingData_Clean[ my3folds[[3]] , ]
subValidationSet3 = TotalTrainingData_Clean[ -my3folds[[3]] , ]


# We now fit 3 random forest model, and each time we display the best model
# for the purpose of the generation of this html file we comment 2 of the 3 trainings
#modFitRandomForest1 <- train(subTrainingSet1$classe ~ . , method="rf", data = subTrainingSet1)
modFitRandomForest2 <- train(subTrainingSet2$classe ~ . , method="rf", data = subTrainingSet2)
#modFitRandomForest3 <- train(subTrainingSet3$classe ~ . , method="rf", data = subTrainingSet3)


# let's display each of the 3 fitted models :
modFitRandomForest1$finalModel
modFitRandomForest2$finalModel
#modFitRandomForest3$finalModel

# the 3 models provide a very good and consistent accuracy on the training sets
# We now evaluate the 3 fitted models on the validation sets 
#predictedRF1 <- predict(modFitRandomForest1$finalModel,subValidationSet1)
predictedRF2 <- predict(modFitRandomForest2$finalModel,subValidationSet2)
#predictedRF3 <- predict(modFitRandomForest3$finalModel,subValidationSet3)



# let's display the confusion matrix, accuracy on each of the 3 models
#confusionMatrix(data = predictedRF1, subValidationSet1$classe)
confusionMatrix(data = predictedRF2, subValidationSet2$classe)
#confusionMatrix(data = predictedRF3, subValidationSet3$classe)

# the 3 models provide as well a very good and consistent accuracy on the validation sets
# the out of sample accuracy is >98% on the 3 random forest trained models ( on 3-fold splitted data ) 
# hence our out of sample accuracy should be in excess of 98% 
# I'm happy with this result
# thanks for your reading

# The stopCluster is necessary to terminate the extra processes
stopCluster(cl)

