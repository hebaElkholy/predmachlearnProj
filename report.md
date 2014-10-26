# Prediction Machine Learning Assig.
Heba Elkholy  
10/26/2014  

This report depends on the data 


## Reading the testing and training data

Taking a quick look on the raw data, it was noticed that NA strings were either "NA", empty values or "#DIV/0!" so that was fed to the `read.csv` function

```r
dir = "/home/heba/Documents/gitRepos/predmachlearnProj";
setwd(dir);
train.file = "Data/pml-training.csv"
test.file = "Data/pml-testing.csv"
if (!file.exists(train.file)) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = train.file, method="curl")
}
if (!file.exists(test.file)) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = test.file, method="curl")
}
training <- read.csv(train.file, na.strings=c("NA","#DIV/0!"), stringsAsFactors = F)
testing <- read.csv(test.file, na.strings=c("NA","#DIV/0!"), stringsAsFactors = F)
training$classe <- factor(training$classe)
```

By inspecting the data, the first 7 columns were user specific, like the user name and the time and date of the reading, this was removed from the dataset that will be used for training. Also, the columns having complete measurements (non NA enteries) are the ones which will give the most accurate prediction results, so those are the ones that will be used.


```r
training.cleaned <- training[ ,8:ncol(training)]
testing.cleaned <- testing[, 8:ncol(testing)]
IDX <- NULL
for (i in seq(ncol(training.cleaned))){
  if (sum(is.na(training.cleaned[,i]))==0){
    IDX <- c(IDX, i)
  }
}
training.complete <- training.cleaned[, IDX]
testing.complete <- testing.cleaned[ ,IDX]
## Predicting
```


The data will be divided into training and validation using a 70, 30 ratio as shown

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(y = training.complete$classe, p = 0.7, list = FALSE)
training.complete.70 <- training.complete[inTrain, ]
training.complete.30 <- training.complete[-inTrain, ]
```

The prediction model will be training using random forests, due to the big size of the data and the number of features, parallelization is done as shown:

```r
library(foreach)
library(doParallel)
```

```
## Loading required package: iterators
## Loading required package: parallel
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(123)
registerDoParallel()
model <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
        randomForest(x=training.complete.70[,1:ncol(training.complete.70)-1], y=training.complete.70$classe, ntree=ntree)
}
```

## Model Validation

In sample validation:


```r
confusionMatrix(predict(model, newdata=training.complete.70),training.complete.70$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

As you see the model was 100% accurate to predict all the data it was trained on. 

out of sample validation:


```r
confusionMatrix(predict(model, newdata=training.complete.30),training.complete.30$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    1 1136    9    0    0
##          C    0    1 1015    8    0
##          D    0    0    2  956    8
##          E    0    0    0    0 1074
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9925, 0.9964)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9893   0.9917   0.9926
## Specificity            0.9995   0.9979   0.9981   0.9980   1.0000
## Pos Pred Value         0.9988   0.9913   0.9912   0.9896   1.0000
## Neg Pred Value         0.9998   0.9994   0.9977   0.9984   0.9983
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1930   0.1725   0.1624   0.1825
## Detection Prevalence   0.2846   0.1947   0.1740   0.1641   0.1825
## Balanced Accuracy      0.9995   0.9976   0.9937   0.9948   0.9963
```

The model had accuracy of 99.5% which is pretty good for a prediction model and that is the range of the expected accuracy in the test data. i.e. out of sample error of about 1%.

## Test set results


```r
(res <- predict(model, newdata=testing.complete))
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

### Producing the output files

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
dir = "/home/heba/Documents/gitRepos/predmachlearnProj/results";
setwd(dir);
pml_write_files(res)
```
