library("randomForest")
library("neuralnet")
library("class")
library("caret")
library("glmnet")

set.seed(42)

setwd("~/Documents/Academics/4B/STAT 441/Kaggle Competition/Data")
data.train <- read.csv(file = "train.csv", stringsAsFactors = FALSE, header = TRUE)
data.train$satisfied <- as.factor(data.train$satisfied)
data.test <- read.csv(file = "test.csv", stringsAsFactors = FALSE, header = TRUE)
variables <- read.csv(file = "variable_preprocessing.csv", stringsAsFactors = FALSE, header = TRUE)

#-PREPROCESSING---------------------------------------------------------------------------------------------
mode <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}

imputecol <- function(c) {
  method <- variables$method[variables$variable == names(c)]
  missingvalues <- variables$missingvalues[variables$variable == names(c)]
  type1 <- variables$type1[variables$variable == names(c)]
  
  y <- ifelse(grepl(missingvalues, unlist(c)) | c == "" , NA, unlist(c))
  if (type1 == "num") {y <- as.numeric(y)}

  if (method == "mean") {
    m <- mean(y, na.rm = TRUE)
  } else if (method == "median") {
    m <- median(y, na.rm = TRUE)
  } else if (method == "mode") {
    m <- mode(y)
  } else if (method == "category") {
    m <- variables$category_val[variables$variable == names(c)]
  }
  
  ifelse(is.na(y), m, y)
}

for (i in 1:ncol(data.test)) {
  data.train[i] <- imputecol(data.train[i])
  data.test[i] <- imputecol(data.test[i])
}

data.train.tree <- subset(data.train, select = "satisfied")
data.train.logreg <- subset(data.train, select = "satisfied")
data.test.tree <- data.frame(row.names = 1:nrow(data.test))
data.test.logreg <- data.frame(row.names = 1:nrow(data.test))

for (i in 1:ncol(data.test)) {
  tree_include <- variables$tree_include[variables$variable == names(data.train[i])]
  tree_asfactor <- variables$tree_asfactor[variables$variable == names(data.train[i])]
  logreg_include <- variables$logreg_include[variables$variable == names(data.train[i])]
  logreg_asfactor <- variables$logreg_asfactor[variables$variable == names(data.train[i])]
  if (tree_include == 1) {
    data.train.tree <- cbind(data.train.tree, data.train[i])
    data.test.tree <- cbind(data.test.tree, data.test[i])
    if (tree_asfactor == 1) {
      data.train.tree[,ncol(data.train.tree)] <- as.factor(data.train.tree[,ncol(data.train.tree)])
      data.test.tree[,ncol(data.test.tree)] <- as.factor(data.test.tree[,ncol(data.test.tree)])
    }
  }
  if (logreg_include == 1) {
    data.train.logreg <- cbind(data.train.logreg, data.train[i])
    data.test.logreg <- cbind(data.test.logreg, data.test[i])
    if (logreg_asfactor == 1) {
      data.train.logreg[,ncol(data.train.logreg)] <- as.factor(data.train.logreg[,ncol(data.train.logreg)])
      data.test.logreg[,ncol(data.test.logreg)] <- as.factor(data.test.logreg[,ncol(data.test.logreg)])
    }
  }
}

#---LOGISTIC REGRESSION------------------------------------------------------------------------------------
id_col <- data.test$id

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

#x <- model.matrix(as.numeric(satisfied) ~., data.train.logreg)[, -1]
#y <- as.numeric(data.train.logreg$satisfied)
#fit_ridge_cv <- cv.glmnet(x, y, family = "binomial", alpha = 0)
#fit_lasso_cv <- cv.glmnet(x, y, family = "binomial", alpha = 1)
#coef(fit_lasso_cv) #To see which values get selected for

start_time <- Sys.time()
model.logreg.elastic <- train(
  satisfied ~ ., data = data.train.logreg,
  method = "glmnet",
  family = "binomial",
  trControl = trainControl("cv", number = 5)
)
end_time <- Sys.time()
print(end_time - start_time)
get_best_result(model.logreg.elastic)

sub6 <- data.frame("id" = data.test$id, "Predicted" = predict(model.logreg.elastic, newdata=data.test.logreg, s ="lambda.min", type = "prob")[,2])
write.csv(sub6, "sub6.csv")

#---RANDOM FORREST-----------------------------------------------------------------------------------------
#Tuning number of trees
model.rf.t1 <- randomForest(satisfied ~., data = data.train.tree, ntree = 800, importance = TRUE)
plot(x = 1:800, model.rf.t1$err.rate[,1], type = "l", col = "blue", ylab = "", xlab = "Number of Trees")
  #selected 700

#Tuning number of variables
model.rf.t2 <- randomForest(satisfied ~., data = data.train.tree, mtry = 2, ntree = 700, importance = TRUE)
model.rf.t3 <- randomForest(satisfied ~., data = data.train.tree, mtry = 4, ntree = 700, importance = TRUE)
model.rf.t4 <- randomForest(satisfied ~., data = data.train.tree, mtry = 8, ntree = 700, importance = TRUE)
model.rf.t5 <- randomForest(satisfied ~., data = data.train.tree, mtry = 16, ntree = 700, importance = TRUE)
model.rf.t6 <- randomForest(satisfied ~., data = data.train.tree, mtry = 32, ntree = 700, importance = TRUE)
model.rf.t7 <- randomForest(satisfied ~., data = data.train.tree, mtry = 64, ntree = 700, importance = TRUE)
model.rf.t8 <- randomForest(satisfied ~., data = data.train.tree, mtry = 128, ntree = 700, importance = TRUE)

mtryerrors <- c(model.rf.t2$err.rate[700,1], 
                model.rf.t3$err.rate[700,1], 
                model.rf.t4$err.rate[700,1], 
                model.rf.t5$err.rate[700,1], 
                model.rf.t6$err.rate[700,1], 
                model.rf.t7$err.rate[700,1], 
                model.rf.t8$err.rate[700,1])

plot(x = c(2,4,8,16,32,64,128), 
     mtryerrors,
     type = "l", col = "blue", ylab = "", xlab = "Number of Trees", xaxt = "n", log = "x")
axis(1, at = c(2,4,8,16,32,64,128), labels = c(2,4,8,16,32,64,128))
  #Selected 32

par(las=2) # make label perpendicular to axis
par(mar=c(5,8,4,2)) # increase y‐axis margin.
barplot(model.rf$importance[order(model.rf$importance[,3],decreasing=TRUE)[1:15],3], horiz=TRUE, 
        main="OOB Importance (Accuracy)", xlab="Variables", cex.names=1)

#sub8 <- data.frame("id" = data.test$id, "Preidcted" = predict(model.rf.t6, data.test.tree, type="prob")[,2])
#write.csv(sub8, "sub8.csv", row.names = FALSE)

#---NEURAL NETWORK-----------------------------------------------------------------------------------------
#nn <- neuralnet(satisfied ~., data = data.train.nn, hidden=c(5,1), err.fct = "ce", linear.output=F)

#---KNN----------------------------------------------------------------------------------------------------
#data.train.knn <- predict(preProcess(data.train.knn, method=c("range"), rangeBounds = c(0, 1)), data.train.knn)
#data.test.knn <- predict(preProcess(data.test.knn, method=c("range"), rangeBounds = c(0, 1)), data.test.knn)

#for (i in 1:100) {
#  y <- knn(data.train.knn[1:5000,], data.train.knn[1:5000,], data.train.knn[1:5000,1], 3)
#}

