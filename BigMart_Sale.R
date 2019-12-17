# clean up the environment and load the required packages

rm(list=ls())
pkgs <- c("data.table","plyr","dplyr","ggplot2","caret","mlr")
sapply(pkgs,require,character.only=TRUE)

# read the data
setwd("")
train <- fread("train.csv",na.strings = c("NA","?"," "))
test <- fread("test.csv",na.strings = c("NA","?"," "))
#check the NAs
sum(is.na(train))
table(is.na(train))
sum(is.na(test))
sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))
str(train);
summary(train);

# check uniques values in each variable
sapply(train, function(x) length(unique(x)))

# visualize the data
ggplot(train, aes(x=Item_Visibility, y= Item_Outlet_Sales))+geom_point(size = 2.5, color="navy") + 
  ggtitle("Item_Visibility Vs Item_Outlet_Sales") +
  xlab("Item Visibility") + ylab("Item Outlet Sales")
# higher visibility does not seem to have higher sales

ggplot(train, aes(Outlet_Identifier, Item_Outlet_Sales)) + 
  geom_bar(stat = "identity", color = "purple") +
  theme(axis.text.x = element_text(angle = 60, vjust = 0.5, color = "black"))  + 
  ggtitle("Outlets vs Total Sales")

#OUT027 has contributed majority of sales 
#OUT10 and OUT19 have contributed the least outlet sales.

ggplot(train, aes(Item_Type, Item_Outlet_Sales)) + 
  geom_bar(stat = "identity", color = "purple") +
  theme(axis.text.x = element_text(angle = 60, vjust = 0.5, color = "black"))  + 
  ggtitle("Item_Type vs Total Sales")

# Fruits & Vegetables and snacks contribute to the highest amount of outlet sales.

ggplot(train, aes(Item_Type, Item_MRP)) +
  geom_boxplot(outlier.color = "red") + 
  theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "navy")) + 
  xlab("Item Type") + ylab("Item MRP") + ggtitle("Item Type vs Item MRP")

# There does not seem to be many outliers, except one outlier point in Health & Hygiene

## Missing Values
# combine the test and train
test$Item_Outlet_Sales <- 0
combined <- rbind(train,test)

# fill the missing values of Item_Weight and Item Visibility=0 with median value
combined$Item_Weight[is.na(combined$Item_Weight)] <- median(combined$Item_Weight, na.rm = TRUE)
combined$Item_Visibility <- ifelse(combined$Item_Visibility == 0,median(combined$Item_Visibility), combined$Item_Visibility) 

# clean outlet size variable
table(combined$Outlet_Size)
class(combined$Outlet_Size)
combined$Outlet_Size <- factor(combined$Outlet_Size)
levels(combined$Outlet_Size)
levels(combined$Outlet_Size)[1] <- "Other";
levels(combined$Outlet_Size)

# clean Item Fat Content Variable
str(combined)
combined$Item_Fat_Content <- plyr::revalue(combined$Item_Fat_Content,c("LF" = "Low Fat", "reg" = "Regular",
                                                                 "low fat" = "Low Fat"))
table(combined$Item_Fat_Content)

## Feature Engineering
# count of Outlet Identifiers
a <- combined%>%
  group_by(Outlet_Identifier)%>%
  tally()
head(a)
names(a)[2] <- "Outlet_Count"
combined <- left_join(combined,a, by = "Outlet_Identifier")

# count of Outlet Item Identifiers
a <- combined%>%
  group_by(Item_Identifier)%>%
  tally()
head(a)
names(a)[2] <- "Item_Count"
combined <- left_join(combined,a, by = "Item_Identifier")

# Outler Years
combined <- combined%>%
  mutate(Outlet_Year = 2019 - combined$Outlet_Establishment_Year)

#Item Category can be derived from Item Type Identifier
combined$Item_Type_New <- substr(combined$Item_Identifier,1,2)
combined$Item_Type_New <- revalue(combined$Item_Type_New,c("FD"="Food","DR"="Drinks","NC"="Non-Consumable"))
table(combined$Item_Type_New)

# drop identifier variables
combined$Item_Identifier<-NULL
combined$Outlet_Identifier<-NULL
combined$Outlet_Establishment_Year<-NULL

# check the data type of features
str(combined)
combined$Item_Fat_Content <- factor(combined$Item_Fat_Content)
combined$Item_Type <- factor(combined$Item_Type)
combined$Outlet_Location_Type <- factor(combined$Outlet_Location_Type)
combined$Outlet_Type <- factor(combined$Outlet_Type)
combined$Item_Type_New <- factor(combined$Item_Type_New) 

# separate train and test data
d_train <- combined[1:nrow(train),]
d_test <- combined[-(1:nrow(train)),]
################ Modelling #############
# linear reg model
linear_reg_model <- lm(Item_Outlet_Sales~.,data=d_train)
summary(linear_reg_model)

step_model <- step(lm(Item_Outlet_Sales~.,data=d_train), direction = 'backward')

linear_reg_model <- lm(Item_Outlet_Sales ~ Item_Fat_Content + Item_MRP + Outlet_Size + 
                             Outlet_Location_Type + Outlet_Type + Outlet_Count,data=d_train)
summary(linear_reg_model)
par(mfrow=c(2,2))
plot(linear_reg_model)
# looking at the data we can see there is heteroscedasticity issue in the data
step_model <- step(lm(log(Item_Outlet_Sales)~.,data=d_train), direction = 'backward')
linear_reg_model <- lm(log(Item_Outlet_Sales) ~ Item_MRP + Outlet_Size + Outlet_Location_Type + 
                         Outlet_Type + Outlet_Count, data=d_train)
summary(linear_reg_model)
par(mfrow=c(2,2))
plot(linear_reg_model)

library(car)
car::vif(linear_reg_model)
# library(usdm)
# usdm::vif(d_train)
# sqrt(mean((exp(linear_reg_model$fitted.values)-d_train$Item_Outlet_Sales)^2))
test_predict <- exp(predict(linear_reg_model,newdata = d_test))

#### random forest
set.seed(111)
fitControl = trainControl(method = "cv", number = 10)
rfGrid=expand.grid(mtry=c(3:4))
cv_train <- caret::train(Item_Outlet_Sales ~ ., data=d_train, method="rf", trControl = fitControl, tuneGrid = rfGrid)
rf_model <- cv_train$
rf_prediction <- predict(rf_model, newdata = d_train)
varImp(object = rf_model)

#### gbm
fitControl <- trainControl(method = "cv",number = 5)
tune_Grid <-  expand.grid(interaction.depth = 4,n.trees = 150,shrinkage = 0.1,
                          n.minobsinnode = 30)
set.seed(825)
gbm_model <- caret::train(Item_Outlet_Sales~.,data=d_train,method="gbm",
                          trControl = fitControl, tuneGrid = tune_Grid)

gbm_prediction= predict(gbm_model,d_train,type= "raw") 
postResample(pred = gbm_prediction, obs = d_train$Item_Outlet_Sales)
test_predict <- predict(gbm_model,newdata = d_test)

#### xgboost using mlr
str(d_train)
train_xgb <- d_train
str(train_xgb)
dummy_cols <- names(train_xgb)[sapply(train_xgb,function(x) class(x)=="factor")]
train_xgb <- dummy.data.frame(train_xgb, names=dummy_cols, sep="_")
names(train_xgb) <- gsub("[^[:alnum:]\\_]", "", names(train_xgb))
train_xgb_matrix <- data.matrix(train_xgb)

test_xgb <- d_test
str(test_xgb)
dummy_cols <- names(test_xgb)[sapply(test_xgb,function(x) class(x)=="factor")]
test_xgb <- dummy.data.frame(test_xgb, names=dummy_cols, sep="_")
names(test_xgb) <- gsub("[^[:alnum:]\\_]", "", names(test_xgb))
test_xgb_matrix <- data.matrix(test_xgb)

str(train_xgb)
train.task <- makeRegrTask(data = train_xgb,target = "Item_Outlet_Sales")
test.task <- makeRegrTask(data=test_xgb,target = "Item_Outlet_Sales")

set.seed(2002)
listLearners("regr")
xgb_learner <- makeLearner("regr.xgboost",predict.type = "response")
xgb_learner$par.vals <- list(
  objective = "reg:linear",
  eval_metric = "rmse",
  nrounds = 150,
  print_every_n = 50
)

#define hyperparameters for tuning
xg_ps <- makeParamSet( 
  makeIntegerParam("max_depth",lower=3,upper=10),
  makeNumericParam("lambda",lower=0.05,upper=0.5),
  makeNumericParam("eta", lower = 0.01, upper = 0.5),
  makeNumericParam("subsample", lower = 0.50, upper = 1),
  makeNumericParam("min_child_weight",lower=2,upper=10),
  makeNumericParam("colsample_bytree",lower = 0.50,upper = 0.80)
)

#define search function
rancontrol <- makeTuneControlRandom(maxit = 5L) #do 5 iterations

#5 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#tune parameters
xgb_tune <- tuneParams(learner = xgb_learner, task = train.task, resampling = set_cv, par.set = xg_ps, control = rancontrol)

#set optimal parameters
xgb_new <- setHyperPars(learner = xgb_learner, par.vals = xgb_tune$x)

#train model
xgmodel <- train(xgb_new, train.task)

#test model
predict.xg <- predict(xgmodel, train.task)
#make prediction
xg_prediction <- predict.xg$data$response

postResample(pred = gbm_prediction, obs = d_train$Item_Outlet_Sales)

test_predict <- predict(xgmodel,newdata = test_xgb)

# random forest

train.task <- makeRegrTask(data = train_xgb,target = "Item_Outlet_Sales")
test.task <- makeRegrTask(data=test_xgb,target = "Item_Outlet_Sales")

getParamSet("regr.randomForest")

#create a learner
rf <- makeLearner("regr.randomForest", predict.type = "response", 
                  par.vals = list(ntree = 200, mtry = 3))

#set tunable parameters
#grid search to find hyperparameters
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 100),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

#let's do random search for 50 iterations
rancontrol <- makeTuneControlRandom(maxit = 5L)
#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#hypertuning
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = train.task,
                      par.set = rf_param, control = rancontrol, measures = rmse)

#cv accuracy
rf_tune$y
#best parameters
rf_tune$x

#using hyperparameters for modeling
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)

#train a model
rf_model <- train(rf.tree, train.task)

#make predictions
rf_prediction <- predict(rforest, train.task)

postResample(pred = rf_prediction$data$response, obs = d_train$Item_Outlet_Salestlet_Sales)