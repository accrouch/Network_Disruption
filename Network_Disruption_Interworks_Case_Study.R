rm(list = ls())

# Adam Crouch
# InterWorks Case Study - Network Disruption 
# Kaggle Competition - https://www.kaggle.com/c/telstra-recruiting-network

library(ggplot2)
library(ggthemes)
library(gmodels)
library(tidyverse)
library(ggthemes)
library(devtools)
library(easyGgplot2)
library(data.table)
library(xgboost)

# read in data
setwd("C:/Users/Adam Crouch/Google Drive/Crouch Professional Docs/Interworks Application/Case Study/")
train <- read_csv("train.csv")
test <- read_csv("test.csv")
event <- read_csv("event_type.csv")
log <- read_csv("log_feature.csv")
resource <- read_csv("resource_type.csv")
severity <- read_csv("severity_type.csv")


# explore data
p1 <- ggplot(event, aes(x=reorder(event_type,event_type, function(x)-length(x)))) + 
  ggtitle("Event Type Freq") + geom_bar() + labs(x="event type", y="count")
p2 <- ggplot(resource, aes(x=reorder(resource_type,resource_type, function(x)-length(x)))) + 
  ggtitle("Resource Type Freq") + geom_bar() + labs(x="resource type", y="count")
p3 <- ggplot(severity, aes(x=reorder(severity_type,severity_type, function(x)-length(x)))) + 
  ggtitle("Severity Type Freq") + geom_bar() + labs(x="severity type", y="count")
p4 <- ggplot(train, aes(fault_severity)) + ggtitle("Fault Type Freq") + geom_bar() +
  labs(x="Fault type", y="count")
ggplot2.multiplot(p1, p2, p3, p4, cols=2)

# removing spaces in var names
event$event_type <- gsub(" ", ".", event$event_type)
log$log_feature <- gsub(" ", ".", log$log_feature)
resource$resource_type <- gsub(" ", ".", resource$resource_type)
severity$severity_type <- gsub(" ", ".", severity$severity_type)


# spreading type categories
# event type
event.n <- event %>%
  spread(event_type, event_type)
event.n[,2:ncol(event.n)] <- ifelse(is.na(event.n[,2:ncol(event.n)]),0,1)
event.n$event.sum <- rowSums(event.n[, 2:ncol(event.n)])
# log feature
log.n <- log %>%
  spread(log_feature, volume, fill = 0)
log.n$log.sum <- rowSums(log.n[, 2:ncol(log.n)])
# resource type
resource.n <- resource %>%
  spread(resource_type, resource_type)
resource.n[,2:ncol(resource.n)] <- ifelse(is.na(resource.n[,2:ncol(resource.n)]),0,1)
resource.n$resource.sum <- rowSums(resource.n[, 2:ncol(resource.n)])
# severity type
severity.n <- severity %>%
  spread(severity_type, severity_type)
severity.n[,2:ncol(severity.n)] <- ifelse(is.na(severity.n[,2:ncol(severity.n)]),0,1)
severity.n$severity.sum <- rowSums(severity.n[, 2:ncol(severity.n)])


# merging train and test
l <- list(train, test)
full <- rbindlist(l, fill = T)
m <- merge(full,severity)
m <- m[order(m$location,m$severity_type),]

# location feature engineer
c=1
for( i in 1:(nrow(m)-1))
{ 
  m$c[i]=c
  if(m$location[i]==m$location[i+1])
    c=c+1
  else
    c=1
  m$c[i+1]=c
}



full=merge(full,m[,c(1,5)],by="id")
full$location <- NULL
full$loc_feat <- full$c
full$c <- NULL

# merging datasets together
full <- Reduce(function(x,y) merge(x,y,all=T), list(full, severity.n, resource.n, log.n, event.n))
full$fault_severity <- as.numeric(full$fault_severity)

# sorting out train and test
train.xg <- full[!is.na(full$fault_severity),]
train.id <- train.xg$id
train.xg <- train.xg[,2:ncol(train.xg)]
test.xg <- full[is.na(full$fault_severity),]
test.xg$fault_severity <- NULL
test.id <- test.xg$id
test.xg <- test.xg[,2:ncol(test.xg)]

table(train$fault_severity)

# creating sparse matrix 
library('Matrix')
sparse_train <- sparse.model.matrix(fault_severity ~ . -fault_severity, data=train.xg)
train_label <- train.xg$fault_severity
table(train.xg$fault_severity)

# training the model
set.seed(555)
xgb <- xgboost(sparse_train, label = train_label, eta = 0.5, max_depth = 15, gamma = 0, nround = 100,
               subsample = 0.75, colsample_bytree = 0.75, num_class = 3, objective = "multi:softprob",
               nthread = 6, eval_metric = 'mlogloss', verbose = 0)


summary(xgb)

ptrain <- predict(xgb, sparse_train)
ptest <- predict(xgb, data.matrix(test.xg))

test_xgb <- as.data.frame(matrix(ptest, nrow(test.xg), 3, byrow=T))

submission <- cbind(test.id,test_xgb)
colnames(submission) <- c("id","predict_0","predict_1","predict_2")

write.csv(submission, "submit.csv", row.names=F)


