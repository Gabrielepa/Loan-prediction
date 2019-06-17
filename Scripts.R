
#Reading the data from csv file
##please create insert you own path after downloading the dataset that I have uploaded
setwd("C:/Learn/R/8. Machine Learning/Capstone/IDV/Loan-Prediction-with-R-master")
tr <- read.csv('train.csv', header = TRUE)
head(tr)

#Check the data
summary(tr)

#fixing and cleaning some NA data
setwd("C:/Learn/R/8. Machine Learning/Capstone/IDV/Loan-Prediction-with-R-master")
tr <- read.csv(file="train.csv", na.strings=c("", "NA"), header=TRUE) 
library(plyr); library(dplyr)
tr$Dependents <- revalue(tr$Dependents, c("3+"="3"))

#Visualize the data for easy understanding
sapply(tr, function(x) sum(is.na(x)))
#please install the following packages below if you do not have it in your library
library(mice)
library(VIM)
mice_plot <- aggr(tr, col=c('navyblue','red'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(tr), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

par(mfrow=c(2,2))
hist(tr$LoanAmount, 
     main="Histogram for LoanAmount", 
     xlab="Loan Amount", 
     border="blue", 
     col="maroon",
     las=1, 
     breaks=20, prob = TRUE)
boxplot(tr$LoanAmount, col='maroon',xlab = 'LoanAmount', main = 'Box Plot for Loan Amount')

hist(tr$ApplicantIncome, 
     main="Histogram for Applicant Income", 
     xlab="Income", 
     border="blue", 
     col="maroon",
     las=1, 
     breaks=50, prob = TRUE)

boxplot(tr$ApplicantIncome, col='maroon',xlab = 'ApplicantIncome', main = 'Box Plot for Applicant Income')

library(ggplot2)
data(tr, package="lattice")
ggplot(data=tr, aes(x=LoanAmount, fill=Education)) +
  geom_density() +
  facet_grid(Education~.)

#Loan Status by Gender
par(mfrow=c(2,3))
counts <- table(tr$Loan_Status, tr$Gender)
barplot(counts, main="Loan Status by Gender",
        xlab="Gender", col=c("darkgrey","maroon"),
        legend = rownames(counts))

#Loan Status by Education
counts2 <- table(tr$Loan_Status, tr$Education)
barplot(counts2, main="Loan Status by Education",
        xlab="Education", col=c("darkgrey","maroon"),
        legend = rownames(counts2))

#Loan Status by Married
counts3 <- table(tr$Loan_Status, tr$Married)
barplot(counts3, main="Loan Status by Married",
        xlab="Married", col=c("darkgrey","maroon"),
        legend = rownames(counts3))

#Loan Status by Self Employed
counts4 <- table(tr$Loan_Status, tr$Self_Employed)
barplot(counts4, main="Loan Status by Self Employed",
        xlab="Self_Employed", col=c("darkgrey","maroon"),
        legend = rownames(counts4))

#Loan Status by Property_Area
counts5 <- table(tr$Loan_Status, tr$Property_Area)
barplot(counts5, main="Loan Status by Property_Area",
        xlab="Property_Area", col=c("darkgrey","maroon"),
        legend = rownames(counts5))

#Loan Status by Credit_History
counts6 <- table(tr$Loan_Status, tr$Credit_History)
barplot(counts6, main="Loan Status by Credit_History",
        xlab="Credit_History", col=c("darkgrey","maroon"),
        legend = rownames(counts5))

#TYDING DATA
#The mice() function takes care of the imputing process:
imputed_Data <- mice(tr, m=2, maxit = 2, method = 'cart', seed = 500)
tr <- complete(imputed_Data,2) #here I chose the second round of data imputation


#Check missing data again
sapply(tr, function(x) sum(is.na(x)))

#Handling extreme values
tr$LogLoanAmount <- log(tr$LoanAmount)
par(mfrow=c(1,2))
hist(tr$LogLoanAmount, 
     main="Histogram for Loan Amount", 
     xlab="Loan Amount", 
     border="blue", 
     col="maroon",
     las=1, 
     breaks=20, prob = TRUE)
lines(density(tr$LogLoanAmount), col='black', lwd=3)
boxplot(tr$LogLoanAmount, col='maroon',xlab = 'Income', main = 'Box Plot for Applicant Income')


#combine both ApplicantIncome and Co-applicants as total income and then perform log transformation
tr$Income <- tr$ApplicantIncome + tr$CoapplicantIncome
tr$ApplicantIncome <- NULL
tr$CoapplicantIncome <- NULL
tr$LogIncome <- log(tr$Income)
par(mfrow=c(1,2))
hist(tr$LogIncome, 
     main="Histogram for Applicant Income", 
     xlab="Income", 
     border="blue", 
     col="maroon",
     las=1, 
     breaks=50, prob = TRUE)
lines(density(tr$LogIncome), col='black', lwd=3)
boxplot(tr$LogIncome, col='maroon',xlab = 'Income', main = 'Box Plot for Applicant Income')



#Creating train and test dataset
set.seed(42)
y <- tr$Loan_Status
test_index <- createDataPartition(y, times = 1, p = 0.7, list = FALSE)

sample <- sample.int(n = nrow(tr), size = floor(.70*nrow(tr)), replace = F)
trainnew <- tr[test_index, ]
testnew  <- tr[-test_index, ]

#Perform logistic regression model
fit_glm <- glm(Loan_Status ~ Credit_History, data=trainnew, family = "binomial")
p_hat_glm <- predict(fit_glm, testnew)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, "Y", "N"))
confusionMatrix(data = y_hat_glm, reference = testnew$Loan_Status)

#Perform logistic regression model using more variables
fit_glm <- glm(Loan_Status ~ Credit_History+Education+Self_Employed+Property_Area+LogLoanAmount+
                 LogIncome, data=trainnew, family = "binomial")
p_hat_glm <- predict(fit_glm, testnew)
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, "Y", "N"))
confusionMatrix(data = y_hat_glm, reference = testnew$Loan_Status)

#Using random forest model
library(randomForest)
fit_rf <- randomForest(Loan_Status ~ Credit_History+Education+Self_Employed+Property_Area+LogLoanAmount+
                         LogIncome, data = trainnew) 
y_hat_rf <- predict(fit_rf, testnew)
confusionMatrix(data = y_hat_rf, reference = testnew$Loan_Status)

#Random forest by taking into account of the variables importance
set.seed(42) 
fit.forest2 <- randomForest(Loan_Status ~ Credit_History+LogLoanAmount+
                              LogIncome, data=trainnew,importance=TRUE)
fit.forest2
y_hat_rf <- predict(fit.forest2, testnew)
confusionMatrix(data = y_hat_rf, reference = testnew$Loan_Status)


#Training with various models
###NOTE: this codes will take a long time to finish running. I don't recommend you to run it 
#as I have provided the results on the report
models <- c("glm", "lda",  "naive_bayes",  "svmLinear", "qda", 
                "knn", "kknn", 
                "rf", "ranger",  "wsrf", "Rborist", 
                "avNNet", "mlp", "monmlp",
                "adaboost", "gbm",
                "svmRadial", "svmRadialCost", "svmRadialSigma")
library(caret)
library(dslabs)
set.seed(1)

fits <- lapply(models, function(model){ 
	print(model)
	train(Loan_Status ~ Credit_History+Education+Self_Employed+Property_Area+LogLoanAmount+
                    LogIncome , method = model, data = trainnew)
}) 

names(fits) <- models

pred <- sapply(fits, function(object) 
predict(object, newdata = testnew))

acc <- colMeans(pred == testnew$Loan_Status)
acc
