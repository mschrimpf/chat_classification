setwd("/home/adrian/Documents/Studium/Software Engineering/2. Semester/HCI/Hausaufgaben/chat_classification")

input <- read.csv("features.csv", sep="\t", header=FALSE)
rownames(input) <- make.names(input[,1], unique=TRUE)
colnames(input)[1:2] <- c("text", "result")
input <- input[,-1]
input <- input[,c(-68,-69)] #always zero

library("MASS")
r <- lda(formula = result ~ ., data=input)
r