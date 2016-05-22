library("doParallel")
library("foreach")
library("mxnet")
library("imager")
library("mlbench")
source("helperFunctions.R")

#load train img information
driver_details = read.csv("imgs/driver_imgs_list.csv")

#load a batch of training images
train<-loadTrainImgBatch(numOfCores=6, numOfImages=1000, grey=TRUE, resize=TRUE, width=106, height=80, channels=1)

#load all the training images
train<-loadTestImgs(numOfCores=6, grey=TRUE, resize=TRUE, width=106, height=80, channels=1)

train.x<-train[,3:ncol(train)]
train.y<-train[,2]

train.x<-t(train.x)
train.y<-t(train.y)

#create the network
data <- mx.symbol.Variable("data")
conv1 = mx.symbol.Convolution(data=data, kernel=c(3,3), num_filter=20)
relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))

conv2 = mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
output = mx.symbol.SoftmaxOutput(data=fc2)


#begin training

device = mx.gpu()
mx.set.seed(0)

model<-mx.model.FeedForward.create(mynet, X=train.array, y=train.y,
                                  ctx=device, num.round=1, array.batch.size=100,
                                  learning.rate=0.05, momentum=0.9, wd=0.00001,
                                  eval.metric=mx.metric.accuracy,
                                  epoch.end.callback=mx.callback.log.train.metric(100))





