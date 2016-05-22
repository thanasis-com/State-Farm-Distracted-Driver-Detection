library("imager")
library("doParallel")
library("foreach")

#load an image using imageR package
loadImg<-function(img, grey=FALSE, resize=FALSE, width=0, height=0){
  im<-load.image(img)
  if(grey)
  {
    im<-grayscale(im) 
  }
  if(resize)
  {
    im = resize(im,width,height)
  }
  return(im)
}


#load a batch of training images
loadTrainImgBatch<-function(numOfCores, numOfImages, grey, resize, width, height, channels=1)
{
  tic <- proc.time()
  
  numOfImages<-round(numOfImages/10)
  
  cl<-makeCluster(numOfCores)
  registerDoParallel(cl)
  trainImages<-
    foreach(cls = 0:9, .packages=c('imager'), .export=c("loadImg"), .combine=rbind, .multicombine=T) %dopar% {
      train_files<-list.files(paste0("imgs/train/c", cls, "/"),"*.*",full.names = T)
      train_files<-train_files[1:numOfImages]
      targets<-c()
      m<-data.frame(matrix(0,nrow=length(train_files),ncol=width*height*channels))
      mi<-1
      for(tf in train_files){
        m[mi,]<-as.numeric(loadImg(tf, grey, resize, width, height))
        targets[mi]<-cls
        mi<-mi + 1
      }             
      df = data.frame(m,stringsAsFactors = FALSE)
      df = cbind("target"=targets,df)
      df = cbind("file"=train_files,df)
      return(df)
    }
  stopCluster(cl)
  
  print(paste0("Loaded train images in: ",round((proc.time() - tic)[3]), " seconds"))
  
  return(trainImages)
}


#load a batch of testing images
loadTestImgs<-function(numOfCores, grey, resize, width, height, channels=1)
{
  tic <- proc.time()
  
  cl<-makeCluster(numOfCores)
  registerDoParallel(cl)
  testImages<-
    foreach(cls = 0:9, .packages=c('imager'), .export=c("loadImg"), .combine=rbind, .multicombine=T) %dopar% {
      train_files<-list.files(paste0("imgs/test/c", cls, "/"),"*.*",full.names = T)
      m<-data.frame(matrix(0,nrow=length(train_files),ncol=width*height*channels))
      mi<-1
      for(tf in train_files){
        m[mi,]<-as.numeric(loadImg(tf, grey, resize, width, height))
        mi<-mi + 1
      }             
      df = data.frame(m,stringsAsFactors = FALSE)
      df = cbind("file"=train_files,df)
      return(df)
    }
  stopCluster(cl)
  
  print(paste0("Loaded test images in: ",round((proc.time() - tic)[3]), " seconds"))
  
  return(testImages)
}