## To install the package
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("SIMLR", version = "3.8")

library(SIMLR)

setwd("~/repos/GCB536/")
dat <- read.csv("data/CelegansRawCounts.csv")
true.labels <- read.csv("data/Labels.csv")


for(number.clusters in 2:30) {

  output <- SIMLR(dat, c=number.clusters)
  
  cluster.names <- output$y$cluster
  write(cluster.names, file=paste0("SIMLR_", number.clusters, ".txt"))
}