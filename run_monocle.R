#
#source("http://bioconductor.org/biocLite.R")
#biocLite()
#
#biocLite("monocle")

library(monocle)

data <- read.csv('data/CelegansRawCounts.csv', header=FALSE)
data <- as.matrix(data)

ncol(data)
data <- data[,apply(data,2,function(data) !all(data==0))]
ncol(data)

HSMM <- newCellDataSet(data, expressionFamily=negbinomial())



HSMM <- estimateSizeFactors(HSMM)
HSMM <- estimateDispersions(HSMM)
disp_table <- dispersionTable(HSMM)
unsup_clustering_genes <- subset(disp_table, mean_expression >= 0.1)

unsup_clustering_genes

HSMM <- setOrderingFilter(HSMM, unsup_clustering_genes$gene_id)
plot_ordering_genes(HSMM)

