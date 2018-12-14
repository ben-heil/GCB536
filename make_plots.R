library(mclust)
library(gplots)
library(aricode)

setwd("~/repos/GCB536/")
true.labels <- read.csv("data/Labels.csv", header=FALSE)
data <- read.csv("data/CelegansRawCounts.csv", header=FALSE)

cluster.labels <- list()

for(method in c("tsne_kmeans", "log_tsne_kmeans", "lle_kmeans", "log_lle_kmeans", "raw_kmeans", "log_raw_kmeans")) {
  current.cluster.labels <- list()
  for(k in 1:200) {
    if(k==1) {
      current.clusters <- rep("C1", length(true.labels$V1))
    } else{
      current.clusters <- read.csv(paste0("cluster_results/", method, "_labels_c_", k, ".csv"), header=FALSE)
      current.clusters <- current.clusters[,1]
      current.clusters <- current.clusters + 1
      current.clusters <- paste0("C", current.clusters)
    }
    current.cluster.labels[[k]] <- current.clusters
  }
  cluster.labels[[method]] <- current.cluster.labels
}

set.seed(0)
# pca with no centering no rescaling
pca.nocenter <- prcomp(data[,colSums(data) != 0], center=FALSE, scale. = FALSE)
pca.nocenter.kmeans <- lapply(1:200, function(k) paste0("C",kmeans(pca.nocenter$x[,1:15], k, iter.max=100)$cluster))
cluster.labels[["pca_nocenter_kmeans"]] <- pca.nocenter.kmeans

set.seed(0)
# pca with centering and no rescaling
pca.norescaling <- prcomp(data[,colSums(data) != 0], center=TRUE, scale. = FALSE)
pca.norescaling.kmeans <- lapply(1:200, function(k) paste0("C",kmeans(pca.norescaling$x[,1:15], k, iter.max=100)$cluster))
cluster.labels[["pca_norescaling_kmeans"]] <- pca.rescaling.kmeans

set.seed(0)
# pca with centering and rescaling
pca.rescaling <- prcomp(data[,colSums(data) != 0], scale. = TRUE)
pca.rescaling.kmeans <- lapply(1:200, function(k) paste0("C",kmeans(pca.rescaling$x[,1:15], k, iter.max=100)$cluster))
cluster.labels[["pca_rescaling_kmeans"]] <- pca.rescaling.kmeans

# pca with log and no rescalping
set.seed(0)
pca.log.norescaling <- prcomp(log(data[,colSums(data) != 0]+1), scale. = FALSE)
pca.log.norescaling.kmeans <- lapply(1:200, function(k) paste0("C",kmeans(pca.log.norescaling$x[,1:15], k, iter.max=100)$cluster))
cluster.labels[["pca_log_norescaling_kmeans"]] <- pca.log.norescaling.kmeans

# pca with log and rescaling
set.seed(0)
pca.log.rescaling <- prcomp(log(data[,colSums(data) != 0]+1), scale. = TRUE)
pca.log.rescaling.kmeans <- lapply(1:200, function(k) paste0("C",kmeans(pca.log.rescaling$x[,1:15], k, iter.max=100)$cluster))
cluster.labels[["pca_log_rescaling_kmeans"]] <- pca.log.rescaling.kmeans

adj.rand.vals.6.true.labels <- lapply(cluster.labels, function(current.cluster.labels) sapply(current.cluster.labels, function(x) mclust::adjustedRandIndex(true.labels$V1, x)))
adj.rand.vals.25.true.labels <- lapply(cluster.labels, function(current.cluster.labels) sapply(current.cluster.labels, function(x) mclust::adjustedRandIndex(true.labels$V2, x)))
nmi.vals.25.true.labels <- lapply(cluster.labels, function(current.cluster.labels) sapply(current.cluster.labels, function(x) aricode::NMI(true.labels$V2, x)))

save.image(file = "make_plots.RData")

# Plot for raw data
png(filename = "figures/figure_grid_adj_rand.png", width = 1000, height=1000)
par(mfrow=c(2,2))
plot(1:200, adj.rand.vals.25.true.labels$raw_kmeans, pch=19, col="black",xlab="Number of clusters k", ylab="Adjusted Rand Index", ylim=c(0,1))
legend(110, 1,
              legend=c("Raw data"),
              col=c("black"),
       lty=19, cex=0.8, bty="n", lwd=3)
title("K-means on raw data")

# plot for different types of PCA
plot(1:200, adj.rand.vals.25.true.labels$pca_nocenter_kmeans, pch=19, col="darkgreen",xlab="Number of clusters k", ylab="Adjusted Rand Index", ylim=c(0,1))
points(1:200, adj.rand.vals.25.true.labels$pca_rescaling_kmeans, pch=19, col="green")
points(1:200, adj.rand.vals.25.true.labels$pca_log_rescaling_kmeans, pch=19, col="orange")
points(1:200, adj.rand.vals.25.true.labels$pca_log_norescaling_kmeans, pch=19, col="blue")
legend(110, 1,
       legend=c("PCA (no centering, no scaling)", "PCA (centering, scaling)", "PCA  (log, centering, scaling)", "PCA (log, centering, no scaling)"),
       col=c("darkgreen", "green", "orange", "blue"),
       lty=19, cex=0.8, bty="n", lwd=3)
title("K-means on PCA data")
# plot for t-sne
plot(1:200, adj.rand.vals.25.true.labels$tsne_kmeans, pch=19, col="pink",xlab="Number of clusters k", ylab="Adjusted Rand Index", ylim=c(0,1))
points(1:200, adj.rand.vals.25.true.labels$log_tsne_kmeans, pch=19, col="red")
legend(110, 1,
       legend=c("t-SNE, log-transformed", "t-SNE, raw data"),
       col=c("red", "pink"),
       lty=19, cex=0.8, bty="n", lwd=3)
title("K-means on t-SNE data")
# Plot for LLE
plot(1:200, adj.rand.vals.25.true.labels$log_lle_kmeans, pch=19, col="orange",xlab="Number of clusters k", ylab="Adjusted Rand Index", ylim=c(0,1))
points(1:200, adj.rand.vals.25.true.labels$lle_kmeans, pch=19, col="purple")
legend(110, 1,
       legend=c("LLE, log-transformed", "LLE, raw data"),
       col=c("orange", "purple"),
       lty=19, cex=0.8, bty="n", lwd=3)
title("Local linear embedding")
dev.off()
# Plot best method each
png(filename = "figures/method_comparison_adj_rand.png", width = 500, height=500)
plot(1:200, adj.rand.vals.25.true.labels$raw_kmeans, pch=19, col="black",xlab="Number of clusters k", ylab="Adjusted Rand Index", ylim=c(0,1))
points(1:200, adj.rand.vals.25.true.labels$pca_log_norescaling_kmeans, pch=19, col="blue")
points(1:200, adj.rand.vals.25.true.labels$log_tsne_kmeans, pch=19, col="red")
points(1:200, adj.rand.vals.25.true.labels$log_lle_kmeans, pch=19, col="orange")
legend(110, 1,
       legend=c("t-SNE, log-transformed", "PCA, log, no scaling", "LLE, log-transformed", "raw data"),
       col=c("red", "blue", "orange", "black"),
       lty=19, cex=0.8, bty="n", lwd=3)
title("Comparison of methods")
dev.off()

# repeat for normalized mutual information
png(filename = "figures/method_comparison_nmi.png", width = 500, height=500)
plot(1:200, nmi.vals.25.true.labels$raw_kmeans, pch=19, col="black",xlab="Number of clusters k", ylab="Normalized Mutual Information", ylim=c(0,1))
points(1:200, nmi.vals.25.true.labels$pca_log_norescaling_kmeans, pch=19, col="blue")
points(1:200, nmi.vals.25.true.labels$log_tsne_kmeans, pch=19, col="red")
points(1:200, nmi.vals.25.true.labels$log_lle_kmeans, pch=19, col="orange")
legend(110, 1,
       legend=c("t-SNE, log-transformed", "PCA, log, no scaling", "LLE, log-transformed", "raw data"),
       col=c("red", "blue", "orange", "black"),
       lty=19, cex=0.8, bty="n", lwd=3)
title("Comparison of methods")
dev.off()

points(1:200, adj.rand.vals.25.true.labels$pca_kmeans, pch=19, col="blue")
points(1:200, adj.rand.vals.25.true.labels$pca_rescaling_kmeans, pch=19, col="orange")
points(1:200, adj.rand.vals.25.true.labels$pca_log_rescaling_kmeans, pch=19, col="purple")
points(1:200, adj.rand.vals.25.true.labels$pca_log_norescaling_kmeans, pch=19, col="black")
legend(110, 0.55, legend=c("t-SNE kmeans", "log t-SNE kmeans", "PCA kmeans", "PCA (with rescaling) kmeans", "PCA (with log and rescaling) kmeans", "PCA (with log and no rescaling)"),
       col=c("red", "green", "blue", "orange", "purple", "black"), lty=19, cex=0.8)
title("t-SNE k-means vs PCA k-means validated on 25-cell-type labels")

plot(1:200, adj.rand.vals.6.true.labels$tsne_kmeans, pch=19, col="red",xlab="Number of clusters k", ylab="Adjusted Rand Index", ylim=c(0,1))
points(1:200, adj.rand.vals.6.true.labels$log_tsne_kmeans, pch=19, col="green")
points(1:200, adj.rand.vals.6.true.labels$lle_kmeans, pch=19, col="pink")
points(1:200, adj.rand.vals.6.true.labels$pca_kmeans, pch=19, col="blue")
points(1:200, adj.rand.vals.6.true.labels$pca_rescaling_kmeans, pch=19, col="orange")
points(1:200, adj.rand.vals.6.true.labels$pca_log_rescaling_kmeans, pch=19, col="purple")
points(1:200, adj.rand.vals.6.true.labels$pca_log_norescaling_kmeans, pch=19, col="black")
legend(110, 0.55, legend=c("t-SNE kmeans", "log t-SNE kmeans", "PCA kmeans", "PCA (with rescaling) kmeans", "PCA (with log and rescaling) kmeans", "PCA (with log and no rescaling)"),
       col=c("red", "green", "blue", "orange", "purple", "black"), lty=19, cex=0.8)
title("t-SNE k-means vs PCA k-means validated on 6-cell-type labels")

thing <- table(true.labels$V1, cluster.labels$tsne_kmeans[[6]])

heatmap.2(
          as.matrix(thing),
          cellnote=as.matrix(thing),
          density.info = 'none',
          trace='none',
          dendrogram = 'both', 
          sepcolor='black', 
          col=colorRampPalette(c("white", "red")), 
          notecol="black", 
          key=TRUE, 
          margins=c(12,8), 
          scale='none',
          breaks=seq(0, 500,length.out=101))

biocLite()

biocLite("monocle")
