library(readxl)
library(dplyr)
library(ggplot2)
library(factoextra)
library(cluster)
library(fpc)
library(NbClust)
library(clusterCrit)

wine_data <- read_excel("D:/Uni work/University/Year 2/Machine learning/Whitewine_v6.xlsx")

wine_scaled <- scale(wine_data[, -11])

fviz_nbclust(wine_scaled, kmeans, method = "wss")
fviz_nbclust(wine_scaled, kmeans, method = "silhouette")

set.seed(123)
gap_stat <- clusGap(wine_scaled, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
fviz_gap_stat(gap_stat)

nb <- NbClust(wine_scaled, distance = "euclidean", min.nc = 2, max.nc = 15, method = "complete", index = "all")
set.seed(123)
k <- 3
kmeans_result <- kmeans(wine_scaled, centers = k, nstart = 25)
fviz_cluster(kmeans_result, data = wine_scaled, geom = "point", stand = FALSE)
wine_pca <- prcomp(wine_scaled, center = TRUE, scale. = TRUE)
summary(wine_pca)
fviz_eig(wine_pca, addlabels = TRUE, ylim = c(0, 100))
wine_pca_data <- as.data.frame(wine_pca$x[, 1:5])

set.seed(123)
k_pca <- 3
kmeans_result_pca <- kmeans(wine_pca_data, centers = k_pca, nstart = 25)
fviz_cluster(kmeans_result_pca, data = wine_pca_data, geom = "point", stand = FALSE)
calinski_harabasz_index_pca <- cluster.stats(dist(wine_pca_data), kmeans_result_pca$cluster)$ch
print(calinski_harabasz_index_pca)