import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import math

def kmeans(df_clean):
    # K Means using sklearn
    kmeans = KMeans(n_clusters=6, n_init=10, max_iter=300)
    kmeans.fit(df_clean)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)

class Cluster:
    def __init__(self, center, members):
        self.center = center
        self.members = [members]

    def mergeMembers(self, oldCluster):
        self.members.extend(oldCluster.members)

    def updateCenter(self, df):
        # consider all rows with ID in members from df
        # set center as center of mass of these values
        result_df = df.loc[df['ID'].isin(self.members)]
        result_df = result_df.drop(columns=['ID']).values
        self.center = center_of_mass(result_df)

    def __repr__(self):
        return f"Center: {self.center} and Members: {self.members}\n"

def agglomeration(df_clean, df):
    clusters = []
    smallClusterMerged = []
    # columns_list = df_clean.columns.values.tolist()

    # Add each point as a cluster
    # Find centroid using center of mass
    for i, row in df_clean.iterrows():
        clusters.append(Cluster(center_of_mass(row), i+1))
    print(clusters)

    # Dendograms to view the clusters
    # plt.figure(figsize=(15, 10))
    # plt.title("Shoppers Dendogram")
    # dend = shc.dendrogram(shc.linkage(df_clean.values, method='weighted'))
    # plt.show()

    while len(clusters) != 1:
        # Find 2 closest clusters using Euclidean distance
        minDistance = math.inf
        minDistanceClusterIndices = ()
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if i != j:
                    dist = distance.euclidean(clusters[i].center, clusters[j].center)
                    if minDistance >= dist:
                        minDistance = dist
                        minDistanceClusterIndices = (i, j)

        # Merge these two clusters
        clusterIndToMergeIn, clusterIndToDel = minDistanceClusterIndices
        smallClusterMerged.append(min(len(clusters[clusterIndToMergeIn].members), len(clusters[clusterIndToDel].members)))
        print(round(minDistance, 3), clusterIndToMergeIn, "<<-", clusterIndToDel)

        # Copy the members
        clusters[clusterIndToMergeIn].mergeMembers(clusters[clusterIndToDel])
        # Delete the other cluster
        del clusters[clusterIndToDel]
        # Update the cluster center
        clusters[clusterIndToMergeIn].updateCenter(df)

        if len(clusters) == 6:
            print("6 cluster details:")
            for i, c in enumerate(clusters):
                print(i, "Center of cluster", c.center)
                print(i, "Member size", len(c.members))

    print(clusters)
    print("List of size of last 18 smaller clusters merged in")
    print(smallClusterMerged[::-1][:18])


def cross_correlation(df_clean):
    # Cross correlation coefficient
    C1 = np.corrcoef(df_clean, rowvar=False)

    # Evaluate correlation
    print(C1)
    for i, c in enumerate(C1):
        sums = 0
        for cc in c:
            sums += abs(cc)
        print(i, sums-1)

def main():
    df = pd.read_csv("HW_PCA_SHOPPING_CART_v8961.csv")
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=['ID'])
    print(df_clean.head())

    # # Part A
    cross_correlation(df_clean)

    # # Part B
    agglomeration(df_clean, df)

    # # Part C: K-Means
    kmeans(df_clean)


if __name__ == '__main__':
    main()

