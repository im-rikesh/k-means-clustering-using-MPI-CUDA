# k-means-clustering-using-MPI-CUDA
Implementation of k-means Clustering using MPI and CUDA

# Input file
k-means clustering uses input from an input file (input10K.txt here). It has vertices in the form:
x1 y1
x2 y2
x3 y3
...

# Imlementation:
First k vertices from the input file are assigned as the centroid of the k clusters initially.
The centroids, number of centroids are broadcasted across all the nodes.
Total vertices are splitted into np processes where np is the number of process that we want to run. This is defined when executing the program later.
