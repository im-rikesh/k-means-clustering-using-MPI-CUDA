#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include "ClusterAssign.h"
#include <chrono>

using namespace std;

// CUDA function declaration
extern void calculateDistancesCUDA(
    const double* hostPointsX, const double* hostPointsY, int numPoints,
    const double* hostCentroidsX, const double* hostCentroidsY, int numClusters,
    int* hostClusterAssignments);

// Function to read input data
void readInputData(double* x, double* y, int& numPoints) {
    ifstream inputFile("input10K.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening input.txt file!" << endl;
        exit(EXIT_FAILURE);
    }

    numPoints = 0;
    while (inputFile >> x[numPoints] >> y[numPoints]) {
        numPoints++;
    }
    inputFile.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numPoints = 0, numClusters = 0;
    double allX[1000], allY[1000];

    auto start = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        cout << "Enter the number of clusters: ";
        cin >> numClusters;
        readInputData(allX, allY, numPoints);
        cout << "Total points: " << numPoints << endl;
    }

    MPI_Bcast(&numClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int pointsPerProc = numPoints / numProcs;
    int extraPoints = numPoints % numProcs;
    int* recvCounts = new int[numProcs];
    int* displacements = new int[numProcs];
    for (int i = 0; i < numProcs; ++i) {
        recvCounts[i] = pointsPerProc + (i < extraPoints ? 1 : 0);
        displacements[i] = (i == 0) ? 0 : (displacements[i - 1] + recvCounts[i - 1]);
    }

    double* localX = new double[recvCounts[rank]];
    double* localY = new double[recvCounts[rank]];
    MPI_Scatterv(allX, recvCounts, displacements, MPI_DOUBLE, localX, recvCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(allY, recvCounts, displacements, MPI_DOUBLE, localY, recvCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* centroidsX = new double[numClusters];
    double* centroidsY = new double[numClusters];

    if (rank == 0) {
        for (int i = 0; i < numClusters; ++i) {
            centroidsX[i] = allX[i];
            centroidsY[i] = allY[i];
        }
    }
   /*
    if (rank == 0) {
    for (int i = 0; i < numClusters; ++i) {
        centroidsX[i] = allX[rand() % numPoints];
        centroidsY[i] = allY[rand() % numPoints];
    }
}
*/

    MPI_Bcast(centroidsX, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(centroidsY, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int iter = 0; iter < 100; ++iter) {
        int* clusterAssignments = new int[recvCounts[rank]];

        calculateDistancesCUDA(localX, localY, recvCounts[rank],
                               centroidsX, centroidsY, numClusters,
                               clusterAssignments);

        double* localSumsX = new double[numClusters]();
        double* localSumsY = new double[numClusters]();
        int* localCounts = new int[numClusters]();

        for (int i = 0; i < recvCounts[rank]; ++i) {
            int cluster = clusterAssignments[i];
            localSumsX[cluster] += localX[i];
            localSumsY[cluster] += localY[i];
            localCounts[cluster]++;
        }

        double* globalSumsX = new double[numClusters];
        double* globalSumsY = new double[numClusters];
        int* globalCounts = new int[numClusters];

        MPI_Reduce(localSumsX, globalSumsX, numClusters, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(localSumsY, globalSumsY, numClusters, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(localCounts, globalCounts, numClusters, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < numClusters; ++i) {
                if (globalCounts[i] > 0) {
                    centroidsX[i] = globalSumsX[i] / globalCounts[i];
                    centroidsY[i] = globalSumsY[i] / globalCounts[i];
                } else {
            // Reinitialize empty cluster centroid
            int randomIdx = rand() % numPoints;
            centroidsX[i] = allX[randomIdx];
            centroidsY[i] = allY[randomIdx];
            printf("Cluster %d is empty. Reinitializing to point (%f, %f)\n", i, centroidsX[i], centroidsY[i]);
        }
                
                if (globalCounts[i] == 0) {
    printf("Cluster %d has no points assigned\n", i);
}
            }
        }
        
       // printf("--------------------Itr %d ---------------------------- \n", iter);
        
        if (rank == 0) {
    for (int i = 0; i < numClusters; ++i) {
        printf("Global sum for cluster %d: (%f, %f), count: %d\n", i, globalSumsX[i], globalSumsY[i], globalCounts[i]);
    }
}

        

        for (int i = 0; i < recvCounts[rank]; ++i) {
    printf("Point (%f, %f) assigned to cluster %d\n", localX[i], localY[i], clusterAssignments[i]);
}



        MPI_Bcast(centroidsX, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(centroidsY, numClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        delete[] clusterAssignments;
        delete[] localSumsX;
        delete[] localSumsY;
        delete[] localCounts;
        delete[] globalSumsX;
        delete[] globalSumsY;
        delete[] globalCounts;
    }

    if (rank == 0) {
        cout << "Final centroids:" << endl;
        for (int i = 0; i < numClusters; ++i) {
            cout << "Cluster " << i << ": (" << centroidsX[i] << ", " << centroidsY[i] << ")" << endl;
        }
    }

    delete[] localX;
    delete[] localY;
    delete[] centroidsX;
    delete[] centroidsY;
    delete[] recvCounts;
    delete[] displacements;

    MPI_Finalize();
    
    if (rank == 0) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Running with MPI + CUDA took " << duration.count() << " milliseconds." << std::endl;
    }
    
    return 0;
}

