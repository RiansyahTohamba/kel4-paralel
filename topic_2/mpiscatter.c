#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv){
    int numtasks, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    double start, finish;
    int package[8], recvbuffer, randomval;
    if(rank == 0){
    	printf("Setting up processor %s rank 0 as root. with package = 51, 52, 53, 54, 55, 56, 57, 58. Scatter to all.\n\n", processor_name);
    	package[0] = 51; package[1] = 52; package[2] = 53; package[3] = 54; package[4] = 55; package[5] = 56; package[6] = 57; package[7] = 58;
    }
    start = MPI_Wtime();
    MPI_Scatter(&package, 1, MPI_INT, &recvbuffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Processor: %s rank: %d. Package receive: %d\n change package to: %d\n", processor_name, rank, recvbuffer, rank);
    recvbuffer = rank;
    MPI_Gather(&recvbuffer, 1, MPI_INT, &package, 1, MPI_INT, 0, MPI_COMM_WORLD);
    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("\n\nProcessor: %s rank: %d. Final package receive: %d, %d, %d, %d, %d, %d, %d, %d\n\n", processor_name, rank, package[0], package[1], package[2], package[3], package[4], package[5], package[6], package[7]);
    printf("Processor: %s, Time Elapsed: %f\n", processor_name, finish-start);
    MPI_Finalize();
 }



