#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv){

    int numtasks, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    double start, finish;
    int name_len;
    int package;
    MPI_Get_processor_name(processor_name, &name_len);
    if(rank == 0){
    	printf("Setting up rank 0 as root. with package = 101\n\n");
    	package = 101;
    }
    start = MPI_Wtime();
    MPI_Bcast(&package, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Processor: %s rank: %d. Package receive: %d\n", processor_name, rank, package);
    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("\n");
    printf("Processor: %s, Time Elapsed: %f\n", processor_name, finish-start);
    MPI_Finalize();
 }
