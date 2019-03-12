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
    char package[3], finalpackage[9];
    if(rank == 0){
        printf("Setting up processor %s rank 0 as root. with package = a, b, c.\n", processor_name);
        package[0] = 'a'; package[1] = 'b'; package[2] = 'c';}
    if(rank == 1){
        printf("Setting up processor %s rank 1 as root. with package = d, e, f.\n", processor_name);
        package[0] = 'd'; package[1] = 'e'; package[2] = 'f';}
    if(rank == 2){
        printf("Setting up processor %s rank 2 as root. with package = g, h, i.\n", processor_name);
        package[0] = 'g'; package[1] = 'h'; package[2] = 'i';}
    start = MPI_Wtime();
    MPI_Allgather(&package, 3, MPI_CHAR, &finalpackage, 3, MPI_CHAR, MPI_COMM_WORLD);

    finish = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n\nProcessor: %s rank: %d. Final package receive: %s\n", processor_name, rank, finalpackage);
    printf("Processor: %s, Time Elapsed: %f\n", processor_name, finish-start);
    MPI_Finalize();
 }



