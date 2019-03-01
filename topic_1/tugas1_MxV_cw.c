#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#define matrix_row 4
#define matrix_col 5


int main(int argc, char** argv){
    srand(time(NULL));
    
    int numtasks, rank;
        MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

        int matrix[matrix_row][matrix_col], vector[matrix_row], result[matrix_row][matrix_col], vector_result[matrix_col], recvbuf[100], recvbuf_vector;
        int i, j;
    
    if(rank == 0){
        //generate random val
            for(i = 0; i < matrix_col; i++){
            for(j = 0; j < matrix_row; j++){

                vector[i] = rand() % 10 + 1;
                vector[i] = rand() % 10 + 1;
                matrix[j][i] = rand() % 10 + 1;
                //printf("%d, ", matrix[j][i]);
            }
            //printf("\n");
        }
        //printf("\n%d, %d, %d, %d\n\n", vector[0], vector[1], vector[2], vector[3]);
    }
    
    MPI_Scatter(vector, 1, MPI_INT, &recvbuf_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrix, matrix_col, MPI_INT, recvbuf, matrix_col, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Scatterv(&matrix, sendcount, displ, MPI_INT, &recvbuf, matrix_row, MPI_INT, 0, MPI_COMM_WORLD);

    //printf("rank= %d  Results: %d %d %d %d %d - %d\n",rank,recvbuf[0], recvbuf[1], recvbuf[2], recvbuf[3], recvbuf[4], recvbuf_vector);

    //matrix multiplication
    for(i = 0; i < matrix_col; i++)
        vector_result[i] = recvbuf[i] * recvbuf_vector; 
        
    MPI_Gather(vector_result, matrix_col, MPI_INT, result, matrix_col, MPI_INT, 0, MPI_COMM_WORLD); 
    
    /* for debugging
    if(rank == 0){
        for(i = 0; i < matrix_col; i++){
            for(j = 0; j < matrix_row; j++){
                printf("%d, ", result[j][i]);
            }
            printf("\n");
        }
    }*/
        
    MPI_Finalize();
}

