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
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Request request;
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    int low_bound[numtasks];
    int sendsize[numtasks];
    int fixed_size;
    double start, finish;

    int matrix[matrix_row][matrix_col], vector[matrix_col], result[matrix_row][matrix_col], vector_result[100], recvbuf[100];
    int i, j;
    
    if(rank == 0){
        //generate random val
        for(i = 0; i < matrix_col; i++){
            for(j = 0; j < matrix_row; j++){
                matrix[j][i] = rand() % 10 + 1;
            }
            vector[i] = rand() % 10 + 1;
        }

        /*for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", matrix[i][j]);
            }
            printf("\n");
        }
        
        
        printf("\nVector %d, %d, %d, %d, %d\n\n", vector[0], vector[1], vector[2], vector[3], vector[4]);
        */
        start = MPI_Wtime();
    }
    
    fixed_size = matrix_row/numtasks > 0 ? matrix_row/numtasks : 1;
    low_bound[0] = 0;
        
    for (i = 0; i < numtasks; i++){ 
        if(i > matrix_row - 1){
            low_bound[i] = low_bound[i-1];
            sendsize[i] = 0;
        }else{
            if (((i + 1) == numtasks) && ((matrix_row % numtasks) != 0)) sendsize[i] = (matrix_row * matrix_col) - low_bound[i-1];
            else sendsize[i] = fixed_size * matrix_col;
            if(i < numtasks-1) low_bound[i + 1] = low_bound[i] + sendsize[i];
        }
    }
    MPI_Bcast(vector, matrix_col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&matrix, sendsize, low_bound, MPI_INT, &recvbuf, 100, MPI_INT, 0, MPI_COMM_WORLD);

    //matrix multiplication
    for(j = 0; j < (sendsize[rank]/matrix_col); j++)
        for(i = 0; i < matrix_col; i++)
            vector_result[j * matrix_col + i] = recvbuf[j * matrix_col + i] * vector[i];

    MPI_Gatherv(&vector_result, sendsize[rank], MPI_INT, &result, sendsize, low_bound, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0){
        finish = MPI_Wtime();
        printf("\nRunning Time = %f\n\n", finish - start);
        /*for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", result[i][j]);
            }
            printf("\n");
        }*/
    }
        
    MPI_Finalize();
}
