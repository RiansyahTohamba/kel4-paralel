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

    int low_bound = 0;
    int sendsize = 0;
    double start, finish;

    int matrix[matrix_row][matrix_col], vector[matrix_col], result[matrix_row][matrix_col], vector_result[matrix_col*matrix_row], recvbuff[matrix_col*matrix_row], vecbuff[matrix_col];
    int i, j, k;
    
    if(rank == 0){
        //generate random val
        for(i = 0; i < matrix_col; i++){
            for(j = 0; j < matrix_row; j++){
                matrix[j][i] = rand() % 10 + 1;
            }
            vector[i] = rand() % 10 + 1;
        }

        for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", matrix[i][j]);
            }
            printf("\n");
        }
        start = MPI_Wtime();
        
        printf("\n%d, %d, %d, %d, %d\n\n", vector[0], vector[1], vector[2], vector[3], vector[4]);
        sendsize = matrix_col/(numtasks - 1);
        low_bound = 0;
        
        for (i = 1; i < numtasks; i++){ 
            if (((i + 1) == numtasks) && ((matrix_col % (numtasks - 1)) != 0)) sendsize = matrix_row - low_bound;

            for(j = 0; j < sendsize; j++)
                for(k = 0; k < matrix_row; k++)
                    recvbuff[j*matrix_row + k] = matrix[k][low_bound +j ];

            MPI_Isend(&sendsize, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &request);
            MPI_Isend(&recvbuff, sendsize * matrix_row, MPI_INT, i, 3, MPI_COMM_WORLD, &request);
            MPI_Isend(&vector[low_bound], sendsize, MPI_INT, i, 4, MPI_COMM_WORLD, &request);
            if(low_bound + sendsize > matrix_col) break;
            low_bound += sendsize;
        }
    }


    if(rank > 0){
        MPI_Recv(&sendsize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&recvbuff, sendsize * matrix_row, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&vecbuff, sendsize, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%s - %d  Results: %d %d %d %d %d\n", processor_name, rank,recvbuff[5], recvbuff[6], recvbuff[7], recvbuff[8], recvbuff[9]);
        //matrix multiplication
        for(j = 0; j < sendsize; j++)
            for(i = 0; i < matrix_row; i++)
                vector_result[j * matrix_row + i] = recvbuff[j * matrix_row + i] * vecbuff[j];

        MPI_Isend(&sendsize, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &request);
        MPI_Isend(&vector_result, sendsize * matrix_row, MPI_INT, 0, 6, MPI_COMM_WORLD, &request);
    }

    if(rank == 0){
        low_bound = 0;
        for (i = 1; i < numtasks; i++)
        { 
            MPI_Recv(&sendsize, 1, MPI_INT, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&recvbuff, sendsize * matrix_row, MPI_INT, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(j = 0; j < sendsize; j++)
                for(k = 0; k < matrix_row; k++)
                    matrix[k][low_bound + j] = recvbuff[j*matrix_row + k];

            if(low_bound + sendsize > matrix_row) break;
            low_bound += sendsize;
        }

        finish = MPI_Wtime();
        //printf("\nTime = %f\n\n", finish - start);
        for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", result[i][j]);
            }
            printf("\n");
        }
    }
        
    MPI_Finalize();
}