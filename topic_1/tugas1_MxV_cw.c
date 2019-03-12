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

    int low_bound;
    int sendsize;
    double start, finish;

    int matrix[matrix_row][matrix_col], transpose[matrix_col][matrix_row], vector[matrix_col], result[matrix_row][matrix_col], vector_result[100], recvbuff[100], vecbuff[matrix_col];
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
        
        printf("\n%d, %d, %d, %d, %d\n\n", vector[0], vector[1], vector[2], vector[3], vector[4]);
        */        
        start = MPI_Wtime();        
        //transpose matrix
        for (i = 0; i < matrix_row; i++)
            for(j = 0 ; j < matrix_col; j++)
                transpose[j][i] = matrix[i][j];

        sendsize = matrix_col/(numtasks - 1) > 0? matrix_col/(numtasks - 1) : 1;
        low_bound = 0;
        
        for (i = 1; i < numtasks; i++){ 
            if(i > matrix_col) break;
            else{
                if (((i + 1) == numtasks) && ((matrix_col% (numtasks - 1)) != 0)) sendsize = matrix_col - low_bound;
                MPI_Isend(&sendsize, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &request);
                MPI_Isend(&transpose[low_bound][0], sendsize * matrix_row, MPI_INT, i, 3, MPI_COMM_WORLD, &request);
                MPI_Isend(&vector[low_bound], sendsize, MPI_INT, i, 4, MPI_COMM_WORLD, &request);
                low_bound += sendsize;
            }
        }
    }
    MPI_Bcast(vector, matrix_col, MPI_INT, 0, MPI_COMM_WORLD);


    if(rank > 0 && rank < matrix_row + 1){
        MPI_Recv(&sendsize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&recvbuff, sendsize * matrix_row, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&vecbuff, sendsize, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        //matrix multiplication
        for(j = 0; j < sendsize; j++)
            for(i = 0; i < matrix_row; i++)
                vector_result[(j * matrix_row) + i] = recvbuff[(j * matrix_row) + i] * vecbuff[j];

        MPI_Isend(&sendsize, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &request);
        MPI_Isend(&vector_result, sendsize * matrix_row, MPI_INT, 0, 6, MPI_COMM_WORLD, &request);
    }

    if(rank == 0){
        low_bound = 0;
        for (i = 1; i < numtasks; i++)
        { 
            if(i > matrix_row) break;
            MPI_Recv(&sendsize, 1, MPI_INT, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&transpose[low_bound][0], sendsize * matrix_row, MPI_INT, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            low_bound += sendsize;
        }

        //transpose matrix
        for (i = 0; i < matrix_row; i++)
            for(j = 0 ; j < matrix_col; j++)
                result[i][j] = transpose[j][i];

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
