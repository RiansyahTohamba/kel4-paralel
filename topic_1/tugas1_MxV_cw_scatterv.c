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
    int low_boundv[numtasks];
    int sendsize[numtasks];
    int sendsizev[numtasks];
    int fixed_size;
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
        
        printf("\nVector %d, %d, %d, %d, %d\n\n", vector[0], vector[1], vector[2], vector[3], vector[4]);
        */
        start = MPI_Wtime();

        //transpose matrix
        for (i = 0; i < matrix_row; i++)
            for(j = 0 ; j < matrix_col; j++)
                transpose[j][i] = matrix[i][j];
    }

    fixed_size = matrix_col/numtasks > 0 ? matrix_col/numtasks : 1;
    low_bound[0] = 0;
    low_boundv[0] = 0;
        
    for (i = 0; i < numtasks; i++){ 
        if(i > matrix_col - 1){
            low_bound[i] = low_bound[i-1];
            low_boundv[i] = low_boundv[i-1];
            sendsize[i] = 0;
            sendsizev[i] = 0;
        }else{
            if (((i + 1) == numtasks) && ((matrix_col % numtasks) != 0)){
                sendsize[i] = (matrix_row * matrix_col) - low_bound[i-1];
                sendsizev[i] = matrix_col - low_boundv[i-1];
            }else{
                sendsize[i] = fixed_size * matrix_row;  
                sendsizev[i] = fixed_size;
            } 
            if(i < numtasks-1){
                low_bound[i + 1] = low_bound[i] + sendsize[i];
                low_boundv[i + 1] = low_boundv[i] + fixed_size;
            } 
        }
    }
    MPI_Bcast(vector, matrix_col, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&transpose, sendsize, low_bound, MPI_INT, &recvbuff, 100, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&vector, sendsizev, low_boundv, MPI_INT, &vecbuff, matrix_col, MPI_INT, 0, MPI_COMM_WORLD);

    //if(sendsize[rank] > 0) printf("%s - %d  Results: %d %d %d %d %d\n", processor_name, rank,recvbuff[0], recvbuff[1], recvbuff[2], recvbuff[3], recvbuff[4]);
    //matrix multiplication
    for(j = 0; j < sendsizev[rank]; j++)
            for(i = 0; i < matrix_row; i++)
                vector_result[(j * matrix_row) + i] = recvbuff[(j * matrix_row) + i] * vecbuff[j];

    MPI_Gatherv(&vector_result, sendsize[rank], MPI_INT, &transpose, sendsize, low_bound, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank == 0){
        //transpose matrix
        for (i = 0; i < matrix_row; i++)
            for(j = 0 ; j < matrix_col; j++)
                result[i][j] = transpose[j][i];

        finish = MPI_Wtime();
        /*printf("\nTime = %f\n\n", finish - start);
        for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", result[i][j]);
            }
            printf("\n");
        }*/
    }
        
    MPI_Finalize();
}

