#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


int main(int argc, char** argv){
    srand(time(NULL));

    int numtasks, rank;
    int matrix_size = atoi(argv[1]);
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Request request;
    int name_len;
    int sum_one_row = 0;
    int operation_flag = 1;
    MPI_Get_processor_name(processor_name, &name_len);

    int fixedsize;
    //double start, finish;

    double tol = 0.001;
    double distance, total_xrow;

    int matrix[matrix_size][matrix_size], b[matrix_size], recvbuff[(matrix_size * (matrix_size/numtasks))];
    double x[matrix_size], x_record[matrix_size][1001], v_result[matrix_size];
    int i, j, k, vecbuff[(matrix_size/numtasks)];
    int belowtol = 0;
    
    if(rank == 0){
        //generate random val
        for(i = 0; i < matrix_size; i++){
            sum_one_row = 0;
            for(j = 0; j < matrix_size; j++){
                matrix[i][j] = (rand() % 10) + 1;
                sum_one_row += matrix[i][j];
            }
            matrix[i][i] = (rand() % 10) + sum_one_row;
            b[i] = (rand() % 10) + 1;
            x[i] = 0.0;
            x_record[i][0] = 0.0;

        }
        /*printf("\nMatrix: \n");
        for(i = 0; i < matrix_size; i++){
            for(j = 0; j < matrix_size; j++){
                printf("%d, ", matrix[i][j]);
            }
            printf("\n");
        }
        //start = MPI_Wtime();
        printf("\n\nB: %d, %d, %d\n\n", b[0], b[1], b[2]);
        //printf("\n\nB: %d, %d, %d, %d, %d\n\n", b[0], b[1], b[2], b[3], b[4]);*/
    }
    fixedsize = matrix_size/numtasks;
    MPI_Bcast(&x, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(&matrix, fixedsize * matrix_size, MPI_INT, &recvbuff, (matrix_size * (matrix_size/numtasks)), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(&b, fixedsize, MPI_INT, &vecbuff, matrix_size, MPI_INT, 0, MPI_COMM_WORLD);


    for(i = 0; i < 15; i++){
        //Jacobi Method
        for(j = 0; j < fixedsize; j++){
            total_xrow = 0.0;
            for(k = 0; k < matrix_size; k++){
                if((j * matrix_size) + k != ((j * matrix_size) + (rank * fixedsize) + j))
                total_xrow += (double) recvbuff[(j * matrix_size) + k] * x[k];
                //printf("%d Value: %d\n", (j * matrix_size) + k, recvbuff[(j * matrix_size) + k]);
            }
            v_result[j] = (double) (vecbuff[j] - total_xrow) / recvbuff[(j * matrix_size) + (rank * fixedsize) + j];
            //printf("%d - %f / %d = %f\n", vecbuff[j], total_xrow, recvbuff[(j * matrix_size) + (rank * fixedsize) + j], v_result[j]);
            //printf("lol %d: from %d & %d then %f\n", ((j * matrix_size) + (rank * fixedsize) + j), recvbuff[(j * matrix_size) + (rank * fixedsize) + j], vecbuff[i], v_result[i]);
        }
        MPI_Allgather(&v_result, fixedsize, MPI_DOUBLE, &x, fixedsize, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0){
            /*for(j = 0; j < matrix_size; j++){
                x_record[j][i+1] = x[j];
                printf("%f, ", x_record[j][i+1]);
            }*/
            printf("\n");
            for(k = 0; k < matrix_size; k++)
                if(fabs(x_record[k][i] - x[k]) < tol) belowtol += 1;
        }
        MPI_Bcast(&belowtol, matrix_size, MPI_INT, 0, MPI_COMM_WORLD);
        if(belowtol == matrix_size) break;
        else belowtol = 0;
    }   
    MPI_Finalize();
}