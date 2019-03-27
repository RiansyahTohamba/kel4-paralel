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
    int name_len;
    int sum_one_row = 0;
    MPI_Get_processor_name(processor_name, &name_len);

    int fixedsize;
    double start, finish;
    struct timespec begin, end;

    double tol = 0.0001;
    double total_xrow, total_time = 0.0, run_time = 0.0;

    int *matrix = malloc((matrix_size * matrix_size) * sizeof(int*));
    int *b = malloc(matrix_size * sizeof(int)); 
    int *recvbuff = malloc((matrix_size * (matrix_size/numtasks)) * sizeof(int));
    double *x = malloc(matrix_size * sizeof(double));
    double *x_past = malloc(matrix_size * sizeof(double));
    double **x_record;
    double *v_result = malloc(matrix_size * sizeof(double));
    int i, j, k;
    int *vecbuff = malloc((matrix_size/numtasks) * sizeof(double));
    int belowtol = 0;
    
    for(i = 0; i < matrix_size; i++) x_past[i] = 0.0;
    if(rank == 0){
        x_record = malloc(matrix_size * sizeof(double*));
        printf("Matrix size: %d x %d\n", matrix_size, matrix_size);
        clock_gettime(CLOCK_REALTIME, &begin);
        //generate random val
        for(i = 0; i < matrix_size; i++){
            sum_one_row = 0;
            for(j = 0; j < matrix_size; j++){
                matrix[(i * matrix_size) + j] = (rand() % 10) + 1;
                sum_one_row += matrix[(i * matrix_size) + j];
            }
            x_record[i] = malloc(1001 * sizeof(double));
            matrix[(i * matrix_size) + i] = (rand() % 10) + sum_one_row;
            b[i] = (rand() % 10) + 1;
            x[i] = 0.0;
            x_record[i][0] = 0.0;

        }
        // Debugging
        /*printf("\nMatrix: \n");
        for(i = 0; i < matrix_size; i++){
            for(j = 0; j < matrix_size; j++){
                printf("%d, ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("\nB:");
        for(i = 0; i < matrix_size; i++) printf(" %d, ", b[i]);
        printf("\n");*/
    }

    fixedsize = matrix_size/numtasks;
    if(rank == 0) start = MPI_Wtime();
    MPI_Bcast(x, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrix, fixedsize * matrix_size, MPI_INT, recvbuff, (matrix_size * (matrix_size/numtasks)), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, fixedsize, MPI_INT, vecbuff, fixedsize, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0){ finish = MPI_Wtime(); total_time = finish - start;}

    for(i = 0; i < 1000; i++){
        //Jacobi Method
        for(j = 0; j < fixedsize; j++){
            total_xrow = 0.0;
            for(k = 0; k < matrix_size; k++){
                if((j * matrix_size) + k != ((j * matrix_size) + (rank * fixedsize) + j))
                total_xrow += (double) recvbuff[(j * matrix_size) + k] * x[k];
            }
            v_result[j] = (double) (vecbuff[j] - total_xrow) / recvbuff[(j * matrix_size) + (rank * fixedsize) + j];
            //if(rank == 0) printf("time: %d - %f/ %d\n", vecbuff[j], total_xrow, recvbuff[(j * matrix_size) + (rank * fixedsize) + j]);
        }
        if(rank == 0) start = MPI_Wtime();
        MPI_Allgather(v_result, fixedsize, MPI_DOUBLE, x, fixedsize, MPI_DOUBLE, MPI_COMM_WORLD);
        if(rank == 0){
            finish = MPI_Wtime(); 
            total_time = finish - start;
        }
            //print x result
            /*for(j = 0; j < matrix_size; j++){
                x_record[j][i+1] = x[j];
                printf("%f, ", x_record[j][i+1]);
            }
            printf("\n");*/
        for(k = 0; k < matrix_size; k++)
            if(fabs(x_past[k] - x[k]) < tol) belowtol += 1;
        if(belowtol == matrix_size) break;
        else{
            belowtol = 0;
            for(k = 0; k < matrix_size; k++) x_past[k] = x[k];
        }
    }
    if(rank == 0){
        clock_gettime(CLOCK_REALTIME, &end);
        run_time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        //printf("Final Result:");
        //for(i = 0; i < matrix_size; i++) printf(" %f,", x[i]);
        printf("\nCommunication Time: %f\n", total_time);
        printf("Running Time: %f\n", run_time);
        for(i = 0; i < matrix_size; i++)
            free(x_record[i]);
    }
    MPI_Finalize();
    free(matrix);
    free(b); 
    free(recvbuff);
    free(x);
    free(x_past);
    free(v_result);
    free(vecbuff);
}

