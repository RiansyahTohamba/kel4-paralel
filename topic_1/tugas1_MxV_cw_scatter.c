#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>


int main(int argc, char** argv){
    srand(time(NULL));

    int numtasks, rank;

    int matrix_row = atoi(argv[1]);
    int matrix_col = atoi(argv[2]);;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Request request;
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    //int low_bound[numtasks];
    //int sendsize[numtasks];
    int fixed_size;
    double start, finish, total_time, run_time;
    int right_answer;
    struct timespec begin, end;

    fixed_size = matrix_col/numtasks;
    int *recvbuff = malloc(((matrix_col/numtasks) * matrix_row) * sizeof(int));
    int *transpose = malloc((matrix_col * matrix_row) * sizeof(int));
    int *trans_result = malloc((matrix_col * matrix_row) * sizeof(int));
    int *vector_result = malloc((matrix_row * (matrix_col/numtasks)) * sizeof(int));
    int *vector = malloc(matrix_col * sizeof(int));
    int *vecbuff = malloc(fixed_size * sizeof(int));
    int i, j;
    int *matrix;
    //int *result;


    if(rank == 0){
        matrix = malloc((matrix_row * matrix_col) * sizeof(int));
        //result = malloc((matrix_row * matrix_col) * sizeof(int));
        printf("Matrix: %d x %d\n\n", matrix_row, matrix_col);
        clock_gettime(CLOCK_REALTIME, &begin);
        //Generate random val
        for(i = 0; i < matrix_row * matrix_col; i++){
            matrix[i] = rand() % 100 + 1;
            transpose[matrix_col * (i%matrix_row) + (i/matrix_row)] = matrix[i];
        }
        for (i = 0; i < matrix_col; ++i) vector[i] = rand() % 100 + 1;
        //Debugging
        /*for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", transpose[(i * matrix_col) + j]);
            }
            printf("\n");
        }
        
        printf("\nVector: ");
        for (i = 0; i < matrix_col; ++i) printf("%d, ", vector[i]);
        printf("\n\n");*/
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) start = MPI_Wtime();
    MPI_Scatter(vector, fixed_size, MPI_INT, vecbuff, fixed_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(transpose, fixed_size * matrix_row, MPI_INT, recvbuff, (matrix_row * (matrix_col/numtasks)), MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0){ finish = MPI_Wtime(); total_time = finish - start; printf("Scatter Time: %f\n", finish - start);}
    
    //matrix multiplication
    for(j = 0; j < fixed_size; j++)
        for(i = 0; i < matrix_row; i++)
            vector_result[(j * matrix_row) + i] = recvbuff[(j * matrix_row) + i] * vecbuff[j];

    if(rank == 0) start = MPI_Wtime();
    MPI_Gather(vector_result, fixed_size * matrix_col, MPI_INT, trans_result, fixed_size * matrix_col, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0){ finish = MPI_Wtime(); total_time += finish - start; printf("Gather Time: %f\n", finish - start);}

    if(rank == 0){
        clock_gettime(CLOCK_REALTIME, &end);
        right_answer = 0;
        // sequential matrix vector multiplication
        for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                matrix[(i * matrix_col) + j] = matrix[(i * matrix_col) + j] * vector[j];
                if(matrix[(i * matrix_col) + j] == trans_result[(matrix_col * (((i * matrix_col) + j) % matrix_row)) + (((i * matrix_col) + j) / matrix_row)]) right_answer += 1;
            }
        }
        run_time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        if(right_answer == (matrix_row * matrix_col)) printf("The answer is matched.\n");
        printf("Communication Time: %f\n", total_time);
        printf("Running Time: %f\n\n", run_time);
        //Debugging
        /*for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", trans_result[(i * matrix_col) + j]);
            }
            printf("\n");
        }*/
        free(matrix);
        //free(result);
    }

    MPI_Finalize();
    free(trans_result);
    free(transpose);
    free(vecbuff);
    free(vector);
    free(recvbuff);
    free(vector_result);
}
