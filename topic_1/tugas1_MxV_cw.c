#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
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
    MPI_Request requests[numtasks - 1];
    MPI_Status status[numtasks - 1];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    double start, finish, total_time, run_time;
    int right_answer;
    struct timespec begin, end;
    int *matrix;
    int fixed_size = matrix_row/numtasks;
    int *recvbuff = malloc(((matrix_col/numtasks) * matrix_row) * sizeof(int));
    int *transpose;
    int *trans_result;
    int *vector_result = malloc((matrix_row * (matrix_col/numtasks)) * sizeof(int));
    int *vector;
    int *vecbuff = malloc(fixed_size * sizeof(int));
    int i, j;

    if(rank == 0){
        matrix = malloc((matrix_row * matrix_col) * sizeof(int));
        trans_result = malloc((matrix_col * matrix_row) * sizeof(int));
        transpose = malloc((matrix_col * matrix_row) * sizeof(int));
        vector = malloc(matrix_col * sizeof(int));
        //printf("Buffer Size: %d, Matrix: %d x %d\n\n", (matrix_col * fixed_size + 1), matrix_row, matrix_col);
        clock_gettime(CLOCK_REALTIME, &begin);
        //Generate random val
        for(i = 0; i < matrix_row * matrix_col; i++){
            matrix[i] = rand() % 10 + 1;
            transpose[matrix_col * (i%matrix_row) + (i/matrix_row)] = matrix[i];
        }
        for (i = 0; i < matrix_col; ++i) vector[i] = rand() % 10 + 1;
        //Debugging
        /*for(i = 0; i < matrix_row; i++){
            for(j = 0; j < matrix_col; j++){
                printf("%d, ", matrix[(i * matrix_col) + j]);
            }
            printf("\n");
        }
        
        printf("\nVector: ");
        for (i = 0; i < matrix_col; ++i) printf("%d, ", vector[i]);
        printf("\n\n");*/
    }
    //if(rank > 0) printf("Size: %d\n", fixed_size * matrix_col);
    if(rank == 0){
        start = MPI_Wtime();
        for(j = 1; j < numtasks; j++){
            MPI_Isend(&transpose[(j * matrix_row * fixed_size)], fixed_size * matrix_row, MPI_INT, j, 2, MPI_COMM_WORLD, &request);
            MPI_Isend(&vector[(j * fixed_size)], fixed_size, MPI_INT, j, 3, MPI_COMM_WORLD, &request);
            //printf("To: %d - size: %d from index: %d\n", j, fixed_size * matrix_col, j * fixed_size * matrix_col);
        }
        finish = MPI_Wtime(); total_time = finish - start; 
        printf("Send Time: %f\n", finish - start);

        for(j = 0; j < fixed_size; j++)
            for(i = 0; i < matrix_row; i++){
                trans_result[(j * matrix_row) + i] = transpose[(j * matrix_row) + i] * vector[j];
            }

        start = MPI_Wtime();
        for(j = 1; j < numtasks; j++){
            MPI_Irecv(&trans_result[(j * matrix_row * fixed_size)], fixed_size * matrix_row, MPI_INT, j, 4, MPI_COMM_WORLD, &requests[j-1]);
            //printf("%d - %d\n", j * matrix_col + i, result[j * matrix_col + i]);
        }
        MPI_Waitall(numtasks-1, requests, status);
        finish = MPI_Wtime(); total_time += finish - start;
        printf("Node Send Time: %f\n", finish - start);
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
                printf("%d, ", result[(i * matrix_col) + j]);
            }
            printf("\n");
        }*/
        free(matrix);
        free(vector);
        free(transpose);
        free(trans_result);
    }else{
        MPI_Recv(&recvbuff[0], (fixed_size * matrix_row), MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&vecbuff[0], fixed_size, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //matrix vector multiplication
        for(j = 0; j < fixed_size; j++)
            for(i = 0; i < matrix_row; i++)
                vector_result[(j * matrix_row) + i] = recvbuff[(j * matrix_row) + i] * vecbuff[j];

        MPI_Send(&vector_result[0], fixed_size * matrix_col, MPI_INT, 0, 4, MPI_COMM_WORLD);
    }
    free(recvbuff);
    free(vector_result);
    free(vecbuff);
    MPI_Finalize();
}
