#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char** argv){
	int rank, world_size;
	int matrix_size = atoi(argv[1]);
    double start, finish, total_time = 0.0, run_time;
    struct timespec begin, end;

	MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;

    int i, j, k, l, up, down, reorder;
	int dims = 0, periodic = 0;
	srand(time(NULL));
	int length  = matrix_size/world_size;
	int *matrixA = malloc((matrix_size * matrix_size) * sizeof(int));
	//int *matrixB;
    int *transposed_matrixB = malloc((matrix_size * matrix_size) * sizeof(int)); 
	int *result = malloc((matrix_size * matrix_size) * sizeof(int));
    int *seq_result;
	int *temp_matrixA = malloc((length * matrix_size) * sizeof(int));
	int *temp_trans_matrixB = malloc((length * matrix_size) * sizeof(int));
	int *temp_result = malloc((length * matrix_size) * sizeof(int));
    int *recvbuf = malloc((length * matrix_size) * sizeof(int));
    int *buf_temp;

	if(rank == 0){
        //matrixB = malloc((matrix_size * matrix_size) * sizeof(int));
        seq_result = malloc((matrix_size * matrix_size) * sizeof(int));
		for(i = 0; i < matrix_size * matrix_size; i++){
			matrixA[i] = rand() % 30 + 1;
			//matrixB[i] = rand() % 30 + 1;
            //transposed_matrixB[matrix_size * (i % matrix_size) + (i / matrix_size)] = matrixB[i];
            transposed_matrixB[i] = rand() % 30 + 1;
			//result[i] = 0;
            seq_result[i] = 0;
		}
        //Debugging
        /*printf("\nMatrix A:\n");
        for(i = 0; i < matrix_size; i++){
            for(j = 0; j < matrix_size; j++){
                printf("%d, ", matrixA[(i * matrix_size) + j]);
            }
            printf("\n");
        }
        printf("\nTransposed Matrix B:\n");
        for(i = 0; i < matrix_size; i++){
            for(j = 0; j < matrix_size; j++){
                printf("%d, ", transposed_matrixB[(i * matrix_size) + j]);
            }
            printf("\n");
        }
        printf("\n");*/
        clock_gettime(CLOCK_REALTIME, &begin);
	}

    if(rank == 0) start = MPI_Wtime();
    MPI_Scatter(matrixA, length * matrix_size, MPI_INT, temp_matrixA, length * matrix_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(transposed_matrixB, length * matrix_size, MPI_INT, temp_trans_matrixB, length * matrix_size, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0){ finish = MPI_Wtime(); total_time += finish - start;}
    up = (rank - 1) > -1 ? (rank - 1) : (world_size - 1);
    down = (rank + 1) < world_size ? (rank + 1) : 0;

    for(i = 0; i  < matrix_size * length; i++)
        temp_result[i] = 0;
    int shift;
    int post = rank * length;
    for(shift = 0; shift < world_size; shift++) {
    	// Matrix multiplication
    	for(i = 0;i < length; i++){
    	   for(j = 0; j < length; j++){
    	       for(k = 0; k < matrix_size; k++)
    	           temp_result[i * matrix_size + j + post] += temp_matrixA[i * matrix_size + k] * temp_trans_matrixB[j * matrix_size + k];
            }
        }
        
        if(rank == 0) start = MPI_Wtime();
        MPI_Sendrecv(temp_trans_matrixB, length * matrix_size, MPI_INT, up, 1, recvbuf, length * matrix_size, MPI_INT, down, 1, MPI_COMM_WORLD, &status);
        if(rank == 0){ finish = MPI_Wtime(); total_time += finish - start;}
        buf_temp = recvbuf; recvbuf = temp_trans_matrixB; temp_trans_matrixB = buf_temp;
        //temp_trans_matrixB = recvbuf;
	post  += length;
        if(post >= matrix_size) post = 0;
    }

    if(rank == 0) start = MPI_Wtime();
    MPI_Gather(temp_result, length * matrix_size, MPI_INT, result, length * matrix_size, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0){ finish = MPI_Wtime(); total_time += finish - start;}

    if(rank == 0){
        int right_answer = 0;
        clock_gettime(CLOCK_REALTIME, &end);
        for(i = 0;i < matrix_size; i++)
           for(j = 0; j < matrix_size; j++){
               for(k = 0; k < matrix_size; k++)
                    //seq_result[i * matrix_size + k] += matrixA[i * matrix_size + k] * matrixB[k * matrix_size + j];
                    seq_result[i * matrix_size + j] += matrixA[i * matrix_size + k] * transposed_matrixB[j * matrix_size + k];
            }

        for(i = 0; i < matrix_size * matrix_size; i++){
            if(seq_result[i] == result[i]) right_answer += 1;
            //printf("%d - %d\n", seq_result[i], result[i]);
        }

        if(right_answer == (matrix_size * matrix_size)) printf("The answer is matched.\n");
        run_time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        printf("Communication Time: %f\n", total_time);
        printf("Running Time: %f\n\n", run_time);
        //free(matrixB);
        free(seq_result);
    }

    free(transposed_matrixB);
    free(result);
    free(matrixA);
    free(temp_matrixA);
    free(temp_trans_matrixB);
    free(temp_result);
    free(recvbuf);
    MPI_Finalize();
	return 0;
}
