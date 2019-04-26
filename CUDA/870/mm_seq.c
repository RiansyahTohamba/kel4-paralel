//Note: 
//======= Cara running program ======= 
//./nama_file mode besar_matrix besar_grid besar_block
//Ukuran matrix: besar_matrix x besar matrix 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv){
	srand(time(NULL));
	double run_time;
	struct timespec begin, end;
	int i, j, k;
	
	// Inisialisasi parameter dari user input
	int matrix_size = atoi(argv[1]);

	//Debug print variabel user input
	//printf("Size %d x %d\n", matrix_size, matrix_size);
	
	float *matrixA, *matrixB, *seqresult;
	int matrixBytes = (matrix_size * matrix_size) * sizeof(float);
	matrixA = (float *)malloc(matrixBytes) ;
	matrixB = (float *)malloc(matrixBytes);
	seqresult = (float *)malloc(matrixBytes);
		
	for(i = 0; i < matrix_size * matrix_size; i++){
		matrixA[i] = rand() % 99 + 1;
		matrixB[i] = rand() % 99 + 1;
	}

	clock_gettime(CLOCK_REALTIME, &begin);	
	
	//Operasi sekuensial
	for (i = 0; i < matrix_size; i++)
		for (j = 0; j < matrix_size; j++){ 
			seqresult[i * matrix_size + j] = 0;
			for (k = 0; k < matrix_size; k++)
				seqresult[i * matrix_size + j] += matrixA[i * matrix_size + k] * matrixB[k * matrix_size + j];	
		}

	clock_gettime(CLOCK_REALTIME, &end);	
	run_time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	printf("Running Time: %f\n\n", run_time);
	
	//Membebaskan Host
	
	free(matrixA);
	free(matrixB);
	free(seqresult);
	
	return 0;
}
