//Note: 
// Cara compile nvcc nama_file.cu -o nama_file_output -lcublas -gencode arch=sre_compute,code=seri_sm
// Cara running program ./nama_file mode besar_matrix
//Ukuran matrix: besar_matrix x besar matrix
// Mode:
// 0: Matrix multiplication menggunakan cuBlas tanpa melihat sekuensial
// 1: -||-                                     melihat hasil sekuensial
// 2: Matrix multiplication menggunakan cublasXt tanpa melihat sekuensial
// 3: -||-                                     melihat hasil sekuensial
// Hasil sekuensialnya bisa salah

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cublasXt.h>


//Operasi perkalian matrix pada multi-gpu
void mm_cublasxt(float *gpu_matrixA, float *gpu_matrixB, float *gpu_result, int matrix_size){
	int lda = matrix_size,ldb = matrix_size, ldc = matrix_size, m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasXtHandle_t handle;
	cublasXtCreate(&handle);
	
	int device_count;
	cudaGetDeviceCount(&device_count);
	int device_id[device_count];
	for(m = 0; m < device_count; m++) device_id[m] = m;
	cublasXtDeviceSelect(handle, device_count, device_id);
	
	// matrix multiplication pada multi gpu
	cublasXtSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, matrix_size, matrix_size, matrix_size, alpha, gpu_matrixA, lda, gpu_matrixB, ldb, beta, gpu_result, ldc);

	cudaDeviceSynchronize();
	cublasXtDestroy(handle);
}

//Operasi perkalian pada 1 gpu
void mm_cublas(float *gpu_matrixA, float *gpu_matrixB, float *gpu_result, int matrix_size){
	int lda = matrix_size,ldb = matrix_size, ldc = matrix_size;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);
	
	// matrix multiplication pada satu gpu
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, matrix_size, matrix_size, matrix_size, alpha, gpu_matrixA, lda, gpu_matrixB, ldb, beta, gpu_result, ldc);

	cublasDestroy(handle);
}

int main(int argc, char** argv){
	srand(time(NULL));
	double runtime;
	struct timespec begin, end;
	
	// Inisialisasi parameter dari user input
	int mode = atoi(argv[1]);
	int matrix_size = atoi(argv[2]);
		
	//Debug print variabel user input
	//printf("Mode: %d\n", mode);
	//printf("Size %d x %d\n", matrix_size, matrix_size);

	// Inisailiasai pada Host
	int matrixBytes = (matrix_size * matrix_size) * sizeof(float);
	float *matrixA = (float *)malloc(matrixBytes) ;
	float *matrixB = (float *)malloc(matrixBytes);
	float *result = (float *)malloc(matrixBytes);
	int i, j, k;

	//Inisialisasi martrix
	for(i = 0; i < matrix_size * matrix_size; i++){
		matrixA[i] = rand() % 99 + 1;
		matrixB[i] = rand() % 99 + 1;
	}

	clock_gettime(CLOCK_REALTIME, &begin);
	
	//Inisialisasi pada GPU
	float *gpu_matrixA, *gpu_matrixB, *gpu_result;
	cudaMalloc((void **) &gpu_matrixA, matrixBytes);
	cudaMalloc((void **) &gpu_matrixB, matrixBytes);
	cudaMalloc((void **) &gpu_result, matrixBytes);
	cudaMemcpy(gpu_matrixA, matrixA, matrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_matrixB, matrixB, matrixBytes, cudaMemcpyHostToDevice);

	//Operasi cublas pada gpu
	if(mode < 2)
		mm_cublas(gpu_matrixA, gpu_matrixB, gpu_result, matrix_size);
	else
		mm_cublasxt(gpu_matrixA, gpu_matrixB, gpu_result, matrix_size);

	//Return hasil perkalian
	cudaMemcpy(result, gpu_result, matrixBytes, cudaMemcpyDeviceToHost);
	clock_gettime(CLOCK_REALTIME, &end);
	runtime = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	printf("Running Time: %f\n\n", runtime);
		
	if(mode == 1 || mode == 3){
		int right_answer = 0;
		float *seqresult = (float *)malloc(matrixBytes);
		for (i = 0; i < matrix_size; i++){
			for (j = 0; j < matrix_size; j++){ 
				seqresult[i * matrix_size + j] = 0;
				for (k = 0; k < matrix_size; k++)
					seqresult[i * matrix_size + j] += matrixA[i * matrix_size + k] * matrixB[k * matrix_size + j];
					//seqresult[i + matrix_size * j] += matrixA[i + matrix_size * k] * matrixB[k + matrix_size * j];
				if(seqresult[i * matrix_size + j] == result[i + matrix_size * j])  right_answer += 1;
			  //printf("%d - %d S: %f, CUDA: %f\n", i * matrix_size, j, seqresult[i * matrix_size + j], result[i + matrix_size * j]);
			}
		}
		if(right_answer == (matrix_size * matrix_size)) printf("The answer is matched.\n");
		free(seqresult);
	}

	//Membebaskan Device
	cudaFree(gpu_matrixB);
	cudaFree(gpu_matrixB);
	cudaFree(gpu_result);
	cudaDeviceReset();

	//Membebaskan Host
	free(matrixA);
	free(matrixB);
	free(result);
}
