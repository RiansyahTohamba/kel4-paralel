//Note: 
//Cara running program ./nama_file mode besar_matrix besar_grid besar_block
//Ukuran matrix: besar_matrix x besar matrix
//Grid: besar_grid x besar_grid (block per grid) 					| Max: Mengacu pada NVIDIA Compute Capability dari setiap seri GPU 
//Block: besar_block x besar_block (thread per block)			| Max: Mengacu pada NVIDIA Compute Capability dari setiap seri GPU
// mode 2 ketas belum selesai dikerjakan, masih belum sempurna (ukuran matrix harus setara block)
// Mode:
// 0: Matrix multiplication pada 1 GPU tanpa melihat hasil sekuensial
// 1: Matrix multiplication pada 1 GPU dengan hasil sekuensial
// 2: Matrix multiplication pada multiple GPU tanpa melihat hasil sekuensial
// 3: Matrix multiplication pada multiple GPU dengan hasil sekuensial

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#define sharedsize 32

//Operasi perkalian matrix pada gpu
__global__ void matrixmul_kernel(float *gpu_matrixA, float *gpu_matrixB, float *gpu_result, int matrix_size, int grid, int block){
	int l, m;
	float R = 0;
	//int n, o, displacement;
	//if(matrix_size > (grid * block)) displacement = matrix_size/(grid * block);
	//else displacement = 1;

	__shared__ float SM_A[sharedsize][sharedsize];
	__shared__ float SM_B[sharedsize][sharedsize];
	
	int row_index = blockIdx.y * sharedsize + threadIdx.y;
	int col_index = blockIdx.x * sharedsize + threadIdx.x;
	int mrow_index;
	int mcol_index;
	int max_iter = sharedsize < matrix_size ? sharedsize : matrix_size;

//	if(row_index < matrix_size && col_index < matrix_size){
		//for(n = 0; n < displacement; n++){
			//for(o = 0; o < displacement; o++){
			for(m = 0; m < (matrix_size + sharedsize - 1)/sharedsize; m++){
				mrow_index = row_index * matrix_size + m * sharedsize + threadIdx.x; 
			//	if(threadIdx.y + n < sharedsize && threadIdx.y + o < sharedsize){
				if(mrow_index < matrix_size * matrix_size)
					SM_A[threadIdx.y][threadIdx.x] = gpu_matrixA[mrow_index];
				else	SM_A[threadIdx.y][threadIdx.x] = 0.0; 
			//	}
				mcol_index = (m * sharedsize + threadIdx.y) * matrix_size + col_index;
				//if(threadIdx.y + n < sharedsize && threadIdx.y + o < sharedsize){
				if(mcol_index < matrix_size * matrix_size)	
					SM_B[threadIdx.y][threadIdx.x] = gpu_matrixB[mcol_index];
				else SM_B[threadIdx.y][threadIdx.x] = 0.0;
				//}	
				__syncthreads();
				for(l = 0; l < max_iter; l++)
					R += SM_A[threadIdx.y][l] * SM_B[l][threadIdx.x];
				__syncthreads();
			}		
	if(row_index < matrix_size && col_index < matrix_size) gpu_result[row_index * matrix_size + col_index] = R;
//	}
//	}
//	}
}

int main(int argc, char** argv){
	srand(time(NULL));
	double runtime;
	struct timespec begin, end;
	
	// Inisialisasi parameter dari user input
	int mode = atoi(argv[1]);
	int matrix_size = atoi(argv[2]);
	int igrid = atoi(argv[3]);
	int iblock = atoi(argv[4]);
		
	//Debug print variabel user input
	//printf("Mode: %d\n", mode);
	//printf("Size %d x %d\n", matrix_size, matrix_size);
	//printf("Grid: %d\n", igrid);
	//printf("Block:%d\n", iblock);

	// Inisailiasai pada Host
	//int matrixallsize = matrix_size * matrix_size;
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

  //Operasi dengan 1 GPU
	//if(mode < 2){	
		clock_gettime(CLOCK_REALTIME, &begin);
		//Inisialisasi pada GPU
		float *gpu_matrixA, *gpu_matrixB, *gpu_result;
		cudaMalloc((void **) &gpu_matrixA, matrixBytes);
		cudaMalloc((void **) &gpu_matrixB, matrixBytes);
		cudaMalloc((void **) &gpu_result, matrixBytes);
		cudaMemcpy(gpu_matrixA, matrixA, matrixBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_matrixB, matrixB, matrixBytes, cudaMemcpyHostToDevice);
		
		//int omg;
		//if(sharedsize > matrix_size){
		//	omg = matrix_size;
		//else omg = sharedsize;
				
		//Mulai operasi pada device
		igrid = (matrix_size - 1)/sharedsize + 1;
		iblock = sharedsize;
		
		//iblock = matrix_size/ igrid;
		//if(iblock < 1) iblock = 1;
		printf("Grid: %d - %d\n", igrid, iblock);
		dim3 grid(igrid, igrid);
		dim3 block(iblock, iblock);
		matrixmul_kernel<<<grid, block>>>(gpu_matrixA, gpu_matrixB, gpu_result, matrix_size, igrid, iblock);
		
		//Return hasil perkalian
		cudaMemcpy(result, gpu_result, matrixBytes, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_REALTIME, &end);
		runtime = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
		printf("Running Time: %f\n\n", runtime);
	/*}else{
		//Operasi pada multiple GPU
		//Check Device
		clock_gettime(CLOCK_REALTIME, &begin);
		int device_count;
		cudaGetDeviceCount(&device_count);
		printf("Device: %d\n", device_count);
		
		
		
		clock_gettime(CLOCK_REALTIME, &end);
		runtime = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
		printf("Running Time: %f\n\n", runtime);
	}*/

	//Operasi sekuensial
	if(mode == 1 || mode == 3){
		int right_answer = 0;
		float *seqresult = (float *)malloc(matrixBytes);
		for (i = 0; i < matrix_size; i++){
			for (j = 0; j < matrix_size; j++){ 
				seqresult[i * matrix_size + j] = 0;
				for (k = 0; k < matrix_size; k++)
					seqresult[i * matrix_size + j] += matrixA[i * matrix_size + k] * matrixB[k * matrix_size + j];
				if(seqresult[i * matrix_size + j] == result[i * matrix_size + j]) right_answer += 1;
				//printf("%d - %d S: %f, CUDA: %f\n", i * matrix_size, j, seqresult[i * matrix_size + j], result[i * matrix_size + j]);
			}
		}
		if(right_answer == (matrix_size * matrix_size)) printf("The answer is matched.\n");
		free(seqresult);
	}

	//Membebaskan Device
	cudaFree(gpu_matrixB);
	cudaFree(gpu_matrixB);
	cudaFree(gpu_result);
	
	//Membebaskan Host
	free(matrixA);
	free(matrixB);
	free(result);
}
