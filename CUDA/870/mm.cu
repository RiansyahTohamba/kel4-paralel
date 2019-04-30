//Note: 
//======= Cara compile =======
//nvcc nama_file.cu -o nama_file_output -gencode arch=compute_serinya,code=sm_serinya --default-stream per-thread
//======= Cara running program ======= 
//./nama_file mode besar_matrix besar_grid besar_block


//Ukuran matrix: besar_matrix x besar matrix 
// besar_grid max = 65535 																(Max grid.y adalah 65535)
// besar_block max = 32, 																	(32 x 32  = 1024)
//Grid: besar_grid x besar_grid (block per grid) 					| Max: Mengacu pada NVIDIA Compute Capability dari setiap seri GPU 
//Block: besar_block x besar_block (thread per block)			| Max: 1024, mengacu pada NVIDIA Compute Capability dari setiap seri GPU
// mode 2 ketas belum selesai dikerjakan
// Mode:
// 0: Matrix multiplication pada 1 GPU tanpa melihat hasil sekuensial
// 1: Matrix multiplication pada 1 GPU dengan hasil sekuensial
// 2: Matrix multiplication pada multiple GPU tanpa melihat hasil sekuensial
// 3: Matrix multiplication pada multiple GPU dengan hasil sekuensial

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
//#include <helper_cuda.h>
#include <cuda_runtime.h>
//#include <helper_functions.h>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if(code != cudaSuccess){
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}

//Operasi perkalian matrix pada gpu
__global__ void mm_gpu(float *gpu_matrixA, float *gpu_matrixB, float *gpu_result, int matrix_size, int grid, int block){
	int l, m, n, R, displacement;
	if(matrix_size > (grid * block)) displacement = matrix_size/(grid * block);
	else displacement = 1;
	int row_index = blockIdx.y * blockDim.y + threadIdx.y;
	int col_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(row_index < matrix_size && col_index < matrix_size){
		for(m = 0; m < displacement; m++){
			for(n = 0; n < displacement; n++){
				R = 0;
				for(l = 0; l < matrix_size; l++){
					float A = gpu_matrixA[(row_index * displacement + m) * matrix_size + l];
					float B = gpu_matrixB[l * matrix_size + (col_index * displacement + n)];
					R += A * B;
				}
				gpu_result[(row_index * displacement + m) * matrix_size + (col_index * displacement + n)] = R;
			}
		}
	}
}

__global__ void mm_multigpu(float *gpu_matrixA, float *gpu_matrixB, float *gpu_result, int device_count, int device_index, int matrix_size, int grid, int block){
	int l, m, n, R, row_disp, col_disp;
	int data_split = matrix_size/device_count;
	int row_index = blockIdx.y * blockDim.y + threadIdx.y;
	int col_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(data_split * device_count < matrix_size && device_count == device_index + 1)
		data_split += matrix_size - (data_split * device_count);

	if(data_split > (grid * block)){ 
		row_disp = data_split/(grid * block);
		//if(row_disp * (grid * block) < data_split && row_index == data_split) 
		//	row_disp += data_split - (row_disp * grid * block);
	}else row_disp = 1;
	
	if(matrix_size > (grid * block)){
		col_disp = matrix_size / (grid * block);
		//if(col_disp * grid * block < matrix_size && col_index == (grid * block)) 
		//	col_disp += matrix_size - col_disp * grid * block;
	}else col_disp = 1;
	

	if(col_index < matrix_size && row_index < data_split){
		for(m = 0; m < row_disp; m++){
			for(n = 0; n < col_disp; n++){
				R = 0;
				for(l = 0; l < matrix_size; l++){
					float A = gpu_matrixA[(row_index * row_disp + m) * matrix_size + l];
					float B = gpu_matrixB[l * matrix_size + (col_index * col_disp + n)];
					R += A * B;
				}
				gpu_result[(row_index * row_disp + m) * matrix_size + (col_index * col_disp + n)] = R;			
			}
		}
	}
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
	int i, j, k;
	float *matrixA, *matrixB, *result;
	int device_count;
	cudaGetDeviceCount(&device_count);
	//printf("Device: %d\n", device_count);
	float *gpu_matrixA[device_count], *gpu_matrixB[device_count], *gpu_result[device_count];
	
	//Inisialisasi pada GPU
	dim3 grid(igrid, igrid);
	dim3 block(iblock, iblock);

	//printf("Dim3 Block: {%d, %d, %d}\n", block.x, block.y, block.z);

  //Operasi dengan 1 GPU
	if(mode < 2){	
		//float *gpu_matrixA, *gpu_matrixB, *gpu_result;
		matrixA = (float *)malloc(matrixBytes) ;
		matrixB = (float *)malloc(matrixBytes);
		result = (float *)malloc(matrixBytes);
		
		//Inisialisasi martrix
		for(i = 0; i < matrix_size * matrix_size; i++){
			matrixA[i] = rand() % 99 + 1;
			matrixB[i] = rand() % 99 + 1;
		}

		clock_gettime(CLOCK_REALTIME, &begin);
		//Mulai operasi pada device
		checkCudaErrors(cudaMalloc((void **) &gpu_matrixA[0], matrixBytes));
		checkCudaErrors(cudaMalloc((void **) &gpu_matrixB[0], matrixBytes));
		checkCudaErrors(cudaMalloc((void **) &gpu_result[0], matrixBytes));
		checkCudaErrors(cudaMemcpy(gpu_matrixA[0], matrixA, matrixBytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gpu_matrixB[0], matrixB, matrixBytes, cudaMemcpyHostToDevice));
		mm_gpu<<<grid, block>>>(gpu_matrixA[0], gpu_matrixB[0], gpu_result[0], matrix_size, igrid, iblock);
	  
		cudaError error_kernel;
	  error_kernel = cudaGetLastError();
		if(error_kernel != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(error_kernel));
		//Return hasil perkalian
		checkCudaErrors(cudaMemcpy(result, gpu_result[0], matrixBytes, cudaMemcpyDeviceToHost));
		//cudaDeviceSynchronize();
		clock_gettime(CLOCK_REALTIME, &end);

		runtime = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
		printf("Running Time: %f\n\n", runtime);
	}else{
		//Operasi pada multiple GPU
		//Check Device	
		checkCudaErrors(cudaMallocHost((void**) &matrixA, matrixBytes));
		checkCudaErrors(cudaMallocHost((void**) &matrixB, matrixBytes));
		checkCudaErrors(cudaMallocHost((void**) &result, matrixBytes));
		
		//Inisialisasi martrix
		for(i = 0; i < matrix_size * matrix_size; i++){
			matrixA[i] = rand() % 99 + 1;
			matrixB[i] = rand() % 99 + 1;
		}
	
		clock_gettime(CLOCK_REALTIME, &begin);
		
		int start_p, chunk_size = (matrix_size/device_count);
		int chunkBytes;	
		int rem_size;
		if((chunk_size * device_count) != matrix_size) rem_size = matrix_size - (chunk_size * device_count);
		else rem_size = 0;
		printf("chunk size: %d\n", chunk_size);
		printf("remaining size: %d\n", rem_size);

		//Inisialisasi memori pada tiap gpu
		for(i = 0; i < device_count; i++){
			checkCudaErrors(cudaSetDevice(i));
			if(i == (device_count - 1))
				chunkBytes = ((chunk_size + rem_size) * matrix_size) * sizeof(float);
			else
				chunkBytes = (chunk_size * matrix_size) * sizeof(float);

			checkCudaErrors(cudaMalloc((void **) &gpu_matrixA[i], chunkBytes));
			checkCudaErrors(cudaMalloc((void **) &gpu_matrixB[i], matrixBytes));
			checkCudaErrors(cudaMalloc((void **) &gpu_result[i], chunkBytes));
		}
		
		for(i = 0; i < device_count; i++){
			start_p = i * chunk_size;
			
			if(i == (device_count - 1))
				chunkBytes = ((chunk_size + rem_size) * matrix_size) * sizeof(float);
			else
				chunkBytes = (chunk_size * matrix_size) * sizeof(float);
				
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpyAsync(gpu_matrixA[i], &matrixA[start_p], chunkBytes, cudaMemcpyHostToDevice));
		}

		for(i = 0; i < device_count; i++){
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpyAsync(gpu_matrixB[i], matrixB, matrixBytes, cudaMemcpyHostToDevice));	
		}
		
		for(i = 0; i < device_count; i++){
			checkCudaErrors(cudaSetDevice(i));
			mm_multigpu<<<grid, block>>>(gpu_matrixA[i], gpu_matrixB[i], gpu_result[i], device_count, i, matrix_size, igrid, iblock);
		}

		for(i = 0; i < device_count; i++){
			start_p = i * chunk_size;
			
			if(i == (device_count - 1))
				chunkBytes = ((chunk_size + rem_size) * matrix_size) * sizeof(float);
			else
				chunkBytes = (chunk_size * matrix_size) * sizeof(float);
			
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpyAsync(&result[start_p], gpu_result[i], chunkBytes, cudaMemcpyDeviceToHost));
		}
		
		for(i = 0; i < device_count; i++){
			checkCudaErrors(cudaSetDevice(i));
			cudaDeviceSynchronize();
		}

		cudaError error_kernel;
	  error_kernel = cudaGetLastError();
		if(error_kernel != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(error_kernel));

		clock_gettime(CLOCK_REALTIME, &end);
		runtime = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
		printf("Running Time: %f\n\n", runtime);
	}
	
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

	//Membebaskan Host
	if(mode < 2){
		cudaFree(gpu_matrixA[0]);
		cudaFree(gpu_matrixB[0]);
		cudaFree(gpu_result[0]);
		free(matrixA);
		free(matrixB);
		free(result);
	}else{
		for(i = 0; i < device_count; i++){
			cudaFree(gpu_matrixA[i]);
			cudaFree(gpu_matrixB[i]);
			cudaFree(gpu_result[i]);
		}

		cudaFreeHost(matrixA);
		cudaFreeHost(matrixB);
		cudaFreeHost(result);
	}

	cudaDeviceReset();
	return 0;
}
