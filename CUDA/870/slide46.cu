#include <stdio.h>
#include <cuda_runtime.h>
__global__ void gpu_insertnum(int *num_vector){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	 num_vector[idx] = 7;
}

__global__ void gpu_blockid(int *blockid_vector){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	 blockid_vector[idx] = blockIdx.x;
}

__global__ void gpu_threadid(int *threadid_vector){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	 threadid_vector[idx] = threadIdx.x;
}

int main(int argc, char** argv){
	int grid_x = atoi(argv[1]);
	int grid_y = atoi(argv[2]);
	int block_x = atoi(argv[3]);
	int block_y = atoi(argv[4]);

	int *num_vector, *blockid_vector, *threadid_vector;
	dim3 grid(grid_x, grid_y), block(block_x, block_y);
	int vectorBytes = (grid_row * grid_col * block_row * block_col) * sizeof(int);
	cudaMalloc((void **) &num_vector, vectorBytes);
	cudaMalloc((void **) &blockid_vector, vectorBytes);
	cudaMalloc((void **) &threadid_vector, vectorBytes);
	
	//gpu_insertnum<<<grid, block>>>(num_vector);
	//gpu_blockid<<<grid, block>>>(blockid_vector);
	gpu_threadid<<<grid, block>>>(threadid_vector);
	
	int *cpu_nvector = (int *)malloc(vectorBytes);
	int *cpu_tvector = (int *)malloc(vectorBytes);
	int *cpu_bvector = (int *)malloc(vectorBytes);

	//cudaMemcpy(cpu_nvector, num_vector, vectorBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_tvector, threadid_vector, vectorBytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(cpu_bvector, blockid_vector, vectorBytes, cudaMemcpyDeviceToHost);
	
	//int i;
	/*printf("Number:\n");
	//for(i = 0; i < (grid_row * grid_col * block_row * block_col); i++)
	//	printf("%d ", cpu_nvector[i]);
	//printf("\n\n");
	
	//printf("Block ID:\n");
	//for(i = 0; i < (grid_row * grid_col * block_row * block_col); i++)
	//	printf("%d ", cpu_bvector[i]);
	//printf("\n\n");*/
	cudaError err;
	err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("Failed: %s\n",  cudaGetErrorString(err));
	}else{
		printf("Success:\n");
	//	for(i = 0; i < (grid_row * grid_col * block_row * block_col); i++)
		//	printf("%d ", cpu_tvector[i]);
	//	printf("\n\n");
	}

	cudaFree(num_vector);
	cudaFree(blockid_vector);
	cudaFree(threadid_vector);

	free(cpu_nvector);
	free(cpu_tvector);
	free(cpu_bvector);
}
