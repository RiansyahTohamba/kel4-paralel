//Note:
//Cara running program ./nama_file mode besar_matrix besar_grid besar_block
//Ukuran matrix: besar_matrix x besar matrix
//Grid: besar_grid x besar_grid (block per grid)                        | Max: Mengacu pada NVIDIA Compute Capability dari setiap seri GPU
//Block: besar_block x besar_block (thread per block)                   | Max: Mengacu pada NVIDIA Compute Capability dari setiap seri GPU
// Mode:
// 0: Matrix multiplication pada 1 GPU tanpa melihat hasil sekuensial
// 1: Matrix multiplication pada 1 GPU dengan hasil sekuensial
// 2: Matrix multiplication pada multiple GPU tanpa melihat hasil sekuensial
// 3: Matrix multiplication pada multiple GPU dengan hasil sekuensial
// mode 2 ketas belum selesai dikerjakan

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

//Operasi perkalian matrix pada gpu
__global__ void matrixmul_kernel(int *gpu_matrixA, int *gpu_matrixB, int *gpu_result, int matrix_size, int grid, int block){
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
                    int A = gpu_matrixA[(row_index * displacement + m) * matrix_size + l];
                    int B = gpu_matrixB[l * matrix_size + (col_index * displacement + n)];
                    R += A * B;
                }
                gpu_result[(row_index * displacement + m) * matrix_size + (col_index * displacement + n)] = R;
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
    int matrixBytes = (matrix_size * matrix_size) * sizeof(int);
    int *matrixA = (int *)malloc(matrixBytes);
    int *matrixB = (int *)malloc(matrixBytes);
    int *result = (int *)malloc(matrixBytes);
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
        int *gpu_matrixA, *gpu_matrixB, *gpu_result;
        cudaMalloc((void **) &gpu_matrixA, matrixBytes);
        cudaMalloc((void **) &gpu_matrixB, matrixBytes);
        cudaMalloc((void **) &gpu_result, matrixBytes);
        cudaMemcpy(gpu_matrixA, matrixA, matrixBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_matrixB, matrixB, matrixBytes, cudaMemcpyHostToDevice);

        //Mulai operasi pada device
        dim3 grid(igrid, igrid);
        dim3 block(iblock, iblock);
        matrixmul_kernel<<<grid, block>>>(gpu_matrixA, gpu_matrixB, gpu_result, matrix_size, igrid, iblock);

		//Return hasil perkalian
        cudaMemcpy(result, gpu_result, matrixBytes, cudaMemcpyDeviceToHost);
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
        int *seqresult = (int *)malloc(matrixBytes);
        for (i = 0; i < matrix_size; i++){
            for (j = 0; j < matrix_size; j++){
                seqresult[i * matrix_size + j] = 0;
                for (k = 0; k < matrix_size; k++)
                    seqresult[i * matrix_size + j] += matrixA[i * matrix_size + k] * matrixB[k * matrix_size + j];
                if(seqresult[i * matrix_size + j] == result[i * matrix_size + j]) right_answer += 1;
                //printf("%d - %d S: %d, CUDA: %d\n", i * matrix_size, j, seqresult[i * matrix_size + j], result[i * matrix_size + j]);
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

