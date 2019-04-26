// incrementArray.cu

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

__global__ void cuda_hello(){
	  printf("hello GPU\n");
}
int main(void)
{
	cuda_hello<<<1, 1>>>();
	printf("hai\n");
	return 0;
}
