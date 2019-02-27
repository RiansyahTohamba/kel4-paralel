
#include <mpi.h>
#include <stdio.h>

main(){
	printf("==== result with normal matrix multiplication ===");
	multiple_vector_matrix(5, 12,100,10.5);
	printf("==== result with algorithm ==== ");
	multiple_vector_matrix(5, 12,100,10.5);
}

int multiple_vector_matrix(int n, int p,int max_it, float tol)
{

}

