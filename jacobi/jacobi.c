
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  MPI_Initss(NULL, NULL);

  Parallel_jacobi(5, 4,100,10.5);

  MPI_Finalize();
}

// implements a maximum norm 
int distance(x_old,x_new){
  max_norm = 10;
  return max_norm;
}      
// returns array global x which contains 
// the last approximation vector to be the final result.
void output(x_new,global_x){
  printf("Here is the last approximation vector:\n");
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++)
      printf("%6.2f   ", c[i][j]);
    printf ("\n");
  } 
}
int Parallel_jacobi(int n, int process,int max_it, float tol)
{
   int GLOB_MAX = 100;
   float MPI_FLOAT = 100.0;

   int i_local, i_global, j, i;
   int n_local, it_num;
   float x_temp1[GLOB_MAX], x_temp2[GLOB_MAX], local_x[GLOB_MAX];
   float *x_old, *x_new,*temp;

   n_local = n/process; /*local blocksize*/

   MPI_Allgather(local_b,n_local,MPI_FLOAT,x_temp1,n_local,MPI_FLOAT,MPI_COMM_WORLD);

   x_new = x_temp1;
   x_old = x_temp2;
   it_num = 0;
   do {
    it_num++;
    temp = x_new;
    x_new = x_old;
    x_old = temp;
    for (i_local = 0; i_local < n_local; i_local++){
      i_global = i_local + me * n_local;
      local_x[i_local] = local_b[i_local];
      for (j = 0; j < i_global; j++)
        local_x[i_local] = local_x[i_local] - local_A[i_local][j] * x_old[j];

      for (j = i_global+1; j < n; j++)
        local_x[i_local] = local_x[i_local] - local_A[i_local][j] * x_old[j];


      local_x[i_local] = local_x[i_local] / local_A[i_local][i_global];
    }
    MPI_Allgather(local_x,n_local,MPI_FLOAT,x_new,n_local,MPI_FLOAT,MPI_COMM_WORLD);
   } while((it_num < max_it) && (distance(x_old,x_new) >= tol));
   output(x_new,global_x);
   if(distance(x_old,x_new) < tol) return 1;
   else 0;
}

