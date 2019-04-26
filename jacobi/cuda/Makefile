EXECS=cuda_jacobi

all:  jacobi 
	
jacobi:
	nvcc jacobi.cu -o ${EXECS} -gencode arch=compute_52,code=sm_52 

clean:
	rm -f ${EXECS}
