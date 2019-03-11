EXECS=normal_multiple_matrix
MPICC?=mpicc

normal: normal_multiple_matrix.c
	mpicc -o normal_multiple_matrix normal_multiple_matrix.c
	mpiexec -np $(PRS) ./normal_multiple_matrix
	
clean:
	rm -f ${EXECS}