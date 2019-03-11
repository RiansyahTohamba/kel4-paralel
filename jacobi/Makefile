EXECS=jacobi

all:  jacobi 
	
jacobi: 
	mpicc -o ${EXECS} ${EXECS}.c
	mpiexec -np $(PRS) ./${EXECS}

clean:
	rm -f ${EXECS}
