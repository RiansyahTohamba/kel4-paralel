EXECS=mpi_hello_world
MPICC?=mpicc

all: ${EXECS}

mpi_hello_world: mpi_hello_world.c
	${MPICC} -o mpi_hello_world mpi_hello_world.c

clean:
	rm -f ${EXECS}

compile_mpi:
	mpicc topic_2.c -o topic_2
	mpiexec -np 4 ./topic_2

