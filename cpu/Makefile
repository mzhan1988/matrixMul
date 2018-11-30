all:
	gcc -std=c99 -O3 -c -I./CBLAS/include -L./CBLAS/lib/ blasMatrixMul_d.c 
	gfortran -o blasMatrixMul_d blasMatrixMul_d.o ./CBLAS/lib/cblas_LINUX.a ../BLAS-3.8.0/blas_LINUX.a

mkl:
	gcc -std=c99 mklMatrixMul_d.c -o mklMatrixMul_d -lmkl_rt
#	gcc -c -I./CBLAS/include -L./CBLAS/lib/ blastest.c 
#	gfortran -o blastest blastest.o ./CBLAS/lib/cblas_LINUX.a ../BLAS-3.8.0/blas_LINUX.a
#	gcc -std=c99 simpleMatrixMul_f.c -o simpleMatrixMul_f
#	gcc -O2 -std=c99 simpleMatrixMul_d.c -o simpleMatrixMul_d
