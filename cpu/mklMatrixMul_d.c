
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "mkl.h"

#define MATRIX_SIZE 23040
#define ITER 10

#define min(x,y) (((x) < (y)) ? (x):(y))

typedef struct
{
    unsigned int wA;
    unsigned int hA;
    unsigned int wB;
    unsigned int hB;
    unsigned int wC;
    unsigned int hC;
} MatrixSize;

void randomInit(double *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand() / (double)RAND_MAX;
    }
}

void printMatrix(double *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        printf("%lf\n", data[i]);
    }
}

int main()
{
    //init matrix size
    MatrixSize matrix_size;
    matrix_size.wA = MATRIX_SIZE;
    matrix_size.hA = MATRIX_SIZE;
    matrix_size.wB = MATRIX_SIZE;
    matrix_size.hB = MATRIX_SIZE;
    matrix_size.wC = MATRIX_SIZE;
    matrix_size.hC = MATRIX_SIZE;

    double alpha = 1.0;
    double beta = 0.0;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.wA, matrix_size.hA,
           matrix_size.wB, matrix_size.hB,
           matrix_size.wC, matrix_size.hC);

    //init matrix
    unsigned int size_A = matrix_size.wA * matrix_size.hA;
    unsigned int mem_size_A = sizeof(double) * size_A;
    double *h_A = (double *)mkl_malloc(mem_size_A, 64);
    unsigned int size_B = matrix_size.wB * matrix_size.hB;
    unsigned int mem_size_B = sizeof(double) * size_B;
    double *h_B = (double *)mkl_malloc(mem_size_B, 64);
    unsigned int size_C = matrix_size.wC * matrix_size.hC;
    unsigned int mem_size_C = sizeof(double) * size_C;
    double *h_C = (double *)mkl_malloc(mem_size_C, 64);

    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    memset(h_C, 0, mem_size_C);
    //printMatrix(h_A, 100);
    //printMatrix(h_B, 100);
    //printMatrix(h_C, 100);

    struct timeval start;
    struct timeval end;
    double msdiff;

    printf("Start testing...\n");
    int nIter = ITER;
    gettimeofday(&start, NULL);
    for(int j=0; j<nIter; ++j)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_size.hA, matrix_size.wB, matrix_size.wA, alpha, h_A, matrix_size.wA, h_B, matrix_size.wB, beta, h_C, matrix_size.wC);
    }
    gettimeofday(&end, NULL);
    msdiff = 1000.0*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000.0;
    printf("Done...\n");

    double msecPerMatrixMul = msdiff / nIter;
    double flopsPerMatrixMul = 2.0 * (double)matrix_size.hC * (double)matrix_size.wC * (double)matrix_size.hB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    //printMatrix(h_C, 100);
    mkl_free(h_A);
    mkl_free(h_B);
    mkl_free(h_C);

    return 0;
}
