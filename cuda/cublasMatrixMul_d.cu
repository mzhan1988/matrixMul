
// Utilities and system includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

#define MATRIX_SIZE 11520
#define ITER 10

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

int main(int argc, char **argv)
{
    printf("CUBLAS MatrixMul test - Starting...\n");

    //init matrix size
    MatrixSize matrix_size;
    matrix_size.wA = MATRIX_SIZE;
    matrix_size.hA = MATRIX_SIZE;
    matrix_size.wB = MATRIX_SIZE;
    matrix_size.hB = MATRIX_SIZE;
    matrix_size.wC = MATRIX_SIZE;
    matrix_size.hC = MATRIX_SIZE;
    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.wA, matrix_size.hA,
           matrix_size.wB, matrix_size.hB,
           matrix_size.wC, matrix_size.hC);

    //init matrix
    unsigned int size_A = matrix_size.wA * matrix_size.hA;
    unsigned int mem_size_A = sizeof(double) * size_A;
    double *h_A = (double *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.wB * matrix_size.hB;
    unsigned int mem_size_B = sizeof(double) * size_B;
    double *h_B = (double *)malloc(mem_size_B);
    unsigned int size_C = matrix_size.wC * matrix_size.hC;
    unsigned int mem_size_C = sizeof(double) * size_C;
    double *h_C = (double *)malloc(mem_size_C);

    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    memset(h_C, 0, mem_size_C);

    double *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_A);
    cudaMalloc((void **) &d_C, mem_size_A);

    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    //get cuda device
    int devID = 0;
    cudaError_t error;
    cudaSetDevice(devID);
    cudaDeviceProp deviceProp;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);



    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.wC / threads.x, matrix_size.hC / threads.y);
    const double alpha = 1.0f;
    const double beta  = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //Perform warmup operation with cublas
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.wB, matrix_size.hA, matrix_size.wA, &alpha, d_B, matrix_size.wB, d_A, matrix_size.wA, &beta, d_C, matrix_size.wC);
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    int nIter = ITER;
    cudaEventRecord(start, NULL);
    for (int j = 0; j < nIter; j++)
    {   
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.wB, matrix_size.hA, matrix_size.wA, &alpha, d_B, matrix_size.wB, d_A, matrix_size.wA, &beta, d_C, matrix_size.wC);
    }   
    error = cudaEventRecord(stop, NULL);
    error = cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)matrix_size.wA * (double)matrix_size.hA * (double)matrix_size.wB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops.\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);

    cudaDeviceSynchronize();
    //copy bak
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    //check result
    //printMatrix(h_C, 100);
    return 0;
}
