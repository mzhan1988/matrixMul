
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

#define MATRIX_SIZE 3
#define ITER 1

typedef struct
{
    unsigned int wA; 
    unsigned int hA; 
    unsigned int wB; 
    unsigned int hB; 
    unsigned int wC; 
    unsigned int hC; 
} MatrixSize;

int main(int argc, char **argv)
{
    printf("Simple CUBLAS test - Starting...\n");

    int i;
    double A[9] = { 1.0, 2.0, 1.0, -3.0, 4.0, 0.0, -1.0, 2.0, 3.0 };
    double B[9] = { 2.0, 3.0, 1.0, 1.0, -2.0, -3.0, -1.0, 3.0, 1.0 };
    double C[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double D[9] = { 3.0, 2.0, -4.0, -2.0, -17.0, -15.0, -3.0, 2.0, -4.0 };

    int devID = 0;
    cudaSetDevice(devID);
    cudaDeviceProp deviceProp;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    //init matrix size
    MatrixSize matrix_size;
    matrix_size.wA = MATRIX_SIZE;
    matrix_size.hA = MATRIX_SIZE;
    matrix_size.wB = MATRIX_SIZE;
    matrix_size.hB = MATRIX_SIZE;
    matrix_size.wC = MATRIX_SIZE;
    matrix_size.hC = MATRIX_SIZE;

    //CUBLAS
    int block_size = 3;
    unsigned int size_A = matrix_size.wA * matrix_size.hA;
    unsigned int mem_size_A = sizeof(double) * size_A;
    unsigned int size_B = matrix_size.wB * matrix_size.hB;
    unsigned int mem_size_B = sizeof(double) * size_B;
    unsigned int size_C = matrix_size.wC * matrix_size.hC;
    unsigned int mem_size_C = sizeof(double) * size_C;

    double *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_A);
    cudaMalloc((void **) &d_C, mem_size_A);

    cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);

    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.wC / threads.x, matrix_size.hC / threads.y);

    const double alpha = 1.0f;
    const double beta  = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.wB, matrix_size.hA, matrix_size.wA, &alpha, d_B, matrix_size.wB, d_A, matrix_size.wA, &beta, d_C, matrix_size.wB);

    cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    //check result
    for (i = 0; i<9; i++)
    {   
        printf("%lf ",C[i]);
    }   
    printf("\n");
    for (i = 0; i<9; i++)
    {   
        printf("%lf ",D[i]);
    }   
    printf("\n");
    return 0;
}
