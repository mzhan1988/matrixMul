
// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    int stride;
    double* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32
// Matrix size
#define MATRIX_SIZE 5760

void constantInit(double *data, int size, double val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(const Matrix A, int row, int col, double value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}



__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    double Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for(int m=0; m < (A.width / BLOCK_SIZE); ++m)
    {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();
        for(int e=0; e<BLOCK_SIZE; ++e)
        {
            Cvalue += As[row][e] * Bs[e][col];
        }
        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
}

int main(int argc, char **argv)
{
    printf("[Share memory Matrix Multiply Using CUDA] - Starting...\n");

    // By default, we use device 0, otherwise change here
    int devID = 0;
    cudaSetDevice(devID);

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    //Alloc and init host matrix
    Matrix A,B,C;
    A.width = A.height = A.stride = MATRIX_SIZE;
    B.width = B.height = B.stride = MATRIX_SIZE;
    C.width = C.height = C.stride = MATRIX_SIZE;
    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", A.width, A.height, B.width, B.height);

    unsigned int size_A = A.width * A.height;
    unsigned int mem_size_A = sizeof(double) * size_A;
    A.elements = (double *)malloc(mem_size_A);
    unsigned int size_B = B.width * B.height;
    unsigned int mem_size_B = sizeof(double) * size_B;
    B.elements = (double *)malloc(mem_size_B);
    unsigned int size_C = C.width * C.height;
    unsigned int mem_size_C = sizeof(double) * size_C;
    C.elements = (double *)malloc(mem_size_C);

    const double valA = 1.0f;
    const double valB = 0.01f;
    constantInit(A.elements, size_A, valA);
    constantInit(B.elements, size_B, valB);

    /*
    for(int i=0; i<size_A; i++)
    {
        printf("%f\n", A.elements[i]);
        printf("%f\n", B.elements[i]);
    }
    */

    //Alloc and init device matrix
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    d_A.stride = A.stride;
    cudaMalloc(&d_A.elements, mem_size_A);
    cudaMemcpy(d_A.elements, A.elements, mem_size_A, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    d_B.stride = B.stride;
    cudaMalloc(&d_B.elements, mem_size_B);
    cudaMemcpy(d_B.elements, B.elements, mem_size_B, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    d_C.stride = C.stride;
    cudaMalloc(&d_C.elements, mem_size_C);

    //launch kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(C.width / dimBlock.x, C.height / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    //copy back
    cudaMemcpy(C.elements, d_C.elements, mem_size_C, cudaMemcpyDeviceToHost);

    //result check
    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(C.width * C.height); i++)
    {
        double abs_err = fabs(C.elements[i] - (A.width * valB));
        double dot_length = A.width;
        double abs_val = fabs(C.elements[i]);
        double rel_err = abs_err/abs_val/dot_length ;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, C.elements[i], A.width*valB, eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    /*
    for(int i=0; i<size_C; i++)
    {
        printf("%f\n", C.elements[i]);
    }
    */
    cudaDeviceSynchronize();

    //Get Gflops
    cudaEvent_t start;
    error = cudaEventCreate(&start);
    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    int nIter = 10;
    // Record the start event
    error = cudaEventRecord(start, NULL);
    for (int j = 0; j < nIter; j++)
    {
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    }
    error = cudaEventRecord(stop, NULL);
    error = cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)A.width * (double)A.height * (double)B.width;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        dimBlock.x * dimBlock.y);

    cudaDeviceSynchronize();

    free(A.elements);
    free(B.elements);
    free(C.elements);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    return 0;
}
