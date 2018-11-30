
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
float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32
// Matrix size
#define MATRIX_SIZE 5760

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 0;
    for(int e=0; e < A.width; ++e)
    {
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    }
    C.elements[row * C.width + col] = Cvalue;
}


int main(int argc, char **argv)
{
    printf("[Simple Matrix Multiply Using CUDA] - Starting...\n");

    // By default, we use device 0, otherwise change here
    int devID = 0;
    cudaSetDevice(devID);

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    //Alloc and init host matrix
    Matrix A,B,C;
    A.width = A.height = MATRIX_SIZE;
    B.width = B.height = MATRIX_SIZE;
    C.width = C.height = MATRIX_SIZE;
    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", A.width, A.height, B.width, B.height);

    unsigned int size_A = A.width * A.height;
    unsigned int mem_size_A = sizeof(float) * size_A;
    A.elements = (float *)malloc(mem_size_A);
    unsigned int size_B = B.width * B.height;
    unsigned int mem_size_B = sizeof(float) * size_B;
    B.elements = (float *)malloc(mem_size_B);
    unsigned int size_C = C.width * C.height;
    unsigned int mem_size_C = sizeof(float) * size_C;
    C.elements = (float *)malloc(mem_size_C);

    const float valA = 1.0f;
    const float valB = 0.01f;
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
    cudaMalloc(&d_A.elements, mem_size_A);
    cudaMemcpy(d_A.elements, A.elements, mem_size_A, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    cudaMalloc(&d_B.elements, mem_size_B);
    cudaMemcpy(d_B.elements, B.elements, mem_size_B, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
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
