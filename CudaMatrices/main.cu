#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <ctime>
#include <algorithm>
#include <stdio.h>

/// Tile size used by the OptimizedMMKernel
#define TILE_SIZE 32

/// Naive matrix multiplication CUDA Kernel
__global__ void NaiveMMKernel(float *a, float *b, float *c, int size)
{
	int xOut = blockDim.x * blockIdx.x + threadIdx.x;
	int yOut = blockDim.y * blockIdx.y + threadIdx.y;

	float outValue = 0;
	for (int i = 0; i < size; i++)
	{
		// Row of a mulitplied by the column of b
		float prod = a[yOut * size + i] * b[i * size + xOut];
		outValue += prod;
	}

	// Store sum of dot products in C matrix
	c[yOut * size + xOut] = outValue;
}

/// Tiled 1D Shared Memory No Unrolling
__global__ void OptimizedMMKernel_0(float *a, float *b, float *c, int size)
{
	// Create shared matrices for rows of A and columns of B
	__shared__ float sharedA[TILE_SIZE * TILE_SIZE];
	__shared__ float sharedB[TILE_SIZE * TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	float sum = 0;

	// Divide the matrix up into tiles based on the tile size so each thread
	// Can perform its partial sum of the dot product from the shared matrix
	int tilesPerGrid = size / blockDim.x;
	for (int i = 0; i < tilesPerGrid; i++)
	{
		// Each thread loads element into A and B
		sharedA[ty * TILE_SIZE + tx] = a[(y * size) + (i * TILE_SIZE) + tx];
		sharedB[ty * TILE_SIZE + tx] = b[(i * TILE_SIZE * size) + (ty * size) + x];

		// Wait for all threads to load each section of the shared matrix
		__syncthreads();

		for (int j = 0; j < TILE_SIZE; j++)
		{
			sum += sharedA[ty * TILE_SIZE + j] * sharedB[j * TILE_SIZE + tx];
		}

		// Wait for all threads to compute their partial sum from the shared matrices before loading the next
		__syncthreads();
	}

	// Store the full sum as the result
	c[y * size + x] = sum;
}

/// Tiled 2D Shared Memory No Unrolling
__global__ void OptimizedMMKernel_1(float *a, float *b, float *c, int size)
{
	// Create shared matrices for rows of A and columns of B
	__shared__ float sharedA[TILE_SIZE][TILE_SIZE];
	__shared__ float sharedB[TILE_SIZE][TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	float sum = 0;

	// Divide the matrix up into tiles based on the tile size so each thread
	// Can perform its partial sum of the dot product from the shared matrix
	int tilesPerGrid = size / blockDim.x;
	for (int i = 0; i < tilesPerGrid; i++)
	{
		// Each thread loads element into A and B
		sharedA[ty][tx] = a[(y * size) + (i * TILE_SIZE) + tx];
		sharedB[ty][tx] = b[(i * TILE_SIZE * size) + (ty * size) + x];

		// Wait for all threads to load each section of the shared matrix
		__syncthreads();

		for (int j = 0; j < TILE_SIZE; j++)
		{
			sum += sharedA[ty][j] * sharedB[j][tx];
		}

		// Wait for all threads to compute their partial sum from the shared matrices before loading the next
		__syncthreads();
	}

	// Store the full sum as the result
	c[y * size + x] = sum;
}

/// Tiled 2D Shared Memory With Unrolling (4x4 Tile Size)
__global__ void OptimizedMMKernel_2_4(float *a, float *b, float *c, int size)
{
	// Create shared matrices for rows of A and columns of B
	__shared__ float sharedA[4][4];
	__shared__ float sharedB[4][4];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	float sum = 0;

	// Divide the matrix up into tiles based on the tile size so each thread
	// Can perform its partial sum of the dot product from the shared matrix
	int tilesPerGrid = size / blockDim.x;
	for (int i = 0; i < tilesPerGrid; i++)
	{
		// Each thread loads element into A and B
		sharedA[ty][tx] = a[(y * size) + (i * 4) + tx];
		sharedB[ty][tx] = b[(i * 4 * size) + (ty * size) + x];

		// Wait for all threads to load each section of the shared matrix
		__syncthreads();

		sum += sharedA[ty][0] * sharedB[0][tx];
		sum += sharedA[ty][1] * sharedB[1][tx];
		sum += sharedA[ty][2] * sharedB[2][tx];
		sum += sharedA[ty][3] * sharedB[3][tx];

		// Wait for all threads to compute their partial sum from the shared matrices before loading the next
		__syncthreads();
	}

	// Store the full sum as the result
	c[y * size + x] = sum;
}

/// Tiled 2D Shared Memory With Unrolling (8x8 Tile Size)
__global__ void OptimizedMMKernel_2_8(float *a, float *b, float *c, int size)
{
	// Create shared matrices for rows of A and columns of B
	__shared__ float sharedA[8][8];
	__shared__ float sharedB[8][8];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	float sum = 0;

	// Divide the matrix up into tiles based on the tile size so each thread
	// Can perform its partial sum of the dot product from the shared matrix
	int tilesPerGrid = size / blockDim.x;
	for (int i = 0; i < tilesPerGrid; i++)
	{
		// Each thread loads element into A and B
		sharedA[ty][tx] = a[(y * size) + (i * 8) + tx];
		sharedB[ty][tx] = b[(i * 8 * size) + (ty * size) + x];

		// Wait for all threads to load each section of the shared matrix
		__syncthreads();

		sum += sharedA[ty][0] * sharedB[0][tx];
		sum += sharedA[ty][1] * sharedB[1][tx];
		sum += sharedA[ty][2] * sharedB[2][tx];
		sum += sharedA[ty][3] * sharedB[3][tx];
		sum += sharedA[ty][4] * sharedB[4][tx];
		sum += sharedA[ty][5] * sharedB[5][tx];
		sum += sharedA[ty][6] * sharedB[6][tx];
		sum += sharedA[ty][7] * sharedB[7][tx];

		// Wait for all threads to compute their partial sum from the shared matrices before loading the next
		__syncthreads();
	}

	// Store the full sum as the result
	c[y * size + x] = sum;
}

/// Tiled 2D Shared Memory With Unrolling (16x16 Tile Size)
__global__ void OptimizedMMKernel_2_16(float *a, float *b, float *c, int size)
{
	// Create shared matrices for rows of A and columns of B
	__shared__ float sharedA[16][16];
	__shared__ float sharedB[16][16];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	float sum = 0;

	// Divide the matrix up into tiles based on the tile size so each thread
	// Can perform its partial sum of the dot product from the shared matrix
	int tilesPerGrid = size / blockDim.x;
	for (int i = 0; i < tilesPerGrid; i++)
	{
		// Each thread loads element into A and B
		sharedA[ty][tx] = a[(y * size) + (i * 16) + tx];
		sharedB[ty][tx] = b[(i * 16 * size) + (ty * size) + x];

		// Wait for all threads to load each section of the shared matrix
		__syncthreads();

		sum += sharedA[ty][0] * sharedB[0][tx];
		sum += sharedA[ty][1] * sharedB[1][tx];
		sum += sharedA[ty][2] * sharedB[2][tx];
		sum += sharedA[ty][3] * sharedB[3][tx];
		sum += sharedA[ty][4] * sharedB[4][tx];
		sum += sharedA[ty][5] * sharedB[5][tx];
		sum += sharedA[ty][6] * sharedB[6][tx];
		sum += sharedA[ty][7] * sharedB[7][tx];
		sum += sharedA[ty][8] * sharedB[8][tx];
		sum += sharedA[ty][9] * sharedB[9][tx];
		sum += sharedA[ty][10] * sharedB[10][tx];
		sum += sharedA[ty][11] * sharedB[11][tx];
		sum += sharedA[ty][12] * sharedB[12][tx];
		sum += sharedA[ty][13] * sharedB[13][tx];
		sum += sharedA[ty][14] * sharedB[14][tx];
		sum += sharedA[ty][15] * sharedB[15][tx];

		// Wait for all threads to compute their partial sum from the shared matrices before loading the next
		__syncthreads();
	}

	// Store the full sum as the result
	c[y * size + x] = sum;
}

/// Tiled 2D Shared Memory With Unrolling (32x32 Tile Size)
__global__ void OptimizedMMKernel_2_32(float *a, float *b, float *c, int size)
{
	// Create shared matrices for rows of A and columns of B
	__shared__ float sharedA[32][32];
	__shared__ float sharedB[32][32];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	float sum = 0;

	// Divide the matrix up into tiles based on the tile size so each thread
	// Can perform its partial sum of the dot product from the shared matrix
	int tilesPerGrid = size / blockDim.x;
	for (int i = 0; i < tilesPerGrid; i++)
	{
		// Each thread loads element into A and B
		sharedA[ty][tx] = a[(y * size) + (i * 32) + tx];
		sharedB[ty][tx] = b[(i * 32 * size) + (ty * size) + x];

		// Wait for all threads to load each section of the shared matrix
		__syncthreads();

		sum += sharedA[ty][0] * sharedB[0][tx];
		sum += sharedA[ty][1] * sharedB[1][tx];
		sum += sharedA[ty][2] * sharedB[2][tx];
		sum += sharedA[ty][3] * sharedB[3][tx];
		sum += sharedA[ty][4] * sharedB[4][tx];
		sum += sharedA[ty][5] * sharedB[5][tx];
		sum += sharedA[ty][6] * sharedB[6][tx];
		sum += sharedA[ty][7] * sharedB[7][tx];
		sum += sharedA[ty][8] * sharedB[8][tx];
		sum += sharedA[ty][9] * sharedB[9][tx];
		sum += sharedA[ty][10] * sharedB[10][tx];
		sum += sharedA[ty][11] * sharedB[11][tx];
		sum += sharedA[ty][12] * sharedB[12][tx];
		sum += sharedA[ty][13] * sharedB[13][tx];
		sum += sharedA[ty][14] * sharedB[14][tx];
		sum += sharedA[ty][15] * sharedB[15][tx];
		sum += sharedA[ty][16] * sharedB[16][tx];
		sum += sharedA[ty][17] * sharedB[17][tx];
		sum += sharedA[ty][18] * sharedB[18][tx];
		sum += sharedA[ty][19] * sharedB[19][tx];
		sum += sharedA[ty][20] * sharedB[20][tx];
		sum += sharedA[ty][21] * sharedB[21][tx];
		sum += sharedA[ty][22] * sharedB[22][tx];
		sum += sharedA[ty][23] * sharedB[23][tx];
		sum += sharedA[ty][24] * sharedB[24][tx];
		sum += sharedA[ty][25] * sharedB[25][tx];
		sum += sharedA[ty][26] * sharedB[26][tx];
		sum += sharedA[ty][27] * sharedB[27][tx];
		sum += sharedA[ty][28] * sharedB[28][tx];
		sum += sharedA[ty][29] * sharedB[29][tx];
		sum += sharedA[ty][30] * sharedB[30][tx];
		sum += sharedA[ty][31] * sharedB[31][tx];

		// Wait for all threads to compute their partial sum from the shared matrices before loading the next
		__syncthreads();
	}

	// Store the full sum as the result
	c[y * size + x] = sum;
}

/// Prints a matrix out to the stderr stream
void PrintMatrix(float *matrix, int width, int height)
{
	for (int i = 0; i < width * height; i++)
	{
		if (i % width == 0)
		{
			fprintf(stderr, "[ ");
		}

		if (matrix == nullptr)
		{
			fprintf(stderr, "NULL ");
		}
		else
		{
			fprintf(stderr, "%.6f ", matrix[i]);
		}

		if (i % width == width - 1)
		{
			fprintf(stderr, "]\n");
		}
	}
}

/// Checks a cuda call to make sure its OK
void CheckCudaCall(cudaError_t callResult, const char *message)
{
	if (callResult != cudaSuccess)
	{
		fprintf(stderr, message);
	}
}

/// Calls the naive matrix multiplication kernel
float* NaiveMM(float *a, float *b, int size, int tpb, float *kernelTime, float* totalTime)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Create events
	cudaEvent_t kernelStart, kernelStop, totalStart, totalStop;
	CheckCudaCall(cudaEventCreate(&kernelStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&kernelStop), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStop), "cudaEventCreate failed");

	// Allocate device matrices
 	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// Sstart the timer here for the total time
	cudaEventRecord(totalStart, 0);

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	// Start the timer here for the kernel time
	cudaEventRecord(kernelStart, 0);

	// Call kernel
	dim3 blockSize(size / tpb, size / tpb);
	dim3 threadsPerBlock(tpb, tpb);
	NaiveMMKernel<<<blockSize, threadsPerBlock>>>(devA, devB, devC, size);

	// Stop kernel timer here
	cudaEventRecord(kernelStop, 0);
	cudaEventSynchronize(kernelStop);

	// Copy result back
	cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost);

	// Stop total time here
	cudaEventRecord(totalStop, 0);
	cudaEventSynchronize(totalStop);

	// Calculate elapsed times
	cudaEventElapsedTime(kernelTime, kernelStart, kernelStop);
	cudaEventElapsedTime(totalTime, totalStart, totalStop);

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFree(kernelStart);
	cudaFree(kernelStop);
	cudaFree(totalStart);
	cudaFree(totalStop);
	
	// Return result
	return result;
}

/// Calls the optimized (shared memory) matrix multiplication kernel
float* OptimizedMM_0(float *a, float *b, int size, float *kernelTime, float* totalTime)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Create events
	cudaEvent_t kernelStart, kernelStop, totalStart, totalStop;
	CheckCudaCall(cudaEventCreate(&kernelStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&kernelStop), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStop), "cudaEventCreate failed");

	// Allocate device matrices
	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// Sstart the timer here for the total time
	cudaEventRecord(totalStart, 0);

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	// Start the timer here for the kernel time
	cudaEventRecord(kernelStart, 0);

	// Call kernel
	dim3 blockSize(size / TILE_SIZE, size / TILE_SIZE);
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	OptimizedMMKernel_0<<<blockSize, threadsPerBlock >>>(devA, devB, devC, size);

	// Stop kernel timer here
	cudaEventRecord(kernelStop, 0);
	cudaEventSynchronize(kernelStop);

	// Copy result back
	cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost);

	// Stop total time here
	cudaEventRecord(totalStop, 0);
	cudaEventSynchronize(totalStop);

	// Calculate elapsed times
	cudaEventElapsedTime(kernelTime, kernelStart, kernelStop);
	cudaEventElapsedTime(totalTime, totalStart, totalStop);

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFree(kernelStart);
	cudaFree(kernelStop);
	cudaFree(totalStart);
	cudaFree(totalStop);

	// Return result
	return result;
}

/// Calls the optimized (shared memory) matrix multiplication kernel
float* OptimizedMM_1(float *a, float *b, int size, float *kernelTime, float* totalTime)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Create events
	cudaEvent_t kernelStart, kernelStop, totalStart, totalStop;
	CheckCudaCall(cudaEventCreate(&kernelStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&kernelStop), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStop), "cudaEventCreate failed");

	// Allocate device matrices
	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// Sstart the timer here for the total time
	cudaEventRecord(totalStart, 0);

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	// Start the timer here for the kernel time
	cudaEventRecord(kernelStart, 0);

	// Call kernel
	dim3 blockSize(size / TILE_SIZE, size / TILE_SIZE);
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	OptimizedMMKernel_1<<<blockSize, threadsPerBlock >>> (devA, devB, devC, size);

	// Stop kernel timer here
	cudaEventRecord(kernelStop, 0);
	cudaEventSynchronize(kernelStop);

	// Copy result back
	cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost);

	// Stop total time here
	cudaEventRecord(totalStop, 0);
	cudaEventSynchronize(totalStop);

	// Calculate elapsed times
	cudaEventElapsedTime(kernelTime, kernelStart, kernelStop);
	cudaEventElapsedTime(totalTime, totalStart, totalStop);

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFree(kernelStart);
	cudaFree(kernelStop);
	cudaFree(totalStart);
	cudaFree(totalStop);

	// Return result
	return result;
}

/// Calls the optimized (shared memory) matrix multiplication kernel
float* OptimizedMM_2(float *a, float *b, int size, float *kernelTime, float* totalTime)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Create events
	cudaEvent_t kernelStart, kernelStop, totalStart, totalStop;
	CheckCudaCall(cudaEventCreate(&kernelStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&kernelStop), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStop), "cudaEventCreate failed");

	// Allocate device matrices
	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// Sstart the timer here for the total time
	cudaEventRecord(totalStart, 0);

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	// Start the timer here for the kernel time
	cudaEventRecord(kernelStart, 0);

	// Call kernel
	dim3 blockSize(size / TILE_SIZE, size / TILE_SIZE);
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	switch (TILE_SIZE)
	{
		case 4:
			OptimizedMMKernel_2_4<<<blockSize, threadsPerBlock>>>(devA, devB, devC, size);
			break;
		case 8:
			OptimizedMMKernel_2_8<<<blockSize, threadsPerBlock>>>(devA, devB, devC, size);
			break;
		case 16:
			OptimizedMMKernel_2_16<<<blockSize, threadsPerBlock>>>(devA, devB, devC, size);
			break;
		case 32:
			OptimizedMMKernel_2_32<<<blockSize, threadsPerBlock>>>(devA, devB, devC, size);
			break;
	}
	

	// Stop kernel timer here
	cudaEventRecord(kernelStop, 0);
	cudaEventSynchronize(kernelStop);

	// Copy result back
	cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost);

	// Stop total time here
	cudaEventRecord(totalStop, 0);
	cudaEventSynchronize(totalStop);

	// Calculate elapsed times
	cudaEventElapsedTime(kernelTime, kernelStart, kernelStop);
	cudaEventElapsedTime(totalTime, totalStart, totalStop);

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFree(kernelStart);
	cudaFree(kernelStop);
	cudaFree(totalStart);
	cudaFree(totalStop);

	// Return result
	return result;
}

/// Calls the cublasSgemm function, to multiply 2 matrices
float* CublasMM(cublasHandle_t &handle, float *a, float *b, int size, float *kernelTime, float* totalTime)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Create events
	cudaEvent_t kernelStart, kernelStop, totalStart, totalStop;
	CheckCudaCall(cudaEventCreate(&kernelStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&kernelStop), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStart), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&totalStop), "cudaEventCreate failed");

	// Allocate device matrices
	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// Sstart the timer here for the total time
	cudaEventRecord(totalStart, 0);

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	// Start the timer here for the kernel time
	cudaEventRecord(kernelStart, 0);

	// Initialize cublas params
	const float alp = 1;
	const float bet = 0;
	const float *alpha = &alp;
	const float *beta = &bet;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size, alpha, devA, size, devB, size, beta, devC, size);

	// Stop kernel timer here
	cudaEventRecord(kernelStop, 0);
	cudaEventSynchronize(kernelStop);

	// Copy result back
	CheckCudaCall(cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost), "cudaMemcpy failed");

	// Stop total time here
	cudaEventRecord(totalStop, 0);
	cudaEventSynchronize(totalStop);

	// Calculate elapsed times
	cudaEventElapsedTime(kernelTime, kernelStart, kernelStop);
	cudaEventElapsedTime(totalTime, totalStart, totalStop);

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cudaFree(kernelStart);
	cudaFree(kernelStop);
	cudaFree(totalStart);
	cudaFree(totalStop);

	// Return result
	return result;
}

/// Ranomizes a matrix with random floats in the range [0, 5)
void RandomizeMatrix(float *mat, int size)
{
	for (int i = 0; i < size * size; i++) 
	{
		double f = (double)rand() / RAND_MAX;
		mat[i] = (float)(f * 5);
	}
}

/// Sets a matrix's elements as 1..N for testing purposes
void FillMatrixInOrder(float *mat, int size)
{
	for (int i = 0; i < size * size; i++) 
	{
		mat[i] = (float)(i + 1);
	}
}

/// Calculates residual sum between a transposed cuBLAS matrix and a row-major ordered output matrix
float MatrixResidual(float *cublas, float *test, int w, int h)
{
	float dif = 0;
	for (int i = 0; i < w * h; i++)
	{
		// Since cublas matrix is tranposed, swap rows and columns when calculating residual
		int x = i % w;
		int y = i / w;
		int delta = cublas[(x * w) + y] - test[i];
		dif += delta;
	}

	return dif;
}

/// Tests all methods at a specified matrix size
void TestSize(cublasHandle_t &handle, int size, bool writeOutput)
{
	float *testA = new float[size * size];
	float *testB = new float[size * size];

	FILE *naive_fp; // Naive method file data
	FILE *opt0_fp; // Optimized method 0 file data
	FILE *opt1_fp; // Optimized method 0 file data
	FILE *opt2_fp; // Optimized method 0 file data

	if (writeOutput)
	{
		naive_fp = fopen("naive.csv", "a");
		opt0_fp = fopen("opt0.csv", "a");
		opt1_fp = fopen("opt1.csv", "a");
		opt2_fp = fopen("opt2.csv", "a");
	}

	// Randomize matrices
	RandomizeMatrix(testA, size);
	RandomizeMatrix(testB, size);
	printf("Testing size of %i x %i...\n", size, size);

	// Run cublas
	float totalTime0, kernelTime0;
	float* cublasResult = CublasMM(handle, testA, testB, size, &kernelTime0, &totalTime0);
	printf("Cublas MM | Total (ms) = %.3f | Kernel (ms) = %.3f\n", totalTime0, kernelTime0);

	// Run Naive MM
	const int threadsPerBlock = TILE_SIZE;
	float totalTime1, kernelTime1;
	float* result1 = NaiveMM(testA, testB, size, threadsPerBlock, &kernelTime1, &totalTime1);
	float residual1 = MatrixResidual(cublasResult, result1, size, size);
	printf("Naive MM (%i x %i t/b) | Total (ms) = %.3f | Kernel (ms) = %.3f | Cublas Residual = %.6f\n", threadsPerBlock, threadsPerBlock, totalTime1, kernelTime1, residual1);
	if (writeOutput && naive_fp)
	{
		fprintf(naive_fp, "%i, %.3f, %.3f\n", size, totalTime1, kernelTime1);
	}

	// Run Optimized MM (Version 0)
	float totalTime2, kernelTime2;
	float* result2 = OptimizedMM_0(testA, testB, size, &kernelTime2, &totalTime2);
	float residual2 = MatrixResidual(cublasResult, result2, size, size);
	printf("Optimized MM (Version 0) (%i x %i t/b) | Total (ms) = %.3f | Kernel (ms) = %.3f | Cublas Residual = %.6f\n", TILE_SIZE, TILE_SIZE, totalTime2, kernelTime2, residual2);
	if (writeOutput && opt0_fp)
	{
		fprintf(opt0_fp, "%i, %.3f, %.3f\n", size, totalTime2, kernelTime2);
	}

	// Run Optimized MM (Version 1)
	float totalTime3, kernelTime3;
	float* result3 = OptimizedMM_1(testA, testB, size, &kernelTime3, &totalTime3);
	float residual3 = MatrixResidual(cublasResult, result3, size, size);
	printf("Optimized MM (Verison 1) (%i x %i t/b) | Total (ms) = %.3f | Kernel (ms) = %.3f | Cublas Residual = %.6f\n", TILE_SIZE, TILE_SIZE, totalTime3, kernelTime3, residual3);
	if (writeOutput && opt1_fp)
	{
		fprintf(opt1_fp, "%i, %.3f, %.3f\n", size, totalTime3, kernelTime3);
	}

	// Run Optimized MM (Version 2)
	float totalTime4, kernelTime4;
	float* result4 = OptimizedMM_2(testA, testB, size, &kernelTime4, &totalTime4);
	float residual4 = MatrixResidual(cublasResult, result4, size, size);
	printf("Optimized MM (Verison 2) (%i x %i t/b) | Total (ms) = %.3f | Kernel (ms) = %.3f | Cublas Residual = %.6f\n", TILE_SIZE, TILE_SIZE, totalTime4, kernelTime4, residual4);
	if (writeOutput && opt2_fp)
	{
		fprintf(opt2_fp, "%i, %.3f, %.3f\n", size, totalTime4, kernelTime4);
	}

	// Delete input matrices
	delete[] testA;
	delete[] testB;

	// Delete results
	delete[] cublasResult;
	delete[] result1;
	delete[] result2;
	delete[] result3;
	delete[] result4;

	// Close file pointers
	if (writeOutput && naive_fp)
	{
		fclose(naive_fp);
	}

	if (writeOutput && opt0_fp)
	{
		fclose(opt0_fp);
	}

	if (writeOutput && opt0_fp)
	{
		fclose(opt1_fp);
	}

	if (writeOutput && opt0_fp)
	{
		fclose(opt2_fp);
	}
}

/// Program entry point
int main()
{
	srand(0);

	// Create cublas handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	int sizes[] { 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	for (int i = 0; i < 8; i++)
	{
		TestSize(handle, sizes[i], true);
		printf("\n");
	}

	// Destroy the cublas handle
	cublasDestroy(handle);
}