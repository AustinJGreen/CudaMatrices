#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <algorithm>
#include <stdio.h>

constexpr auto TILE_SIZE = 2;

__global__ void NativeMMKernel(float *a, float *b, float *c, int size)
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

__global__ void OptimizedMMKernel(float *a, float *b, float *c, int size)
{
	__shared__ float sharedA[TILE_SIZE * TILE_SIZE];
	__shared__ float sharedB[TILE_SIZE * TILE_SIZE];

	// Load shared memory
	// Each block loads a shared block from A and B so the partial sums can
	// be done at the same time

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0;
	int tilesPerGrid = size / blockDim.x;
	for (int i = 0; i < tilesPerGrid; i++)
	{
		// Each thread loads element into A and B
		sharedA[threadIdx.y * TILE_SIZE + threadIdx.x] = a[(y * size) + (i * TILE_SIZE) + threadIdx.x];
		sharedB[threadIdx.y * TILE_SIZE + threadIdx.x] = b[(i * TILE_SIZE * size) + (threadIdx.y * size) + x];

		// Wait for all threads to load each section of the shared matrix
		__syncthreads();

		for (int j = 0; j < TILE_SIZE; j++)
		{
			sum += sharedA[threadIdx.y * TILE_SIZE + j] * sharedB[j * TILE_SIZE + threadIdx.x];
		}

		// Wait for all threads to compute their partial sum from the shared matrices before loading the next
		__syncthreads();
	}

	c[y * size + x] = sum;
}

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

void CheckCudaCall(cudaError_t callResult, const char *message)
{
	if (callResult != cudaSuccess)
	{
		fprintf(stderr, message);
	}
}

float* NativeMM(float *a, float *b, int size, bool timeMemory, float *elapsed)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Create events
	cudaEvent_t startEvent, stopEvent;
	CheckCudaCall(cudaEventCreate(&startEvent), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&stopEvent), "cudaEventCreate failed");

	// Allocate device matrices
 	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// If we're timing memory communication, start the timer here
	if (timeMemory)
	{
		cudaEventRecord(startEvent, 0);
	}

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	// If we're NOT timing memory communication, start the timer here
	if (!timeMemory)
	{
		cudaEventRecord(startEvent, 0);
	}

	// Call kernel
	dim3 blockSize(size / TILE_SIZE, size / TILE_SIZE);
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);	
	NativeMMKernel<<<blockSize, threadsPerBlock>>>(devA, devB, devC, size);

	// If we're NOT timing memory communication, stop the timer here
	if (!timeMemory)
	{
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
	}

	// Copy result back
	cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost);

	// If we're timing memory communication, stop the timer here
	if (timeMemory)
	{
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
	}

	// Calculate elapsed time
	cudaEventElapsedTime(elapsed, startEvent, stopEvent);

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	// Cleanup host memory
	delete[] a;
	delete[] b;
	
	return result;
}

float* OptimizedMM(float *a, float *b, int size, bool timeMemory, float *elapsed)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Create events
	cudaEvent_t startEvent, stopEvent;
	CheckCudaCall(cudaEventCreate(&startEvent), "cudaEventCreate failed");
	CheckCudaCall(cudaEventCreate(&stopEvent), "cudaEventCreate failed");

	// Allocate device matrices
	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// If we're timing memory communication, start the timer here
	if (timeMemory)
	{
		cudaEventRecord(startEvent, 0);
	}

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	// If we're NOT timing memory communication, start the timer here
	if (!timeMemory)
	{
		cudaEventRecord(startEvent, 0);
	}

	// Call kernel
	dim3 blockSize(size / TILE_SIZE, size / TILE_SIZE);
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

	// TODO: Pass in Shared memory size (http://courses.cms.caltech.edu/cs179/2019_lectures/cs179_2019_lec10.pdf)
	OptimizedMMKernel<<<blockSize, threadsPerBlock>>>(devA, devB, devC, size);

	// If we're NOT timing memory communication, stop the timer here
	if (!timeMemory)
	{
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
	}

	// Copy result back
	cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost);

	// If we're timing memory communication, stop the timer here
	if (timeMemory)
	{
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent);
	}

	// Calculate elapsed time
	cudaEventElapsedTime(elapsed, startEvent, stopEvent);

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return result;
}

float* CublasMM(cublasHandle_t &handle, float *a, float *b, int size)
{
	const int matrixSizeBytes = size * size * sizeof(float);
	float *result = new float[size * size];
	float *devA, *devB, *devC;

	// Allocate device matrices
	CheckCudaCall(cudaMalloc((void **)&devA, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devB, matrixSizeBytes), "cudaMalloc failed");
	CheckCudaCall(cudaMalloc((void **)&devC, matrixSizeBytes), "cudaMalloc failed");

	// Copy over to host
	CheckCudaCall(cudaMemcpy(devA, a, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");
	CheckCudaCall(cudaMemcpy(devB, b, matrixSizeBytes, cudaMemcpyHostToDevice), "cudaMemcpy failed");

	const float alp = 1;
	const float bet = 0;
	const float *alpha = &alp;
	const float *beta = &bet;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, alpha, devA, size, devB, size, beta, devC, size);

	// Copy result back
	CheckCudaCall(cudaMemcpy(result, devC, matrixSizeBytes, cudaMemcpyDeviceToHost), "cudaMemcpy failed");

	// Cleanup device memory
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	// Return result
	return result;
}

void RandomizeMatrix(float *mat, int size)
{
	for (int i = 0; i < size * size; i++) 
	{
		mat[i] = (float)(rand() % 10);
	}
}

void SetMatrixAsSequential(float *mat, int size)
{
	for (int i = 0; i < size * size; i++) 
	{
		mat[i] = (float)(i + 1);
	}
}

int main()
{
	srand(0);

	// Create cublas handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	const int size = 4;

	float *testA = new float[size * size];
	float *testB = new float[size * size];

	SetMatrixAsSequential(testA, size);
	SetMatrixAsSequential(testB, size);

	//float elapsed = 0;
	//float* result1 = NativeMM(testA, testB, size, false, &elapsed);
	//printf("Elapsed = %.6fms\n", elapsed);

	float* result2 = CublasMM(handle, testA, testB, size);

	PrintMatrix(result2, size, size);

	// Delete results
	delete[] result1;
	delete[] result2;

	// Destroy the cublas handle
	cublasDestroy(handle);
}