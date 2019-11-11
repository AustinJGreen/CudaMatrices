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
constexpr auto TILE_SIZE = 32;

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

/// Shared memory matrix multiplication CUDA kernel
__global__ void OptimizedMMKernel(float *a, float *b, float *c, int size)
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
float* OptimizedMM(float *a, float *b, int size, float *kernelTime, float* totalTime)
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
	OptimizedMMKernel <<<blockSize, threadsPerBlock >>>(devA, devB, devC, size);

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
	FILE *opt_fp; // Optimized method file data

	if (writeOutput)
	{
		naive_fp = fopen("naive.csv", "a");
		opt_fp = fopen("opt.csv", "a");
	}

	// Randomize matrices
	RandomizeMatrix(testA, size);
	RandomizeMatrix(testB, size);
	printf("Testing size of %i x %i...\n", size, size);

	// Run cublas
	float totalTime0, kernelTime0;
	float* result0 = CublasMM(handle, testA, testB, size, &kernelTime0, &totalTime0);
	printf("Cublas MM | Total (ms) = %.3f | Kernel (ms) = %.3f\n", totalTime0, kernelTime0);

	// Run Naive MM
	const int threadsPerBlock = 32;
	float totalTime1, kernelTime1;
	float* result1 = NaiveMM(testA, testB, size, threadsPerBlock, &kernelTime1, &totalTime1);
	float residual1 = MatrixResidual(result0, result1, size, size);
	printf("Naive MM (%i x %i t/b) | Total (ms) = %.3f | Kernel (ms) = %.3f | Cublas Residual = %.6f\n", threadsPerBlock, threadsPerBlock, totalTime1, kernelTime1, residual1);
	if (writeOutput && naive_fp)
	{
		fprintf(naive_fp, "%i, %.3f, %.3f\n", size, totalTime1, kernelTime1);
	}

	// Run Optimized MM
	float totalTime2, kernelTime2;
	float* result2 = OptimizedMM(testA, testB, size, &kernelTime2, &totalTime2);
	float residual2 = MatrixResidual(result0, result2, size, size);
	printf("Optimized MM (%i x %i t/b) | Total (ms) = %.3f | Kernel (ms) = %.3f | Cublas Residual = %.6f\n", TILE_SIZE, TILE_SIZE, totalTime2, kernelTime2, residual2);
	if (writeOutput && opt_fp)
	{
		fprintf(opt_fp, "%i, %.3f, %.3f\n", size, totalTime2, kernelTime2);
	}

	// Delete input matrices
	delete[] testA;
	delete[] testB;

	// Delete results
	delete[] result0;
	delete[] result1;
	delete[] result2;

	// Close file pointers
	if (writeOutput && naive_fp)
	{
		fclose(naive_fp);
	}

	if (writeOutput && opt_fp)
	{
		fclose(opt_fp);
	}
}

/// Program entry point
int main()
{
	srand(0);

	// Create cublas handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	int sizes[8]{ 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	for (int i = 0; i < 8; i++)
	{
		TestSize(handle, sizes[i], false);
		printf("\n");
	}

	// Destroy the cublas handle
	cublasDestroy(handle);
}