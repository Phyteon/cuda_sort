
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define MAX_THREADS_PER_BLOCK 1024  // According to CUDA article https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
#define MAX_BLOCK_COUNT_PER_GRID_DIM 65535  // https://forums.developer.nvidia.com/t/how-determine-max-number-of-blocks-and-threads-for-a-gpu/68439

cudaError_t oddEvenTranspositionSortWithCuda(int* sorted_array, const int* input_array, unsigned int size);

__global__ void sortKernel(int *sorted_array, const int *input_array, unsigned int size)
{
    // Prepare a shared buffer, long enough to hold 2 sets of data, each set from different stage of processing
    __shared__ int local_data_copy[MAX_THREADS_PER_BLOCK];
    __shared__ int exec_flag;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int local_index = threadIdx.x;
    int compare_flag = 0;
    int temp;
    if (local_index == size - 1)
        exec_flag = 1; // To ensure that the operation is atomic, only one thread writes the global flag
    bool stage = true;
    // Make a local copy in shared memory for manipulation
    if (local_index < size)
    {
        local_data_copy[local_index] = input_array[global_index];
    }
    // Make sure all data has been copied correctly - wait for all threads to arrive at this point
    __syncthreads();
    while (exec_flag == 1)
    {
        if (stage) {
            if ((local_index % 2 == 0) && (local_index < size - 1)) {
                if (local_data_copy[local_index] > local_data_copy[local_index + 1]) {
                    temp = local_data_copy[local_index];
                    local_data_copy[local_index] = local_data_copy[local_index + 1];
                    local_data_copy[local_index + 1] = temp;
                }
            }
        }
        else {
            if ((local_index % 2 != 0) && (local_index < size - 1)) {
                if (local_data_copy[local_index] > local_data_copy[local_index + 1]) {
                    temp = local_data_copy[local_index];
                    local_data_copy[local_index] = local_data_copy[local_index + 1];
                    local_data_copy[local_index + 1] = temp;
                }
            }
        }
        stage = !stage;
        __syncthreads();
        // One thread constantly monitors changes in the sorted array. If there are no more changes compared to previous iteration, sorting is done.
        if (local_index == size - 1) {
            for (int idx = 0; idx < size; idx++) {
                if (local_data_copy[idx] != local_data_copy[idx + MAX_THREADS_PER_BLOCK])
                    compare_flag += 1;
            }
            if (compare_flag != 0)
                compare_flag = 0;
            else {
                exec_flag = 0;
            }
        }
        __syncthreads();
    }
    // Copy the outcome to output array
    if (local_index < size)
        sorted_array[global_index] = local_data_copy[local_index];

}

int main()
{
    const int arraySize = 10;
    const int input_array[arraySize] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    int sorted_array[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = oddEvenTranspositionSortWithCuda(sorted_array, input_array, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "oddEvenTranspositionSortWithCuda failed!");
        return 1;
    }

    printf("Sorted entries:\n");
    for (int i = 0; i < arraySize; i++) {
        printf("%d\n", sorted_array[i]);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t oddEvenTranspositionSortWithCuda(int* sorted_array, const int* input_array, unsigned int size)
{
    int *dev_sorted_array = 0;
    int *dev_input_array = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_sorted_array, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_input_array, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input_array, input_array, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    sortKernel<<<1, size>>>(dev_sorted_array, dev_input_array, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(sorted_array, dev_sorted_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_sorted_array);
    cudaFree(dev_input_array);
    
    return cudaStatus;
}
