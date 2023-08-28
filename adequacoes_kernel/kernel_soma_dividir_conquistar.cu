#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

_global_ int sumDivideConquer(int* a, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (size == 1)
        return array[i];
    int half = size / 2;
    int left[half], right[size - half];

    for (int i = 0; i < size; i++) {
        if (i < half)
            left[i] = array[i];
        else
            right[i - half] = array[i];
    }

    int sumLeft, sumRight;
    sumLeft = sumDivideConquer(left, half);
    sumRight = sumDivideConquer(right, size - half);

    return sumLeft + sumRight;
}

int main() {
    cudaDeviceReset();
    int* a, * d_a;  
    int n = 2048, size = n * sizeof(int), sumRes;

    a = (int*) malloc(size);

    cudaMalloc((void**) &d_a, size);

    for (int i = 0; i < n; i++) {
        a[i] = i;

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    int threads = 1024;

    int blocks = (n + threads - 1) / threads;

    sumRes = vecAdd << <blocks, threads> >> (d_a, n);

    cudaDeviceSynchronize();

    printf("Sum result: %d\n", sumRes);

    cudaFree(d_a);  
    free(a);

    return 0;
}