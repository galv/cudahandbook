#include "assert.h"
#include <cstdio>
#include "chTimer.h"

#define CUDART_CHECK( fn ) do { \
        cudaError_t status =  (fn); \
        if ( status != cudaSuccess ) { \
            fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t" \
                "%s returned 0x%x (%s)\n", \
                __LINE__, __FILE__, #fn, status, cudaGetErrorString(status) ); \
        } \
    } while (0);

template <typename Real>
__device__ inline void atomic_add(Real* address, Real value) {
  Real old = value;
  Real ret = atomicExch(address, 0.0f);
  Real new_old = ret + old;
  while ((old = atomicExch(address, new_old)) != 0.0f) {
    new_old = atomicExch(address, 0.0f);
    new_old += old;
  }
}

template <typename Real>
__global__ void count_threads_library(Real *answer) {
  atomicAdd(answer, Real(1));
}

template <typename Real>
__global__ void count_threads_custom(Real *answer) {
  atomic_add(answer, Real(1));
}


int main() {
  size_t buffer_size = 1;
  size_t num_threads = 256;
  size_t num_blocks = 512;
  size_t num_iters = 10;
  float *buffer_dev;
  float *buffer_host = (float *) malloc(buffer_size * sizeof(float));
  CUDART_CHECK(cudaMalloc(&buffer_dev, buffer_size * sizeof(float)));
  CUDART_CHECK(cudaMemset(buffer_dev, 0, buffer_size * sizeof(float)));
  count_threads_custom<<<1,num_threads>>>(buffer_dev);
  CUDART_CHECK(cudaMemcpy(buffer_host, buffer_dev, buffer_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
  printf("should be %d: %f\n", num_threads, *buffer_host);
  assert(*buffer_host == num_threads);

  chTimerTimestamp custom_start, custom_stop;
  chTimerGetTime(&custom_start);
  for(size_t i = 0; i < num_iters; i++) {
    count_threads_custom<<<num_blocks,num_threads>>>(buffer_dev);
  }
  cudaDeviceSynchronize();
  chTimerGetTime(&custom_stop);
  double custom_time_seconds =
    (chTimerElapsedTime(&custom_start, &custom_stop)) /
    ((double) num_iters);
  printf("Time elapsed for custom implementation: %f seconds.\n",
         custom_time_seconds);

  chTimerTimestamp library_start, library_stop;
  chTimerGetTime(&library_start);
  for(size_t i = 0; i < num_iters; i++) {
    count_threads_library<<<num_blocks,num_threads>>>(buffer_dev);
  }
  cudaDeviceSynchronize();
  chTimerGetTime(&library_stop);
  double library_time_seconds =
    (chTimerElapsedTime(&library_start, &library_stop)) /
    ((double) num_iters);
  printf("Time elapsed for library implementation: %f seconds.\n",
         library_time_seconds);
  CUDART_CHECK(cudaFree(buffer_dev));
  return 0;
}