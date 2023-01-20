#include <iostream>
#include <math.h>
#include <chrono>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y, float *z)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    z[i] = x[i] * y[i] * y[i];
  }
}

__global__
void step(int n , float *x, float *v, float *a, float *next_x, float *next_v)
{
  constexpr float ts = 0.1f;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    next_x[i] = x[i] + ts * v[i] + (ts * ts) / 2.0 * a[i];
    next_x[i] = x[i] + ts * v[i];
    next_v[i] = v[i] + ts * a[i];
  }
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *z, *next_x, *next_v;

  
  // Allocate Unified Memory – accessible from CPU or GPU
  gpuErrchk(cudaMallocManaged(&x, N*sizeof(float)));
  gpuErrchk(cudaMallocManaged(&y, N*sizeof(float)));
  gpuErrchk(cudaMallocManaged(&z, N*sizeof(float)));
  gpuErrchk(cudaMallocManaged(&next_x, N*sizeof(float)));
  gpuErrchk(cudaMallocManaged(&next_v, N*sizeof(float)));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
    z[i] = 1.0f;
    next_x[i] = 1.0f;
    next_v[i] = 1.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Run kernel on 1M elements on the GPU
  int blockSize = 512;
  int numBlocks = (N + blockSize - 1) / blockSize;
  step<<<numBlocks, blockSize>>>(N, x, y, z, next_x, next_v);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
