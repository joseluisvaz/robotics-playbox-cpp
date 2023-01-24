#include "common/book.h"
#include "common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
  float r;
  float i;
  __device__ cuComplex(float a, float b) : r(a), i(b) {}
  __device__ float magnitude2(void)
  {
    return r * r + i * i;
  }
  __device__ cuComplex operator*(const cuComplex &a)
  {
    return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
  }
  __device__ cuComplex operator+(const cuComplex &a)
  {
    return cuComplex(r + a.r, i + a.i);
  }
};

__device__ int julia(int x, int y)
{
  const float scale = 1.5;
  // Convert int coordinates to float coordinates
  float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
  float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

  // Iterate to compute the julia set
  cuComplex c(-0.8, 0.156);
  cuComplex a(jx, jy);
  int i = 0;
  for (i = 0; i < 200; i++)
  {
    a = a * a + c;
    if (a.magnitude2() > 1000)
    {
      return 0;
    }
  }
  return 1;
}

__global__ void kernel(unsigned char *ptr)
{
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * gridDim.x;

  // now calculate the value at that position
  int juliaValue = julia(x, y);

  // We have 4 different channels
  ptr[offset * 4 + 0] = 0;
  ptr[offset * 4 + 1] = 255 * juliaValue;
  ptr[offset * 4 + 2] = 0;
  ptr[offset * 4 + 3] = 200;
}

int main(void)
{
  CPUBitmap bitmap(DIM, DIM);
  unsigned char *dev_bitmap;
  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

  dim3 grid(DIM, DIM);
  kernel<<<grid, 1>>>(dev_bitmap);
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
  bitmap.display_and_exit();
  HANDLE_ERROR(cudaFree(dev_bitmap));
}