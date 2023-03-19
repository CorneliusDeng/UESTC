- **Thread** - Distributed by the CUDA runtime (identified by threadIdx)
- **Warp** - A scheduling unit of up to 32 threads 
- **Block** - A user defined group of 1 to 512 threads (identified by blockIdx)
- **Grid** - A group of one or more blocks. A grid is created for each CUDA kernel function.



- Vector Addition

```c
#include <cuda.h>
// compute vector sum C = A + B
void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{
  int size = n * sizeof(float);
  float *d_A, *d_B, *d_C;
  // part 1 : Allocate device memory for A, B and C, copy A and B to device memory
  // part 2 : Kernel launch code, the device performs the actual vector addition
  // part 3 : copy C from the device memory
}
```



- A PictureKernel

```c
__global__ void pictureKernel(float* d_Pin, float* d_Pout, int height, int width)
{
  // calculate the row of the d_Pin and d_Pout element
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
    
  // calculate the column of the d_Pin and d_Pout element
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
  // each thread computes one element of d_Pout if in range
  if ((Row < height) && (col < width))
  {
    d_Pout[Row*width+Col] = 2.0 * d_Pin[Row*width+Col];
  }
}
```



- RGB To Grayscale Conversion Code

```c
#define CHANNELS 3
// We have 3 channels corresponding to RGB, The input image is encoded as unsigned characters [0,255]
__global__ void colorConvert(unsigned char* grayImage, unsigned char * rgbImage, int width, int height)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (x < width && y < height)
  {
    // get 1D coordinate for the grayscale image
    int grayOffset = y * width + x;
    // one can think of the RGB image having CHANNEL times columns than the gray scale image
    int rgbOffset = grayOffset * CHANNELS;
    unsigned char r = rgbImage[rgbOffset]; // red value for pixel
    unsigned char g = rgbImage[rgbOffset + 1]; // green value for pixel
    unsigned char b = rgbImage[rgbOffset + 2]; // blue value for pixel
    // perform the rescaling and store it. We multiply by floating point constants
    grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}
```



- Image Blur As a 2D Kernel

```c
__global__ void blurKernel(unsigned char* in, unsigned char * out, int w, int h)
{
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (Col < w && Row < h)
  {
    int pixVal = 0; 
    int pixels = 0;
    // Get the average of the surrounding 2xBLUR_SIZE
  }

}
```



- A Basic Matrix Multiplication

```c
__global__ void MatrixMulKernel(float* M, float* N, float * P, int Width)
{
	// Calculate the row index of the P element and M
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Calculate the column index of P and N
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ((Row < Width) && (Col < Width))
  {
    float Pvalue = 0;
    // each thread computes one element of the block sub-matrix 
    for (int k = 0; k < Width; ++k)
    {
      Pvalue += M[Row * Width + k] * N[K*Width+Col];
    }
    P[Row * Width + Col] = Pvalue;
  }
	
}
```

