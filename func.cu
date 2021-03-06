//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__
void box_filter(const unsigned char* __restrict__ inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* __restrict__ filter, const int filterWidth)
{
  // TODO: 
  // NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
  //
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;
  // NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
  // los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.
    
    // Filtro interpretando los bordes como píxeles negros
    const int offset = filterWidth / 2;

    float outputValue;

    for (int i = -offset; i <= offset; i++)
    {
        for (int j = -offset; j <= offset; j++)
        {
            int idX = thread_2D_pos.x + i;
            int idY = thread_2D_pos.y + j;
            if (idX >= 0 && idX < numCols && idY >= 0 && idY < numRows) {
                int pixelId = (idY)*numCols + (idX);
                outputValue += inputChannel[pixelId] * filter[(j + offset) * filterWidth + (i + offset)];
            }           
        }
    }
    if (outputValue < 0) {
        outputValue = 0;
    }
    else if (outputValue > 255) {
        outputValue = 255;
    }
    outputChannel[thread_1D_pos] = outputValue;
    
    // Filtro evitando los bordes
    // Si el vecino cuadra fuera de la imagen no se aplica
    /*if (thread_2D_pos.x < offset || thread_2D_pos.x >= numCols - offset || 
        thread_2D_pos.y < offset || thread_2D_pos.y >= numRows - offset)
        outputChannel[thread_1D_pos] = inputChannel[thread_1D_pos];
    else {
        float outputValue;

        for (int i = -offset; i <= offset; i++)
        {
            for (int j = -offset; j <= offset; j++)
            {
                int idX = thread_2D_pos.x + i;
                int idY = thread_2D_pos.y + j;
                int pixelId = (idY)*numCols + (idX);
                outputValue += inputChannel[pixelId] * filter[(j + offset) * filterWidth + (i + offset)];
            }
        }
        if (outputValue < 0) {
            outputValue = 0;
        }
        else if (outputValue > 255) {
            outputValue = 255;
        }
        outputChannel[thread_1D_pos] = outputValue;
    }*/
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* __restrict__ inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO: 
  // NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
  //

    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
    greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
    blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* __restrict__ redChannel,
                       const unsigned char* __restrict__ greenChannel,
                       const unsigned char* __restrict__ blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;


void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
  // Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)
  
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}


void create_filter(float **h_filter, int *filterWidth){

  const int KernelWidth = 5; //OJO CON EL TAMAÑO DEL FILTRO//
  *filterWidth = KernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[KernelWidth * KernelWidth];
  
  
  ////Filtro gaussiano: blur
  //const float KernelSigma = 2.;

  //float filterSum = 0.f; //for normalization

  //for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
  //  for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
  //    float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
  //    (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
  //    filterSum += filterValue;
  //  }
  //}

  //float normalizationFactor = 1.f / filterSum;

  //for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
  //  for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
  //    (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
  //  }
  //}
  

  //Laplaciano 5x5
  (*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
  (*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
  (*h_filter)[10] = -1.;(*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
  (*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
  (*h_filter)[20] = 1.;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;


  

  //TODO: crear los filtros segun necesidad
  //NOTA: cuidado al establecer el tamaño del filtro a utilizar

  //const int KernelWidth = 3;
  //*filterWidth = KernelWidth;

  ////create and fill the filter we will convolve with
  //*h_filter = new float[KernelWidth * KernelWidth];
 
  //// Detección de línea horizontal 3x3
  //(*h_filter)[0] = -1.; (*h_filter)[1] = -1.; (*h_filter)[2] = -1.;
  //(*h_filter)[3] = 2.;  (*h_filter)[4] = 2.;  (*h_filter)[5] = 2.;
  //(*h_filter)[6] = -1.; (*h_filter)[7] = -1.; (*h_filter)[8] = -1.;

  //// Nitidez 3x3
  /*(*h_filter)[0] = 0; (*h_filter)[1] = -0.25; (*h_filter)[2] = 0;
  (*h_filter)[3] = -0.25;  (*h_filter)[4] = 2.;  (*h_filter)[5] = -0.25;
  (*h_filter)[6] = 0; (*h_filter)[7] = -0.25; (*h_filter)[8] = 0;*/

  //// Nitidez_2 3x3
  /*(*h_filter)[0] = -0.25; (*h_filter)[1] = -0.25; (*h_filter)[2] = -0.25;
  (*h_filter)[3] = -0.25;  (*h_filter)[4] = 3.;  (*h_filter)[5] = -0.25;
  (*h_filter)[6] = -0.25; (*h_filter)[7] = -0.25; (*h_filter)[8] = -0.25;*/

  //// Suavizado 3x3
  /*(*h_filter)[0] = 0.1; (*h_filter)[1] = 0.1; (*h_filter)[2] = 0.1;
  (*h_filter)[3] = 0.1;  (*h_filter)[4] = 0.1;  (*h_filter)[5] = 0.1;
  (*h_filter)[6] = 0.1; (*h_filter)[7] = 0.1; (*h_filter)[8] = 0.1;*/

}


void convolution(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redFiltered, 
                        unsigned char *d_greenFiltered, 
                        unsigned char *d_blueFiltered,
                        const int filterWidth)
{

  
  const dim3 blockSize(16, 16, 1);
  const dim3 gridSize((numCols - 1)/blockSize.x + 1, (numRows - 1)/blockSize.y + 1, 1);

  //TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
  separateChannels <<<gridSize, blockSize>>> (d_inputImageRGBA, numRows, numCols, 
      d_red, d_green, d_blue);

  //TODO: Ejecutar convolución. Una por canal
  box_filter <<<gridSize, blockSize>>> (d_red, d_redFiltered, numRows, numCols, d_filter, filterWidth);
  box_filter <<<gridSize, blockSize>>> (d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
  box_filter <<<gridSize, blockSize>>> (d_blue, d_blueFiltered, numRows, numCols, d_filter, filterWidth);

  // Recombining the results. 
  recombineChannels<<<gridSize, blockSize>>>(d_redFiltered,
                                             d_greenFiltered,
                                             d_blueFiltered,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
