#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
//__constant__ float d_min, d_range;
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)


float* d_auxLuminanceForMin;
float* d_auxLuminanceForMax;
float* d_auxLuminance;
unsigned int* d_histogram;



template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}


__global__ void maxReduce(const float* const d_logLuminance, float* d_auxLuminanceForMax, int len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i < len) {

        d_auxLuminanceForMax[i] = d_logLuminance[i];
    }
    __syncthreads();

    for (unsigned int s = len / 2; s > 0; s /= 2) {
        if (i< s) {
           d_auxLuminanceForMax[i] = fmaxf(d_auxLuminanceForMax[i + s], d_auxLuminanceForMax[i]);
        }
        __syncthreads();

    }
}

__global__ void minReduce(const float* const d_logLuminance, float* d_auxLuminanceForMin, int len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {

        d_auxLuminanceForMin[i] = d_logLuminance[i];
    }
    __syncthreads();

    for (unsigned int s = len / 2; s > 0; s /= 2) {
        if (i< s) {
            d_auxLuminanceForMin[i] = fminf(d_auxLuminanceForMin[i + s], d_auxLuminanceForMin[i]);
        }
        __syncthreads();

    }
}

__global__ void histogram(const float* const d_logLuminance, unsigned int* histo, int len, int numBins,float min, float range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bin;
    
    if (i < len) {
        bin = (d_logLuminance[i] - min) / range * numBins;
        atomicAdd(&(histo[bin]), 1); 
    }
   
}


__global__ void exclusiveScan(unsigned int* histo, int len) {

    extern __shared__ float tempArray[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;
    int offset = 1, temp;
    int ai = threadId;
    int bi = threadId + len / 2;
    int i;

    
    tempArray[ai] = histo[id];
    tempArray[bi] = histo[id + len / 2];
    
    for (i = len >> 1; i > 0; i >>= 1)
    {
        __syncthreads();
        if (threadId < i)
        {
            ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            tempArray[bi] += tempArray[ai];
        }
        offset <<= 1;
    }
    
    if (threadId == 0)
        tempArray[len - 1] = 0;
   
    for (i = 1; i < len; i <<= 1) 
    {
        offset >>= 1;
        __syncthreads();
        if (threadId < i)
        {
            ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            temp = tempArray[ai];
            tempArray[ai] = tempArray[bi];
            tempArray[bi] += temp;
        }
    }
    __syncthreads();
    histo[id] = tempArray[threadId];
    histo[id + len / 2] = tempArray[threadId + len / 2];
   
   

}



void calculate_cdf(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    const dim3 blockSize(512, 1 , 1);
    const dim3 gridSize((numCols*numRows - 1) / blockSize.x + 1, 1, 1);

    const dim3 gridSizeReduce(((numCols * numRows)/2 - 1) / blockSize.x + 1, 1, 1);
  /* TODO
    1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance 

    */
    
    
    int size = numRows * numCols;

    cudaMalloc(&d_auxLuminance, size * sizeof(float));
    //cudaMalloc(&d_auxLuminanceForMax,size*sizeof(float));
    //cudaMalloc(&d_auxLuminanceForMin, size * sizeof(float));
    cudaMalloc(&d_histogram, numBins* sizeof(unsigned int));
    
    //llamada al kernel
    maxReduce<<<gridSizeReduce , blockSize >>> (d_logLuminance, d_auxLuminance, size);
    cudaMemcpy(&max_logLum, d_auxLuminance, sizeof(float), cudaMemcpyDeviceToHost);

    minReduce <<<gridSizeReduce, blockSize >>> (d_logLuminance, d_auxLuminance, size);
    cudaMemcpy(&min_logLum, d_auxLuminance, sizeof(float), cudaMemcpyDeviceToHost);
    

    //cudaMemcpyToSymbol(&d_min, d_auxLuminanceForMin, sizeof(float),cudaMemcpyDeviceToDevice);

   /*cudaFree(d_auxLuminanceForMax);
    cudaFree(d_auxLuminanceForMin);*/
    cudaFree(d_auxLuminance);

    float range = max_logLum - min_logLum;
    
    std::cout << max_logLum << std::endl;
    std::cout << min_logLum << std::endl;
    std::cout << range << std::endl;

   // cudaMemcpyToSymbol(&d_range,&range , sizeof(float), cudaMemcpyHostToDevice);

    histogram <<<gridSize, blockSize >>> (d_logLuminance, d_histogram, size, numBins,min_logLum,range);

    std::cout << numBins << std::endl;
    //Para que la llamada al kernel funcione correctamente, el resultado de numBins/2 debe ser menor o igual a 1024
    exclusiveScan <<<1, (numBins/2),(numBins*2 * sizeof(unsigned int)) >>> (d_histogram, numBins);

    checkCudaErrors(cudaMemcpy(d_cdf, d_histogram, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToDevice));

    cudaFree(d_histogram);
    /*
	2) Obtener el rango a representar
	3) Generar un histograma de todos los valores del canal logLuminance usando la formula 
	bin = (Lum [i] - lumMin) / lumRange * numBins
	4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf) 
	de los valores de luminancia. Se debe almacenar en el puntero c_cdf
  */    
}


