#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

//__constant__ float d_min, d_range;



float* d_auxLuminanceForMin;
float* d_auxLuminanceForMax;
unsigned int* d_histogram;


__global__ void maxAndMinReduce(const float* const d_logLuminance,float& min_logLum, float& max_logLum, float* d_auxLuminanceForMin, float* d_auxLuminanceForMax, int len) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // RELLENAR CON EL VALOR IDENTIDAD
    if (i >= len) {
        d_auxLuminanceForMin[threadIdx.x] = 0.f;
        d_auxLuminanceForMax[threadIdx.x] = 0.f;
    }
    else {
        d_auxLuminanceForMin[threadIdx.x] = d_logLuminance[i];
        d_auxLuminanceForMax[threadIdx.x] = d_logLuminance[i];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (i < s) {
            
            if (d_auxLuminanceForMin[threadIdx.x] > d_auxLuminanceForMin[threadIdx.x + s]) {
                d_auxLuminanceForMin[threadIdx.x] = d_auxLuminanceForMin[threadIdx.x + s];
            }

            if (d_auxLuminanceForMax[threadIdx.x] <= d_auxLuminanceForMax[threadIdx.x + s]) {

                d_auxLuminanceForMax[threadIdx.x] = d_auxLuminanceForMax[threadIdx.x + s];
            }

        }
        __syncthreads();
    }
}

__global__ void histogram(const float* const d_logLuminance, unsigned int* histo, int len, int numBins,float min, float range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bin;
    // All threads handle blockDim.x * gridDim.x consecutive elements
    if (i < len) {
        bin = (d_logLuminance[i] - min) / range * numBins;
        atomicAdd(&(histo[bin]), 1); //Varios threads podrían intentar incrementar el mismo valor a la vez
    }
}


__global__ void exclusiveScan(unsigned int* histo, int len) {

    extern  __shared__ float tempArray[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;
    int offset = 1, temp;
    int ai = threadId;
    int bi = threadId + len / 2;
    int i;
    //assign the shared memory
    tempArray[ai] = histo[id];
    tempArray[bi] = histo[id + len / 2];
    //up tree
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
    //put the last one 0
    if (threadId == 0)
        tempArray[len - 1] = 0;
    //down tree
    for (i = 1; i < len; i <<= 1) // traverse down tree & build scan
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
    const dim3 blockSize(32, 32, 1);
    const dim3 gridSize((numCols - 1) / 32 + 1, (numRows - 1) / 32 + 1, 1);

  /* TODO
    1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance 

    */
    
    
    int size = numRows * numCols;

    cudaMalloc(&d_auxLuminanceForMax,size*sizeof(float));
    cudaMalloc(&d_auxLuminanceForMin, size * sizeof(float));
    cudaMalloc(&d_histogram, numBins* sizeof(unsigned int));
    

    //llamada al kernel
    maxAndMinReduce <<<gridSize,blockSize >>> (d_logLuminance,min_logLum, max_logLum, d_auxLuminanceForMin, d_auxLuminanceForMax, size);
    //


    cudaMemcpy(&min_logLum, d_auxLuminanceForMin, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_logLum, d_auxLuminanceForMax, sizeof(float), cudaMemcpyDeviceToHost);

    //cudaMemcpyToSymbol(&d_min, d_auxLuminanceForMin, sizeof(float),cudaMemcpyDeviceToDevice);

   /* cudaFree(d_auxLuminanceForMax);
    cudaFree(d_auxLuminanceForMin);*/

    float range = max_logLum - min_logLum;
    

   // cudaMemcpyToSymbol(&d_range,&range , sizeof(float), cudaMemcpyHostToDevice);

    histogram <<<gridSize, blockSize >>> (d_logLuminance, d_histogram, size, numBins,min_logLum,range);

    exclusiveScan <<<gridSize, blockSize,numBins*2 >>> (d_histogram, numBins);

    cudaMemcpy(d_cdf, d_histogram, sizeof( unsigned int) * numBins, cudaMemcpyDeviceToDevice);
    /*
	2) Obtener el rango a representar
	3) Generar un histograma de todos los valores del canal logLuminance usando la formula 
	bin = (Lum [i] - lumMin) / lumRange * numBins
	4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf) 
	de los valores de luminancia. Se debe almacenar en el puntero c_cdf
  */    
}


