/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include "utils.h"

//#define TRACE
#undef TRACE

#if defined(TRACE)
	#define DUMP(a, b, c, d, e) dump(a, b, c, d, e)
#else
	#define DUMP(a, b, c, d, e)
#endif

const size_t NUM_OPS = 2;
typedef float (*reduceOp_t)(float, float);
__device__ reduceOp_t d_pMin = min;
__device__ reduceOp_t d_pMax = max;
typedef float const & (*reduceOp_std_t) (float const &, float const &);

const size_t BLOCK_SIZE = 256;

__global__ void reduce(const float* const d_input, size_t inputSize, int neutralIdx, reduceOp_t* reduceOps, int opIdx, float *d_output)
{
	/*extern*/ __shared__ float working[BLOCK_SIZE];

	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int idx = offset + tid;

	working[tid] = idx < inputSize ? d_input[idx] : d_input[neutralIdx];
	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride <<= 1)
	{
		int i = 2 * stride * tid;

		if (i < blockDim.x)
			working[i] = reduceOps[opIdx](working[i], working[i+stride]);

		__syncthreads();
	}

	if (0 == tid)
		d_output[blockIdx.x] = working[0];
}


__global__ void histo(const float* const d_input, size_t inputSize, float minVal, float range, size_t numBins, unsigned int* d_output)
{
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int idx = offset + tid;

	if (idx < inputSize)
	{
		float lum = d_input[idx];
		unsigned int bin = min( (lum - minVal) / range * numBins, float(numBins -1));
		//d_output[bin]++;
		atomicAdd(&d_output[bin], 1);
	}
}

__host__ __device__ void dump(int thid, size_t n, unsigned int *buffer, int pout, int iter) {
	if (thid == 0) {
		printf("%i\n", iter++);
		for (size_t i = 0; i < n; i++) printf("%u ", buffer[pout * n + i]);
		printf("\n");
	}

#ifdef __CUDA_ARCH__
	__syncthreads();
#endif //__CUDA_ARCH__
}

__global__ void exclusivePrefixSum(const unsigned int* const g_idata, unsigned int *g_odata, size_t n)
{
	 extern __shared__ unsigned int temp[]; // allocated on invocation

	 int thid = threadIdx.x;
	 int pout = 0, pin = 1;
	 int iter = 0;

	 // load input into shared memory.
	 // This is exclusive scan, so shift right by one and set first elt to 0
	  /*temp[pin*n + thid] = */ temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;

	 __syncthreads();
	 DUMP(thid, n, temp, pout, iter);

	 for (int offset = 1; offset < n; offset *= 2)
	 {
		 iter++;

		 pout = 1 - pout; // swap double buffer indices
		 pin = 1 - pout;

		 if (thid >= offset)
			 //temp[pout*n+thid] += temp[pin*n+thid - offset];
			 temp[pout*n+thid] = temp[pin*n+thid] + temp[pin*n+thid - offset];
		 else
			 temp[pout*n+thid] = temp[pin*n+thid];

		 __syncthreads();
		DUMP(thid, n, temp, pout, iter);
	 }

	 g_odata[thid] = temp[pout*n+thid]; // write output
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum */
	reduceOp_t h_Ops[NUM_OPS];
	checkCudaErrors(cudaMemcpyFromSymbol(&h_Ops[0], d_pMin, sizeof(reduceOp_t)));
	checkCudaErrors(cudaMemcpyFromSymbol(&h_Ops[1], d_pMax, sizeof(reduceOp_t)));
	reduceOp_t* d_Ops;
	checkCudaErrors(cudaMalloc(&d_Ops, NUM_OPS * sizeof(reduceOp_t)));
	checkCudaErrors(cudaMemcpy(d_Ops, h_Ops, NUM_OPS * sizeof(reduceOp_t), cudaMemcpyHostToDevice));
	reduceOp_std_t std_Ops[NUM_OPS];
	std_Ops[0] = std::min;
	std_Ops[1] = std::max;
	float reduceResults[NUM_OPS];

	const dim3 blockSize(256);
	const dim3 gridSize(((numRows * numCols) + blockSize.x - 1) / blockSize.x);
	const size_t outputSize = gridSize.x * sizeof(float);
	float *h_outputBuffer = (float*) malloc(outputSize);
	float *d_outputBuffer;
	checkCudaErrors(cudaMalloc(&d_outputBuffer, outputSize));
	for (size_t i = 0; i < NUM_OPS; i++)
	{
		checkCudaErrors(cudaMemset(d_outputBuffer, 0, outputSize));
		reduce<<<gridSize, blockSize>>>(d_logLuminance, numRows * numCols, 0, d_Ops, i, d_outputBuffer);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpy(h_outputBuffer, d_outputBuffer, outputSize, cudaMemcpyDeviceToHost));
		float result = h_outputBuffer[0];
		for (size_t j = 0; j < gridSize.x; j++)
		{
			result = std_Ops[i](result, h_outputBuffer[j]);
		}
		reduceResults[i] = result;
	}
    cudaFree(d_outputBuffer);
    free(h_outputBuffer);
    cudaFree(d_Ops);

    min_logLum = reduceResults[0];
    max_logLum = reduceResults[1];

//    size_t TEST_SIZE = 257;
//	const dim3 blockSize(256);
//	const dim3 gridSize((TEST_SIZE + blockSize.x - 1) / blockSize.x);
//    float* h_testBuffer;
//    h_testBuffer = (float*) malloc(TEST_SIZE * sizeof(float));
//    memset(h_testBuffer, 0, TEST_SIZE * sizeof(float));
//    h_testBuffer[0] = 6;
//    h_testBuffer[1] = -8;
//    h_testBuffer[2] = 5;
//    h_testBuffer[3] = -4;
//    h_testBuffer[4] = 10;
//    h_testBuffer[TEST_SIZE-1] = 11;
//    float* d_testBuffer;
//    checkCudaErrors(cudaMalloc(&d_testBuffer, TEST_SIZE * sizeof(float)));
//    checkCudaErrors(cudaMemcpy(d_testBuffer, h_testBuffer, TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice));
//	const size_t outputSize = gridSize.x * sizeof(float);
//	float *d_outputBuffer;
//	checkCudaErrors(cudaMalloc(&d_outputBuffer, outputSize));
//	checkCudaErrors(cudaMemset(d_outputBuffer, 0, outputSize));
//	reduce<<<gridSize, blockSize>>>(d_testBuffer, TEST_SIZE, 0, d_Ops, 1, d_outputBuffer); // , blockSize.x * sizeof(float)
//	float *h_outputBuffer = (float*) malloc(outputSize);
//	checkCudaErrors(cudaMemcpy(h_outputBuffer, d_outputBuffer, outputSize, cudaMemcpyDeviceToHost));
//    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//    free(h_testBuffer);
//    cudaFree(d_testBuffer);
//	cudaFree(d_outputBuffer);
//    for (size_t i = 0; i < gridSize.x; i++) printf("%f ", h_outputBuffer[i]);


    // 2) subtract them to find the range
    float range = max_logLum - min_logLum;

    		/* Here are the steps you need to implement

    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
    size_t histoSize = numBins * sizeof(unsigned int);
    unsigned int* d_histo;
    checkCudaErrors(cudaMalloc(&d_histo, histoSize));
    checkCudaErrors(cudaMemset(d_histo, 0, histoSize));
    histo<<<gridSize, blockSize>>>(d_logLuminance, numRows * numCols, min_logLum, range, numBins, d_histo);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	unsigned int* h_histo = (unsigned int*) malloc(histoSize);
	checkCudaErrors(cudaMemcpy(h_histo, d_histo, histoSize, cudaMemcpyDeviceToHost));
	DUMP(0, numBins, h_histo, 0, 111);
	free(h_histo);

       /*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, pow(10, 7));
    exclusivePrefixSum<<<1, 1024, 2 * histoSize>>>(d_histo, d_cdf, 1024);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	unsigned int* h_cdf = (unsigned int*) malloc(histoSize);
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, histoSize, cudaMemcpyDeviceToHost));
	DUMP(0, numBins, h_cdf, 0, 222);
	free(h_cdf);
	cudaFree(d_histo);

}
