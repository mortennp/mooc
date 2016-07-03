//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

const size_t BLOCK_SIZE = 256;
const size_t BITS_PR_PASS = 1;
const size_t PATTERNS_PR_PASS = 1 << BITS_PR_PASS;

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

//#define TRACE
#undef TRACE

#if defined(TRACE)
	#define DUMP(a, b, c, d, e) dump(a, b, c, d, e)
#else
	#define DUMP(a, b, c, d, e)
#endif

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

__device__ void _exclusivePrefixSum(const unsigned int* const s_idata, unsigned int *s_odata, size_t n, unsigned int *temp)
{
	 int thid = threadIdx.x;
	 int pout = 0, pin = 1;
	 int iter = 0;

	 // load input into shared memory.
	 // This is exclusive scan, so shift right by one and set first elt to 0
	  temp[pout*n + thid] = (thid > 0) ? s_idata[thid-1] : 0;

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

	 s_odata[thid] = temp[pout*n+thid]; // write output
}

__global__ void exclusivePrefixSum(const unsigned int* const s_idata, unsigned int *s_odata, size_t n)
{
	 extern __shared__ unsigned int temp[];

	 _exclusivePrefixSum(s_idata, s_odata, n, temp);
}

__global__ void radixSortTilePhase(
		size_t passNo,
		unsigned int* const d_inputVals,
		unsigned int* const d_inputPos,
		unsigned int* const d_outputVals,
		unsigned int* const d_outputPos,
		const size_t numElems,
		unsigned int* d_blockHistos)
{
	// Calc offsets into shared working memory
	__shared__ unsigned int working[2 * PATTERNS_PR_PASS * BLOCK_SIZE]; // a predicate array and a excl sum array pr bit pattern

	size_t predicateOffsets[PATTERNS_PR_PASS];
	size_t exclSumOffsets[PATTERNS_PR_PASS];
	size_t workingIdx = 0;
	for (int i = 0; i < PATTERNS_PR_PASS; i++)
	{
		predicateOffsets[i] = workingIdx;
		workingIdx += BLOCK_SIZE;
		exclSumOffsets[i] = workingIdx;
		workingIdx += BLOCK_SIZE;
	}

	// Set indices
	size_t inputIdx = threadIdx.x;
	size_t tileOffset = blockIdx.x * blockDim.x;
	size_t globalInputIdx = tileOffset + inputIdx;

	// Calc predicate vectors for each pattern
	size_t bitNo = (passNo * BITS_PR_PASS);
	unsigned int mask = (PATTERNS_PR_PASS - 1) << bitNo;
	unsigned int val = 	globalInputIdx < numElems ? d_inputVals[globalInputIdx] : 0;
	unsigned int pattern = val & mask;
	for (int i = 0; i < PATTERNS_PR_PASS; i++)
	{
		working[predicateOffsets[i] + inputIdx] = (pattern == i);
	}
	__syncthreads();

	// Excl prefix sum each pattern vector
	for (int i = 0; i < PATTERNS_PR_PASS; i++)
	{
		__shared__ unsigned int temp[2 * BLOCK_SIZE];
		_exclusivePrefixSum(&working[predicateOffsets[i]], &working[exclSumOffsets[i]], BLOCK_SIZE, temp);
		__syncthreads();
	}

	// Output counts
	if (inputIdx == 0)
	{
		for (int i = 0; i < PATTERNS_PR_PASS; i++)
		{
			d_blockHistos[i*gridDim.x + blockIdx.x] =
					working[exclSumOffsets[i] + BLOCK_SIZE - 1] +
					working[predicateOffsets[i] + BLOCK_SIZE - 1];
		}
	}

	// Sort tile in place
	if (globalInputIdx < numElems)
	{
		size_t histoInc = pattern > 0 ? d_blockHistos[(pattern - 1)*gridDim.x + blockIdx.x] : 0;
		size_t globalOutputIdx = tileOffset + histoInc + working[exclSumOffsets[pattern] + inputIdx];
		d_outputVals[globalOutputIdx] = d_inputVals[globalInputIdx];
		d_outputPos[globalOutputIdx] = d_inputPos[globalInputIdx];
	}
}

__global__ void radixSortGlobalPhase(
		size_t passNo,
		unsigned int* const d_inputVals,
		unsigned int* const d_inputPos,
		unsigned int* const d_outputVals,
		unsigned int* const d_outputPos,
		const size_t numElems,
		unsigned int* d_blockHistos,
		unsigned int* d_blockHistoSums)
{
	size_t inputIdx = threadIdx.x;
	size_t tileOffset = blockIdx.x * blockDim.x;
	size_t globalInputIdx = tileOffset + inputIdx;

	size_t bitNo = (passNo * BITS_PR_PASS);
	unsigned int mask = (PATTERNS_PR_PASS - 1) << bitNo;
	unsigned int val = 	globalInputIdx < numElems ? d_inputVals[globalInputIdx] : 0;
	unsigned int pattern = val & mask;

	size_t prevPatternTotal = pattern > 0 ? d_blockHistoSums[(pattern - 1) * gridDim.x + (gridDim.x - 1)]: 0;
	size_t myPatternPrevBlocksTotal = d_blockHistoSums[pattern * gridDim.x + blockIdx.x];
	size_t prevPatternMyBlockTotal = pattern > 0 ? d_blockHistos[(pattern - 1) * gridDim.x + blockIdx.x] : 0;
	size_t globalOutputIdx =  prevPatternTotal + myPatternPrevBlocksTotal + (threadIdx.x - prevPatternMyBlockTotal);

	d_outputVals[globalOutputIdx] = d_inputVals[globalInputIdx];
	d_outputPos[globalOutputIdx] = d_inputPos[globalInputIdx];
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
	const dim3 blockSize(BLOCK_SIZE);
	const dim3 gridSize((numElems + BLOCK_SIZE - 1) / BLOCK_SIZE);

	unsigned int* d_blockHistos;
	checkCudaErrors(cudaMalloc(&d_blockHistos, gridSize.x * PATTERNS_PR_PASS * sizeof(unsigned int)));
	unsigned int* d_blockHistoSums;
	checkCudaErrors(cudaMalloc(&d_blockHistoSums, gridSize.x * PATTERNS_PR_PASS * sizeof(unsigned int)));

	for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += BITS_PR_PASS)
	{
		// cudaMemset histos?

		radixSortTilePhase<<<gridSize, blockSize, 2 * blockSize.x * sizeof(unsigned int)>>>(0, d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems, d_blockHistos);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		for (int i = 0; i < PATTERNS_PR_PASS; i++)
		{
			size_t offset = i * gridSize.x;
			exclusivePrefixSum<<<1, gridSize, 2 * gridSize.x * sizeof(unsigned int)>>>(&d_blockHistos[offset], &d_blockHistoSums[offset], gridSize.x);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}

		// Note reversed order of d_output and d_input args
		radixSortGlobalPhase<<<gridSize, blockSize>>>(0, d_outputVals, d_outputPos, d_inputVals, d_inputPos, numElems, d_blockHistos, d_blockHistoSums);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	checkCudaErrors(cudaFree(d_blockHistos));
	checkCudaErrors(cudaFree(d_blockHistoSums));

//	//debugTest<<<1,1>>>(d_inputVals);
//	thrust::device_ptr<unsigned int> td_inputVals = thrust::device_pointer_cast(d_inputVals);
//	thrust::device_ptr<unsigned int> td_inputPos = thrust::device_pointer_cast(d_inputPos);
//	dump(td_inputVals);
//
//	thrust::stable_sort_by_key(td_inputVals, td_inputVals + numElems, td_inputPos);
//	thrust::device_ptr<unsigned int> td_outputVals = thrust::device_pointer_cast(d_outputVals);
//	thrust::copy(td_inputVals, td_inputVals + numElems, td_outputVals);
//	thrust::device_ptr<unsigned int> td_outputPos = thrust::device_pointer_cast(d_outputPos);
//	thrust::copy(td_inputPos, td_inputPos + numElems, td_outputPos);
//	td_outputPos[10] = 666;
//	dump(td_outputVals);

}

//void dump(thrust::device_ptr<unsigned int>& dptr)
//{
//	thrust::copy(dptr, dptr + 30, std::ostream_iterator<float>(std::cout, " "));
//	std::cout << "\n";
//}
