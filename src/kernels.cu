#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template<typename T>
__device__ T warp_reduce(T input) {
#pragma unroll
  for (int offset=16;offset>0;offset>>=1) {
    input+=__shfl_down_sync(0xFFFFFFFF,input,offset);
  }
  return input;
}

template<typename T>
__global__ void traceKernels(const T* d_input,T* d_result,size_t diag_len ,size_t cols) {
  extern __shared__ char smem_raw[];
  T* shared_mem = reinterpret_cast<T*>(smem_raw);

  const size_t tid=threadIdx.x;
  const size_t idx=blockIdx.x*blockDim.x+tid;

  T warp_sum_val=0;
  for (size_t i=idx;i<diag_len;i+=blockDim.x*gridDim.x) {
    warp_sum_val+=d_input[i*cols+i];
  }
  T warp_sum=warp_reduce(warp_sum_val);
  const size_t laneID=tid%32;
  const size_t warpID=tid/32;
  if (laneID==0) {
    shared_mem[warpID]=warp_sum;
  }
  __syncthreads();

  if (warpID==0) {
    T block_sum_val=(tid<(blockDim.x+31)/32)?shared_mem[tid]:T(0);
    T block_sum= warp_reduce(block_sum_val);
    if (tid==0) {
      atomicAdd(d_result,block_sum);
    }
  }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  if (rows==0||cols==0)return T(0);

  const size_t diag_len=std::min(rows,cols);
  const size_t input_size=rows*cols;

  T* d_input=nullptr;
  T* d_result=nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input,input_size*sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_result,sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_input,h_input.data(),input_size*sizeof(T),cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_result,0,sizeof(T)));

  size_t blockSize=256;
  size_t grimSize=(diag_len+blockSize-1)/blockSize;
  size_t smemSize = ((blockSize+31)/32) * sizeof(T);
  traceKernels<T><<<grimSize,blockSize,smemSize>>>(d_input,d_result,diag_len,cols);

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  T result;
  RUNTIME_CHECK(cudaMemcpy(&result,d_result,sizeof(T),cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_result));

  return result;
}
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
constexpr int WMMA_M=16,WMMA_N=16,WMMA_K=16;
constexpr int Br=64,Bc=64;
constexpr int PAD=8;

template<typename T,int D>
__global__ void flashAttentionKernel(
                              const T* __restrict__ Q,
                              const T* __restrict__ K,
                              const T* __restrict__ V,
                              T* __restrict__ O,
                              int q_len,
                              int kv_len,
                              int num_q_heads,
                              int num_kv_heads,
                              int head_dim,
                              float scale,
                              bool causal
  ) {
  const int batch=blockIdx.x,head=blockIdx.y,q_blk=blockIdx.z;
  const int tID =threadIdx.x,warpID=tID/32;
  const int num_warps=blockDim.x/32;
  const int kv_head=head/(num_q_heads/num_kv_heads);//算出当前q对应的kv头

  const size_t q_off=(size_t)batch*q_len*num_q_heads*head_dim+head*head_dim;
  const size_t kv_off=(size_t)batch*kv_len*num_kv_heads*head_dim+kv_head*head_dim;

  constexpr int D_pad=D+PAD;
  extern __shared__ T smem[];
  T* sQ=(T*)smem;
  T* sK=(T*)sQ+Br*D_pad;
  T* sV=(T*)sK+Bc*D_pad;
  float* sS=(float*)(sV+Bc*D_pad);//在共享内存上分配QKVS的指针




}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);