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
constexpr int PAD = 8;


template <typename T, int D, int Br, int Bc>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ O,
    int q_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale, bool causal
) {
    constexpr int D_PAD = D + PAD;

    const int batch = blockIdx.x, head = blockIdx.y, q_blk = blockIdx.z;
    const int tid = threadIdx.x;
    const int kv_head = head / (num_q_heads / num_kv_heads);//算出当前q对应的kv头

    const size_t q_off = (size_t)batch * q_len * num_q_heads * head_dim + head * head_dim;
    const size_t kv_off = (size_t)batch * kv_len * num_kv_heads * head_dim + kv_head * head_dim;
    const int row_stride = num_q_heads * head_dim;
    const int kv_stride = num_kv_heads * head_dim;

    // Shared Memory 布局
    extern __shared__ char smem[];
    T* sQ = (T*)smem;
    T* sK = sQ + Br * D_PAD;
    T* sV = sK + Bc * D_PAD;
    float* sS = (float*)(sV + Bc * D_PAD);//在共享内存上分配QKVS的指针

    float acc[D] = {0};
    float m_prev = -INFINITY, l_prev = 0;

    const int q_start = q_blk * Br;

    for (int i = tid; i < Br * D; i += blockDim.x) {
        int r = i / D, d = i % D;
        int gq = q_start + r;
        sQ[r * D_PAD + d] = (gq < q_len && d < head_dim) ? Q[q_off + gq * row_stride + d] : T(0);
    }
    __syncthreads();

    for (int kv_blk = 0; kv_blk < (kv_len + Bc - 1) / Bc; kv_blk++) {
        int kv_start = kv_blk * Bc;
        if (causal && kv_start > q_start + Br - 1) break;

        for (int i = tid; i < Bc * D; i += blockDim.x) {
            int r = i / D, d = i % D;
            int gk = kv_start + r;
            bool valid = gk < kv_len && d < head_dim;
            size_t idx = kv_off + gk * kv_stride + d;
            sK[r * D_PAD + d] = valid ? K[idx] : T(0);
            sV[r * D_PAD + d] = valid ? V[idx] : T(0);
        }
        __syncthreads();

        if (tid < Br) {
            for (int k = 0; k < Bc; k++) {
                float dot = 0;
                for (int d = 0; d < D; d++) {
                    dot += float(sQ[tid * D_PAD + d]) * float(sK[k * D_PAD + d]);
                }
                sS[tid * Bc + k] = dot;
            }
        }
        __syncthreads();

        if (tid < Br) {
            int gq = q_start + tid;
            if (gq < q_len) {
                float local_max = -INFINITY;

                for (int k = 0; k < Bc; k++) {
                    int gk = kv_start + k;
                    float s = (causal && gk > gq) || gk >= kv_len ? -INFINITY : sS[tid * Bc + k] * scale;
                    sS[tid * Bc + k] = s;
                    local_max = fmaxf(local_max, s);
                }

                float m_new = fmaxf(m_prev, local_max);
                float exp_diff = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
                float row_sum = 0;

                for (int d = 0; d < D; d++) acc[d] *= exp_diff;

                for (int k = 0; k < Bc; k++) {
                    float s = sS[tid * Bc + k];
                    if (s > -INFINITY) {
                        float p = expf(s - m_new);
                        row_sum += p;
                        for (int d = 0; d < D; d++) {
                            acc[d] += p * float(sV[k * D_PAD + d]);
                        }
                    }
                }

                m_prev = m_new;
                l_prev = l_prev * exp_diff + row_sum;
            }
        }
        __syncthreads();
    }

    if (tid < Br) {
        int gq = q_start + tid;
        if (gq < q_len) {
            T* op = O + q_off + gq * row_stride;
            float inv_l = l_prev > 0.0f ? 1.0f / l_prev : 0.0f;
            for (int d = 0; d < head_dim; d++) {
                op[d] = T(acc[d] * inv_l);
            }
        }
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch, int q_len, int kv_len, int q_heads, int kv_heads, int head_dim, bool causal) {
    size_t o_size = batch * q_len * q_heads * head_dim;
    h_o.resize(o_size);

    T *d_q, *d_k, *d_v, *d_o;
    size_t q_bytes = h_q.size() * sizeof(T);
    size_t k_bytes = h_k.size() * sizeof(T);
    size_t v_bytes = h_v.size() * sizeof(T);

    RUNTIME_CHECK(cudaMalloc(&d_q, q_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));

    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    if (head_dim <= 64) {
        constexpr int D = 64;
        constexpr int Br = 32;
        constexpr int Bc = 32;
        constexpr int D_PAD = D + PAD;

        dim3 grid(batch, q_heads, (q_len + Br - 1) / Br);
        dim3 block(Br);

        size_t smem_size = (Br * D_PAD + 2 * Bc * D_PAD) * sizeof(T) + Br * Bc * sizeof(float);

        flashAttentionKernel<T, D, Br, Bc><<<grid, block, smem_size>>>(
            d_q, d_k, d_v, d_o, q_len, kv_len, q_heads, kv_heads, head_dim, scale, causal);

    } else if (head_dim <= 128) {
        constexpr int D = 128;
        constexpr int Br = 16;
        constexpr int Bc = 16;
        constexpr int D_PAD = D + PAD;

        dim3 grid(batch, q_heads, (q_len + Br - 1) / Br);
        dim3 block(Br);

        size_t smem_size = (Br * D_PAD + 2 * Bc * D_PAD) * sizeof(T) + Br * Bc * sizeof(float);

        flashAttentionKernel<T, D, Br, Bc><<<grid, block, smem_size>>>(
            d_q, d_k, d_v, d_o, q_len, kv_len, q_heads, kv_heads, head_dim, scale, causal);
    } else {
        std::cerr << "Error: head_dim > 128 not supported yet." << std::endl;
    }

    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));

    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
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