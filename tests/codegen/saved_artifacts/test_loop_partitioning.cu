
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(float* __restrict__ X, float* __restrict__ W, float* __restrict__ T_matmul_NT) {
  float T_matmul_NT_local[64];
  __shared__ float X_shared[512];
  __shared__ float W_shared[512];
  if (((int)blockIdx.x) < 126) {
    T_matmul_NT_local[(0)] = 0.000000e+00f;
    T_matmul_NT_local[(32)] = 0.000000e+00f;
    T_matmul_NT_local[(1)] = 0.000000e+00f;
    T_matmul_NT_local[(33)] = 0.000000e+00f;
    T_matmul_NT_local[(2)] = 0.000000e+00f;
    T_matmul_NT_local[(34)] = 0.000000e+00f;
    T_matmul_NT_local[(3)] = 0.000000e+00f;
    T_matmul_NT_local[(35)] = 0.000000e+00f;
    T_matmul_NT_local[(4)] = 0.000000e+00f;
    T_matmul_NT_local[(36)] = 0.000000e+00f;
    T_matmul_NT_local[(5)] = 0.000000e+00f;
    T_matmul_NT_local[(37)] = 0.000000e+00f;
    T_matmul_NT_local[(6)] = 0.000000e+00f;
    T_matmul_NT_local[(38)] = 0.000000e+00f;
    T_matmul_NT_local[(7)] = 0.000000e+00f;
    T_matmul_NT_local[(39)] = 0.000000e+00f;
    T_matmul_NT_local[(8)] = 0.000000e+00f;
    T_matmul_NT_local[(40)] = 0.000000e+00f;
    T_matmul_NT_local[(9)] = 0.000000e+00f;
    T_matmul_NT_local[(41)] = 0.000000e+00f;
    T_matmul_NT_local[(10)] = 0.000000e+00f;
    T_matmul_NT_local[(42)] = 0.000000e+00f;
    T_matmul_NT_local[(11)] = 0.000000e+00f;
    T_matmul_NT_local[(43)] = 0.000000e+00f;
    T_matmul_NT_local[(12)] = 0.000000e+00f;
    T_matmul_NT_local[(44)] = 0.000000e+00f;
    T_matmul_NT_local[(13)] = 0.000000e+00f;
    T_matmul_NT_local[(45)] = 0.000000e+00f;
    T_matmul_NT_local[(14)] = 0.000000e+00f;
    T_matmul_NT_local[(46)] = 0.000000e+00f;
    T_matmul_NT_local[(15)] = 0.000000e+00f;
    T_matmul_NT_local[(47)] = 0.000000e+00f;
    T_matmul_NT_local[(16)] = 0.000000e+00f;
    T_matmul_NT_local[(48)] = 0.000000e+00f;
    T_matmul_NT_local[(17)] = 0.000000e+00f;
    T_matmul_NT_local[(49)] = 0.000000e+00f;
    T_matmul_NT_local[(18)] = 0.000000e+00f;
    T_matmul_NT_local[(50)] = 0.000000e+00f;
    T_matmul_NT_local[(19)] = 0.000000e+00f;
    T_matmul_NT_local[(51)] = 0.000000e+00f;
    T_matmul_NT_local[(20)] = 0.000000e+00f;
    T_matmul_NT_local[(52)] = 0.000000e+00f;
    T_matmul_NT_local[(21)] = 0.000000e+00f;
    T_matmul_NT_local[(53)] = 0.000000e+00f;
    T_matmul_NT_local[(22)] = 0.000000e+00f;
    T_matmul_NT_local[(54)] = 0.000000e+00f;
    T_matmul_NT_local[(23)] = 0.000000e+00f;
    T_matmul_NT_local[(55)] = 0.000000e+00f;
    T_matmul_NT_local[(24)] = 0.000000e+00f;
    T_matmul_NT_local[(56)] = 0.000000e+00f;
    T_matmul_NT_local[(25)] = 0.000000e+00f;
    T_matmul_NT_local[(57)] = 0.000000e+00f;
    T_matmul_NT_local[(26)] = 0.000000e+00f;
    T_matmul_NT_local[(58)] = 0.000000e+00f;
    T_matmul_NT_local[(27)] = 0.000000e+00f;
    T_matmul_NT_local[(59)] = 0.000000e+00f;
    T_matmul_NT_local[(28)] = 0.000000e+00f;
    T_matmul_NT_local[(60)] = 0.000000e+00f;
    T_matmul_NT_local[(29)] = 0.000000e+00f;
    T_matmul_NT_local[(61)] = 0.000000e+00f;
    T_matmul_NT_local[(30)] = 0.000000e+00f;
    T_matmul_NT_local[(62)] = 0.000000e+00f;
    T_matmul_NT_local[(31)] = 0.000000e+00f;
    T_matmul_NT_local[(63)] = 0.000000e+00f;
    for (int k_outer_outer = 0; k_outer_outer < 192; ++k_outer_outer) {
      __syncthreads();
      if (((int)threadIdx.x) < 128) {
        X_shared[((((int)threadIdx.x) * 4))] = X[(((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer * 4)))];
        X_shared[(((((int)threadIdx.x) * 4) + 1))] = X[((((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer * 4)) + 1))];
        X_shared[(((((int)threadIdx.x) * 4) + 2))] = X[((((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer * 4)) + 2))];
        X_shared[(((((int)threadIdx.x) * 4) + 3))] = X[((((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer * 4)) + 3))];
      }
      W_shared[((((int)threadIdx.x) * 2))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((int)threadIdx.x) >> 1) * 770)) + (k_outer_outer * 4)) + ((((int)threadIdx.x) & 1) * 2)))];
      W_shared[(((((int)threadIdx.x) * 2) + 1))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 770)) + (k_outer_outer * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)))];
      __syncthreads();
      T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
      T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
      T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
      T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
      T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
    }
    __syncthreads();
    if (((int)threadIdx.x) < 128) {
      X_shared[((((int)threadIdx.x) * 4))] = X[(((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + 768))];
      X_shared[(((((int)threadIdx.x) * 4) + 1))] = X[(((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + 769))];
    }
    if ((((int)threadIdx.x) & 1) < 1) {
      W_shared[((((int)threadIdx.x) * 2))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((int)threadIdx.x) >> 1) * 770)) + ((((int)threadIdx.x) & 1) * 2)) + 768))];
    }
    if ((((((int)threadIdx.x) * 2) + 1) & 3) < 2) {
      W_shared[(((((int)threadIdx.x) * 2) + 1))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 770)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 768))];
    }
    __syncthreads();
    T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
    T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
    T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
    T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
    T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    for (int i_inner = 0; i_inner < 16; ++i_inner) {
      for (int j_inner = 0; j_inner < 2; ++j_inner) {
        T_matmul_NT[((((((((((int)blockIdx.x) / 18) * 294912) + ((((int)threadIdx.x) >> 5) * 36864)) + (i_inner * 2304)) + ((((int)blockIdx.x) % 18) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j_inner))] = T_matmul_NT_local[(((i_inner * 2) + j_inner))];
        T_matmul_NT[(((((((((((int)blockIdx.x) / 18) * 294912) + ((((int)threadIdx.x) >> 5) * 36864)) + (i_inner * 2304)) + ((((int)blockIdx.x) % 18) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j_inner) + 64))] = T_matmul_NT_local[((((i_inner * 2) + j_inner) + 32))];
      }
    }
  } else {
    T_matmul_NT_local[(0)] = 0.000000e+00f;
    T_matmul_NT_local[(32)] = 0.000000e+00f;
    T_matmul_NT_local[(1)] = 0.000000e+00f;
    T_matmul_NT_local[(33)] = 0.000000e+00f;
    T_matmul_NT_local[(2)] = 0.000000e+00f;
    T_matmul_NT_local[(34)] = 0.000000e+00f;
    T_matmul_NT_local[(3)] = 0.000000e+00f;
    T_matmul_NT_local[(35)] = 0.000000e+00f;
    T_matmul_NT_local[(4)] = 0.000000e+00f;
    T_matmul_NT_local[(36)] = 0.000000e+00f;
    T_matmul_NT_local[(5)] = 0.000000e+00f;
    T_matmul_NT_local[(37)] = 0.000000e+00f;
    T_matmul_NT_local[(6)] = 0.000000e+00f;
    T_matmul_NT_local[(38)] = 0.000000e+00f;
    T_matmul_NT_local[(7)] = 0.000000e+00f;
    T_matmul_NT_local[(39)] = 0.000000e+00f;
    T_matmul_NT_local[(8)] = 0.000000e+00f;
    T_matmul_NT_local[(40)] = 0.000000e+00f;
    T_matmul_NT_local[(9)] = 0.000000e+00f;
    T_matmul_NT_local[(41)] = 0.000000e+00f;
    T_matmul_NT_local[(10)] = 0.000000e+00f;
    T_matmul_NT_local[(42)] = 0.000000e+00f;
    T_matmul_NT_local[(11)] = 0.000000e+00f;
    T_matmul_NT_local[(43)] = 0.000000e+00f;
    T_matmul_NT_local[(12)] = 0.000000e+00f;
    T_matmul_NT_local[(44)] = 0.000000e+00f;
    T_matmul_NT_local[(13)] = 0.000000e+00f;
    T_matmul_NT_local[(45)] = 0.000000e+00f;
    T_matmul_NT_local[(14)] = 0.000000e+00f;
    T_matmul_NT_local[(46)] = 0.000000e+00f;
    T_matmul_NT_local[(15)] = 0.000000e+00f;
    T_matmul_NT_local[(47)] = 0.000000e+00f;
    T_matmul_NT_local[(16)] = 0.000000e+00f;
    T_matmul_NT_local[(48)] = 0.000000e+00f;
    T_matmul_NT_local[(17)] = 0.000000e+00f;
    T_matmul_NT_local[(49)] = 0.000000e+00f;
    T_matmul_NT_local[(18)] = 0.000000e+00f;
    T_matmul_NT_local[(50)] = 0.000000e+00f;
    T_matmul_NT_local[(19)] = 0.000000e+00f;
    T_matmul_NT_local[(51)] = 0.000000e+00f;
    T_matmul_NT_local[(20)] = 0.000000e+00f;
    T_matmul_NT_local[(52)] = 0.000000e+00f;
    T_matmul_NT_local[(21)] = 0.000000e+00f;
    T_matmul_NT_local[(53)] = 0.000000e+00f;
    T_matmul_NT_local[(22)] = 0.000000e+00f;
    T_matmul_NT_local[(54)] = 0.000000e+00f;
    T_matmul_NT_local[(23)] = 0.000000e+00f;
    T_matmul_NT_local[(55)] = 0.000000e+00f;
    T_matmul_NT_local[(24)] = 0.000000e+00f;
    T_matmul_NT_local[(56)] = 0.000000e+00f;
    T_matmul_NT_local[(25)] = 0.000000e+00f;
    T_matmul_NT_local[(57)] = 0.000000e+00f;
    T_matmul_NT_local[(26)] = 0.000000e+00f;
    T_matmul_NT_local[(58)] = 0.000000e+00f;
    T_matmul_NT_local[(27)] = 0.000000e+00f;
    T_matmul_NT_local[(59)] = 0.000000e+00f;
    T_matmul_NT_local[(28)] = 0.000000e+00f;
    T_matmul_NT_local[(60)] = 0.000000e+00f;
    T_matmul_NT_local[(29)] = 0.000000e+00f;
    T_matmul_NT_local[(61)] = 0.000000e+00f;
    T_matmul_NT_local[(30)] = 0.000000e+00f;
    T_matmul_NT_local[(62)] = 0.000000e+00f;
    T_matmul_NT_local[(31)] = 0.000000e+00f;
    T_matmul_NT_local[(63)] = 0.000000e+00f;
    for (int k_outer_outer1 = 0; k_outer_outer1 < 192; ++k_outer_outer1) {
      __syncthreads();
      if (((int)threadIdx.x) < 128) {
        if (((int)threadIdx.x) < 64) {
          X_shared[((((int)threadIdx.x) * 4))] = X[(((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer1 * 4)))];
          X_shared[(((((int)threadIdx.x) * 4) + 1))] = X[((((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer1 * 4)) + 1))];
          X_shared[(((((int)threadIdx.x) * 4) + 2))] = X[((((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer1 * 4)) + 2))];
          X_shared[(((((int)threadIdx.x) * 4) + 3))] = X[((((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + (k_outer_outer1 * 4)) + 3))];
        }
      }
      W_shared[((((int)threadIdx.x) * 2))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((int)threadIdx.x) >> 1) * 770)) + (k_outer_outer1 * 4)) + ((((int)threadIdx.x) & 1) * 2)))];
      W_shared[(((((int)threadIdx.x) * 2) + 1))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 770)) + (k_outer_outer1 * 4)) + (((((int)threadIdx.x) * 2) + 1) & 3)))];
      __syncthreads();
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 2))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 6))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 3))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 7))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 10))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 14))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 11))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 15))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 18))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 22))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 19))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 23))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 26))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 30))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 27))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 31))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 34))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 38))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 35))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 39))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 42))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 46))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 43))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 47))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 50))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 54))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 51))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 55))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
        T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
        T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
        T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
        T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 58))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 2))]));
        T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 258))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 6))]));
        T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 62))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 262))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 59))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 3))]));
        T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 259))]));
      }
      if (((int)threadIdx.x) < 128) {
        T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 7))]));
        T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 63))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 263))]));
      }
    }
    __syncthreads();
    if (((int)threadIdx.x) < 128) {
      if (((int)threadIdx.x) < 64) {
        X_shared[((((int)threadIdx.x) * 4))] = X[(((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + 768))];
        X_shared[(((((int)threadIdx.x) * 4) + 1))] = X[(((((((int)blockIdx.x) / 18) * 98560) + (((int)threadIdx.x) * 770)) + 769))];
      }
    }
    if ((((int)threadIdx.x) & 1) < 1) {
      W_shared[((((int)threadIdx.x) * 2))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((int)threadIdx.x) >> 1) * 770)) + ((((int)threadIdx.x) & 1) * 2)) + 768))];
    }
    if ((((((int)threadIdx.x) * 2) + 1) & 3) < 2) {
      W_shared[(((((int)threadIdx.x) * 2) + 1))] = W[((((((((int)blockIdx.x) % 18) * 98560) + ((((((int)threadIdx.x) * 2) + 1) >> 2) * 770)) + (((((int)threadIdx.x) * 2) + 1) & 3)) + 768))];
    }
    __syncthreads();
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[(((((int)threadIdx.x) >> 5) * 64))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 4))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(0)] = (T_matmul_NT_local[(0)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(32)] = (T_matmul_NT_local[(32)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(1)] = (T_matmul_NT_local[(1)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(33)] = (T_matmul_NT_local[(33)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 1))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(2)] = (T_matmul_NT_local[(2)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(34)] = (T_matmul_NT_local[(34)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(3)] = (T_matmul_NT_local[(3)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(35)] = (T_matmul_NT_local[(35)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 5))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 8))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 12))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(4)] = (T_matmul_NT_local[(4)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(36)] = (T_matmul_NT_local[(36)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(5)] = (T_matmul_NT_local[(5)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(37)] = (T_matmul_NT_local[(37)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 9))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(6)] = (T_matmul_NT_local[(6)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(38)] = (T_matmul_NT_local[(38)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(7)] = (T_matmul_NT_local[(7)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(39)] = (T_matmul_NT_local[(39)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 13))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 16))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 20))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(8)] = (T_matmul_NT_local[(8)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(40)] = (T_matmul_NT_local[(40)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(9)] = (T_matmul_NT_local[(9)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(41)] = (T_matmul_NT_local[(41)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 17))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(10)] = (T_matmul_NT_local[(10)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(42)] = (T_matmul_NT_local[(42)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(11)] = (T_matmul_NT_local[(11)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(43)] = (T_matmul_NT_local[(43)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 21))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 24))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 28))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(12)] = (T_matmul_NT_local[(12)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(44)] = (T_matmul_NT_local[(44)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(13)] = (T_matmul_NT_local[(13)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(45)] = (T_matmul_NT_local[(45)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 25))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(14)] = (T_matmul_NT_local[(14)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(46)] = (T_matmul_NT_local[(46)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(15)] = (T_matmul_NT_local[(15)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(47)] = (T_matmul_NT_local[(47)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 29))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 32))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 36))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(16)] = (T_matmul_NT_local[(16)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(48)] = (T_matmul_NT_local[(48)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(17)] = (T_matmul_NT_local[(17)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(49)] = (T_matmul_NT_local[(49)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 33))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(18)] = (T_matmul_NT_local[(18)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(50)] = (T_matmul_NT_local[(50)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(19)] = (T_matmul_NT_local[(19)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(51)] = (T_matmul_NT_local[(51)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 37))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 40))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 44))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(20)] = (T_matmul_NT_local[(20)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(52)] = (T_matmul_NT_local[(52)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(21)] = (T_matmul_NT_local[(21)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(53)] = (T_matmul_NT_local[(53)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 41))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(22)] = (T_matmul_NT_local[(22)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(54)] = (T_matmul_NT_local[(54)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(23)] = (T_matmul_NT_local[(23)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(55)] = (T_matmul_NT_local[(55)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 45))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 48))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 52))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(24)] = (T_matmul_NT_local[(24)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(56)] = (T_matmul_NT_local[(56)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(25)] = (T_matmul_NT_local[(25)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(57)] = (T_matmul_NT_local[(57)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 49))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(26)] = (T_matmul_NT_local[(26)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(58)] = (T_matmul_NT_local[(58)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(27)] = (T_matmul_NT_local[(27)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(59)] = (T_matmul_NT_local[(59)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 53))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 56))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[(((((int)threadIdx.x) & 31) * 8))]));
      T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 256))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 4))]));
      T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 60))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 260))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(28)] = (T_matmul_NT_local[(28)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(60)] = (T_matmul_NT_local[(60)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(29)] = (T_matmul_NT_local[(29)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(61)] = (T_matmul_NT_local[(61)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 57))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(30)] = (T_matmul_NT_local[(30)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 1))]));
      T_matmul_NT_local[(62)] = (T_matmul_NT_local[(62)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 257))]));
    }
    if (((int)threadIdx.x) < 128) {
      T_matmul_NT_local[(31)] = (T_matmul_NT_local[(31)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 5))]));
      T_matmul_NT_local[(63)] = (T_matmul_NT_local[(63)] + (X_shared[((((((int)threadIdx.x) >> 5) * 64) + 61))] * W_shared[((((((int)threadIdx.x) & 31) * 8) + 261))]));
    }
    for (int i_inner1 = 0; i_inner1 < 16; ++i_inner1) {
      for (int j_inner1 = 0; j_inner1 < 2; ++j_inner1) {
        if ((((((int)threadIdx.x) >> 5) * 16) + i_inner1) < 64) {
          T_matmul_NT[((((((((((int)blockIdx.x) / 18) * 294912) + ((((int)threadIdx.x) >> 5) * 36864)) + (i_inner1 * 2304)) + ((((int)blockIdx.x) % 18) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j_inner1))] = T_matmul_NT_local[(((i_inner1 * 2) + j_inner1))];
          T_matmul_NT[(((((((((((int)blockIdx.x) / 18) * 294912) + ((((int)threadIdx.x) >> 5) * 36864)) + (i_inner1 * 2304)) + ((((int)blockIdx.x) % 18) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j_inner1) + 64))] = T_matmul_NT_local[((((i_inner1 * 2) + j_inner1) + 32))];
        }
      }
    }
  }
}

