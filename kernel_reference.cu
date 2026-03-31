// Исторический артефакт, сейчас все работает внутри grasshopper_module.cu

struct Block128 {
    unsigned int w[4];
};

extern __shared__ unsigned int s_all[];

__device__ __forceinline__ unsigned int swap32(unsigned int x) {
    return ((x << 24) & 0xff000000) | ((x << 8) & 0x00ff0000) |
           ((x >> 8) & 0x0000ff00) | ((x >> 24) & 0x000000ff);
}

__device__ void load_t_tables(const unsigned int* global_T) {
    int tid = threadIdx.x;
    unsigned int* s_ptr = (unsigned int*)&s_all[0];
    for (int i = tid; i < 16384; i += blockDim.x) {
        s_ptr[i] = global_T[i];
    }
    __syncthreads();
}

__device__ __forceinline__ Block128 apply_t_layer(Block128 state) {
    Block128 next = {0, 0, 0, 0};
    unsigned int (*s_T)[256][4] = (unsigned int (*)[256][4])s_all;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned char b = (unsigned char)(state.w[i / 4] >> ((i % 4) * 8));
        next.w[0] ^= s_T[i][b][0];
        next.w[1] ^= s_T[i][b][1];
        next.w[2] ^= s_T[i][b][2];
        next.w[3] ^= s_T[i][b][3];
    }
    return next;
}

extern "C" __global__ void grasshopper_ctr_v4(
    const uint4* __restrict__ data_in,
    uint4*       __restrict__ data_out,
    const uint4* __restrict__ rk,
    const unsigned char* __restrict__ nonce,
    const unsigned int* __restrict__ global_T,
    int n_blocks,
    int offset_idx 
) {
    load_t_tables(global_T);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;

    int global_block_idx = offset_idx + idx;

    Block128 ctr;
    unsigned int* n32 = (unsigned int*)nonce;
    ctr.w[0] = n32[0];
    ctr.w[1] = n32[1];
    unsigned long long b_idx = (unsigned long long)global_block_idx;
    ctr.w[2] = swap32((unsigned int)(b_idx >> 32));
    ctr.w[3] = swap32((unsigned int)(b_idx & 0xFFFFFFFF));

    #pragma unroll
    for (int r = 0; r < 9; r++) {
        uint4 k = rk[r];
        ctr.w[0] ^= k.x; ctr.w[1] ^= k.y; ctr.w[2] ^= k.z; ctr.w[3] ^= k.w;
        ctr = apply_t_layer(ctr);
    }
    uint4 k9 = rk[9];
    ctr.w[0] ^= k9.x; ctr.w[1] ^= k9.y; ctr.w[2] ^= k9.z; ctr.w[3] ^= k9.w;

    uint4 chunk = data_in[idx];
    chunk.x ^= ctr.w[0]; chunk.y ^= ctr.w[1]; chunk.z ^= ctr.w[2]; chunk.w ^= ctr.w[3];
    data_out[idx] = chunk;
}