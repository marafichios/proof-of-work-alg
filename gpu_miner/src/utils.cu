#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>

// ─────────── tuning constants ──────────────────────────────────────
#define MERKLE_BLOCK   512          // threads per block pentru Merkle
#define NONCE_BLOCK    256
#define NONCE_BATCH_SIZE (1u<<20)
#define NONCE_BATCH    (NONCE_BLOCK * NONCE_BLOCK)

// ─────────── device‐side flags pentru PoW (dummy) ─────────────────────
__device__ int d_flag;
__device__ uint32_t d_found_nonce;
__device__ BYTE      d_found_hash_raw[32];
__device__ BYTE d_found_hash[SHA256_HASH_SIZE];

// puse în constant memory pentru viteză
__constant__ BYTE c_difficulty[SHA256_HASH_SIZE];
__constant__ BYTE c_prefix   [BLOCK_SIZE];

// ─────────── buffere persistente pe GPU pentru Merkle ────────────────
static bool merkle_inited = false;
static BYTE *d_tx    = nullptr;
static BYTE *d_hash0 = nullptr;
static BYTE *d_hash1 = nullptr;

// ─────────── funcţii utilitare CUDA ───────────────────────────────────
__host__ __device__ size_t d_strlen(const char *s) {
    size_t i = 0;
    while (s[i]) i++;
    return i;
}
__device__ void d_strcpy(char *d, const char *s) {
    int i = 0;
    while ((d[i] = s[i]) != '\0') i++;
}
__device__ void d_strcat(char *d, const char *s) {
    while (*d) d++;
    while ((*d++ = *s++) != '\0');
}
__device__ int intToString(uint64_t num, char *out) {
    if (num == 0) { out[0] = '0'; out[1] = '\0'; return 1; }
    int i = 0;
    while (num) {
        out[i++] = '0' + (num % 10);
        num /= 10;
    }
    for (int j = 0; j < i/2; j++) {
        char t = out[j]; out[j] = out[i-1-j]; out[i-1-j] = t;
    }
    out[i] = '\0';
    return i;
}

// ─────────── SHA256 wrappers (host+device) ────────────────────────────
__host__ __device__ void apply_sha256(const BYTE *in, BYTE *out) {
    size_t len = d_strlen((const char*)in);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex[] = "0123456789abcdef";
    sha256_init(&ctx);
    sha256_update(&ctx, in, len);
    sha256_final(&ctx, buf);
    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        out[i*2]   = hex[(buf[i] >> 4) & 0xF];
        out[i*2+1] = hex[ buf[i]       & 0xF];
    }
    out[SHA256_BLOCK_SIZE*2] = '\0';
}
__host__ __device__ int compare_hashes(const BYTE *h1, const BYTE *h2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (h1[i] < h2[i]) return -1;
        if (h1[i] > h2[i]) return  1;
    }
    return 0;
}

// ─────────── Merkle‐tree kernels ────────────────────────────────────
__global__ void kernel_hash_txs(const BYTE* tx, int tx_sz, BYTE* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    apply_sha256(tx + i*tx_sz, out + i*SHA256_HASH_SIZE);
}
__global__ void kernel_hash_pairs(const BYTE* in, BYTE* out, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = (n + 1)/2;
    if (idx >= total) return;
    int b = idx*2;
    BYTE buf[2*SHA256_HASH_SIZE];
    const BYTE *h1 = in + b*SHA256_HASH_SIZE;
    const BYTE *h2 = (b+1 < n ? in + (b+1)*SHA256_HASH_SIZE : h1);
    d_strcpy((char*)buf, (const char*)h1);
    d_strcat((char*)buf, (const char*)h2);
    apply_sha256(buf, out + idx*SHA256_HASH_SIZE);
}

// ─────────── Merkle‐root optimizat ─────────────────────────────────
void construct_merkle_root(int transaction_size,
                           BYTE *transactions,
                           int max_transactions_in_a_block,
                           int n,
                           BYTE merkle_root[SHA256_HASH_SIZE]) {
    if (!merkle_inited) {
        cudaMalloc(&d_tx,    max_transactions_in_a_block * transaction_size);
        cudaMalloc(&d_hash0, max_transactions_in_a_block * SHA256_HASH_SIZE);
        cudaMalloc(&d_hash1, max_transactions_in_a_block * SHA256_HASH_SIZE);
        merkle_inited = true;
    }
    cudaMemcpy(d_tx, transactions, n*transaction_size, cudaMemcpyHostToDevice);
    int blocks = (n + MERKLE_BLOCK - 1)/MERKLE_BLOCK;
    kernel_hash_txs<<<blocks, MERKLE_BLOCK>>>(d_tx, transaction_size, d_hash0, n);
    cudaDeviceSynchronize();
    int cur = n, turn = 0;
    while (cur > 1) {
        int next = (cur + 1)/2;
        int b    = (next + MERKLE_BLOCK - 1)/MERKLE_BLOCK;
        BYTE *in  = (turn & 1) ? d_hash1 : d_hash0;
        BYTE *out = (turn & 1) ? d_hash0 : d_hash1;
        kernel_hash_pairs<<<b, MERKLE_BLOCK>>>(in, out, cur);
        cudaDeviceSynchronize();
        cur = next;  turn++;
    }
    BYTE *src = (turn & 1) ? d_hash1 : d_hash0;
    cudaMemcpy(merkle_root, src, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
}

// ----- Nonce Finding ----- //

// Kernel to find nonce
__global__ void find_nonce_kernel(BYTE *block_content, size_t content_len, BYTE *difficulty, uint32_t start_nonce, uint32_t max_nonce) {
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    if (nonce > max_nonce || d_flag) return;
    
    // Prepare block with nonce
    __shared__ BYTE shared_block[1024];  // Adjust based on expected block size
    
    // Copy block content to shared memory (only first thread in block)
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < content_len; i++) {
            shared_block[i] = block_content[i];
        }
    }
    __syncthreads();
    
    // Create thread-local copy with the nonce
    BYTE local_block[1024];
    for (size_t i = 0; i < content_len; i++) {
        local_block[i] = shared_block[i];
    }
    
    // Add nonce
    char nonce_str[16];
    int nonce_len = intToString(nonce, nonce_str);
    for (int i = 0; i < nonce_len; i++) {
        local_block[content_len + i] = nonce_str[i];
    }
    local_block[content_len + nonce_len] = '\0';
    
    // Calculate hash
    BYTE hash[SHA256_HASH_SIZE];
    apply_sha256(local_block, hash);
    
    // Check if hash meets difficulty
    if (compare_hashes(hash, difficulty) <= 0) {
        if (atomicExch(&d_flag, 1) == 0) {
            d_found_nonce = nonce;
            for (int i = 0; i < SHA256_HASH_SIZE; i++) {
                d_found_hash[i] = hash[i];
            }
        }
    }
}

// Optimized nonce finding
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    BYTE *d_block_content, *d_difficulty;
    int found = 0;
    
    // Reset device flags
    cudaMemcpyToSymbol(d_flag, &found, sizeof(int));
    
    // Allocate and copy memory
    cudaMalloc((void **)&d_block_content, current_length + 16);
    cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
    
    cudaMemcpy(d_block_content, block_content, current_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple blocks
    int threads_per_block = 256;
    int batch_size = 1024 * 1024;  // Process 1M nonces per batch
    
    for (uint32_t start_nonce = 0; start_nonce <= max_nonce; start_nonce += batch_size) {
        uint32_t current_batch = min(batch_size, max_nonce - start_nonce + 1);
        int num_blocks = (current_batch + threads_per_block - 1) / threads_per_block;
        
        find_nonce_kernel<<<num_blocks, threads_per_block>>>(
            d_block_content, 
            current_length, 
            d_difficulty, 
            start_nonce, 
            max_nonce
        );
        
        // Check if nonce was found
        cudaMemcpyFromSymbol(&found, d_flag, sizeof(int));
        if (found) break;
    }
    
    // Get results if nonce was found
    if (found) {
        cudaMemcpyFromSymbol(valid_nonce, d_found_nonce, sizeof(uint32_t));
        cudaMemcpyFromSymbol(block_hash, d_found_hash, SHA256_HASH_SIZE);
    }
    
    // Free memory
    cudaFree(d_block_content);
    cudaFree(d_difficulty);
    
    return found ? 0 : 1;
}




// ─────────── dummy & warmup ────────────────────────────────────────
__global__ void dummy_kernel() {}
void warm_up_gpu() {
    BYTE *tmp;
    cudaMalloc(&tmp, 256);
    dummy_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaFree(tmp);
}

