#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>


#define MERKLE_BLOCK   512
#define NONCE_BLOCK    256
#define NONCE_BATCH_SIZE (1u<<20)
#define NONCE_BATCH    (NONCE_BLOCK * NONCE_BLOCK)


__device__ int d_flag;
__device__ uint32_t d_found_nonce;
__device__ BYTE      d_found_hash_raw[32];
__device__ BYTE d_found_hash[SHA256_HASH_SIZE];


__constant__ BYTE c_difficulty[SHA256_HASH_SIZE];
__constant__ BYTE c_prefix   [BLOCK_SIZE];


static bool merkle_inited = false;
static BYTE *d_tx    = nullptr;
static BYTE *d_hash0 = nullptr;
static BYTE *d_hash1 = nullptr;


// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 2;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src){
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}


// Calculate hash for each transaction
__global__ void kernel_hash_txs(const BYTE* tx, int tx_sz, BYTE* out, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    //check to not go past vector limit
    if (i >= n)
        return;

    //calculate sha for transaction i and save the outpuy
    apply_sha256(tx + i*tx_sz, out + i*SHA256_HASH_SIZE);
}

// BUild pairs of hashes
__global__ void kernel_hash_pairs(const BYTE* in, BYTE* out, int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = (n + 1)/2;

    if (idx >= total)
        return;
    
    //double the pair if it doesnt have a pair
    int b = idx * 2;
    BYTE buf[2 * SHA256_HASH_SIZE];
    const BYTE *h1 = in + b * SHA256_HASH_SIZE;
    const BYTE *h2 = (b + 1 < n ? in + (b + 1) * SHA256_HASH_SIZE : h1);

    d_strcpy((char*)buf, (const char*)h1);
    d_strcat((char*)buf, (const char*)h2);

    apply_sha256(buf, out + idx * SHA256_HASH_SIZE);
}

// Optimized merkle root func
void construct_merkle_root(int transaction_size,
                           BYTE *transactions,
                           int max_transactions_in_a_block,
                           int n,
                           BYTE merkle_root[SHA256_HASH_SIZE]) {
    //Allocate buffers only once
    if (!merkle_inited) {
        cudaMalloc(&d_tx,    max_transactions_in_a_block * transaction_size);
        cudaMalloc(&d_hash0, max_transactions_in_a_block * SHA256_HASH_SIZE);
        cudaMalloc(&d_hash1, max_transactions_in_a_block * SHA256_HASH_SIZE);
        merkle_inited = true;
    }
    //copy all transactions on gpu
    cudaMemcpy(d_tx, transactions, n*transaction_size, cudaMemcpyHostToDevice);
    
   //calculate hash for each transaction 
    int blocks = (n + MERKLE_BLOCK - 1) / MERKLE_BLOCK;
    kernel_hash_txs<<<blocks, MERKLE_BLOCK>>>(d_tx, transaction_size, d_hash0, n);
    cudaDeviceSynchronize();

    //build merkle tree
    int cur = n, turn = 0;
    while (cur > 1) {
        int next = (cur + 1) / 2;
        int b    = (next + MERKLE_BLOCK - 1) / MERKLE_BLOCK;

        //alternate between buffers to avoid other allocs
        BYTE *in  = (turn & 1) ? d_hash1 : d_hash0;
        BYTE *out = (turn & 1) ? d_hash0 : d_hash1;
        
        //calculate hash for current level
        kernel_hash_pairs<<<b, MERKLE_BLOCK>>>(in, out, cur);
        cudaDeviceSynchronize();
        cur = next;
        turn++;
    }
    
    //copy merkle root on host
    BYTE *src = (turn & 1) ? d_hash1 : d_hash0;
    cudaMemcpy(merkle_root, src, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
}



//nonce for each thread
__global__ void find_nonce_kernel(BYTE *block_content, size_t content_len, BYTE *difficulty, uint32_t start_nonce, uint32_t max_nonce) {
   
    //calculate the nonce for corresponding thread
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    if (nonce > max_nonce || d_flag)
	return;

    //use shared memory to copy the block contents
    __shared__ BYTE shared_block[1024];
    
    //copy the block content to shared memory only for the first thread
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < content_len; i++) {
            shared_block[i] = block_content[i];
        }
    }
    __syncthreads();
    
    //each thread copies the block locally to add its nonce
    BYTE local_block[1024];
    for (size_t i = 0; i < content_len; i++) {
        local_block[i] = shared_block[i];
    }
    
    char nonce_str[16];
    int nonce_len = intToString(nonce, nonce_str);

    //add nonce to the ending of the block
    for (int i = 0; i < nonce_len; i++) {
        local_block[content_len + i] = nonce_str[i];
    }
    local_block[content_len + nonce_len] = '\0';
    
    //calculate hash for updated block
    BYTE hash[SHA256_HASH_SIZE];
    apply_sha256(local_block, hash);
    
    //compare hash with difficutly
    if (compare_hashes(hash, difficulty) <= 0) {
        if (atomicExch(&d_flag, 1) == 0) {
            //set nonce as found
            d_found_nonce = nonce;
            for (int i = 0; i < SHA256_HASH_SIZE; i++) {
                d_found_hash[i] = hash[i];
            }
        }
    }
}

//optimized nonce finding
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    BYTE *d_block_content, *d_difficulty;
    int found = 0;
    
    //reset device flags to allow a new search
    cudaMemcpyToSymbol(d_flag, &found, sizeof(int));
    
    //allocate and copy memory
    cudaMalloc((void **)&d_block_content, current_length + 16);
    cudaMalloc((void **)&d_difficulty, SHA256_HASH_SIZE);
    
    //copy the block content and the difficulty
    cudaMemcpy(d_block_content, block_content, current_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    
    //test with 1 mill nonce per kernel
    int threads_per_block = 256;
    int batch_size = 1024 * 1024;
    
    //look for the nonce in batches
    for (uint32_t start_nonce = 0; start_nonce <= max_nonce; start_nonce += batch_size) {
        uint32_t current_batch = min(batch_size, max_nonce - start_nonce + 1);
        int num_blocks = (current_batch + threads_per_block - 1) / threads_per_block;
        
        //launce the kernel for current batch
        find_nonce_kernel<<<num_blocks, threads_per_block>>>(
            d_block_content, 
            current_length, 
            d_difficulty, 
            start_nonce, 
            max_nonce
        );
        
        //check if the nonce was found
        cudaMemcpyFromSymbol(&found, d_flag, sizeof(int));
        if (found) break;
    }
    
    //if yes copy the result back on host
    if (found) {
        cudaMemcpyFromSymbol(valid_nonce, d_found_nonce, sizeof(uint32_t));
        cudaMemcpyFromSymbol(block_hash, d_found_hash, SHA256_HASH_SIZE);
    }
    
    cudaFree(d_block_content);
    cudaFree(d_difficulty);
    
    return found ? 0 : 1;
}

__global__ void dummy_kernel() {}


// Warm-up function
void warm_up_gpu() {
    BYTE *tmp;
    cudaMalloc(&tmp, 256);
    dummy_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaFree(tmp);
}

