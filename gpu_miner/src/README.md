# Homewrok no. 2 - proof of work algorithm in CUDA
# Mara Fichios - 331CA

For this project I learned how to implement an optimized version for two core components in blockchain mining:
constructing the merkle root for transactions and finding a valid nonce for the POW algorithm. My goal was to
optimize these computationally heavy functions by parallelizing them on GPU, achieving significant speedup
compared to a serial CPU implementation. (all tasks are implemented)

## General program flow:
- Transactions are read and grouped into blocks.
- For each block:
	- The Merkle root of the transactions is computed using `construct_merkle_root`.
	- The block content is formed by concatenating the previous block's hash and the Merkle root.
- Results for each block and execution times are outputted to a file.

Implemented functions:

## `construct_merkle_root`
- **Single Allocation:**  
  On the first call, GPU memory buffers are allocated once for all transactions and intermediate hashes
  (`d_tx`, `d_hash0`, `d_hash1`).

- **Transaction Copy:**  
  Only the current transactions (`n * transaction_size`) are copied from host to device memory.

- **Leaf Hashing:**  
  The kernel `kernel_hash_txs` is launched with `MERKLE_BLOCK` threads per block to compute SHA256
  hashes of each transaction in parallel.

- **Tree Construction:**  
  The `kernel_hash_pairs` kernel is alternately called for each Merkle tree level.  
  Buffers `d_hash0` and `d_hash1` are swapped at each level to avoid unnecessary intermediate copying.

- **Result Return:**  
  The final hash (merkle root) is copied from device memory back to `merkle_root` on the host.

### Optimizations:
- GPU buffers are allocated once and reused for all blocks.  
- Alternating between `d_hash0` and `d_hash1` buffers removes intermediate copy overhead.  
- Each tree level is fully computed in parallel on the GPU.  
- SHA256 computations are performed entirely on GPU.


## `find_nonce`
- **Reset GPU Flag:**  
  The device flag `d_flag` is reset to 0, indicating no nonce found yet.

- **Allocation and Copy:**  
  GPU memory is allocated for `block_content` and `difficulty`.  
  Input data is copied from host to device memory.

- **Nonce Batching:**  
  Nonces are processed in large batches (e.g., 1 million nonces per kernel launch).  
  The `find_nonce_kernel` is launched for each batch to test nonce candidates in parallel.

- **Result Checking:**  
  After each batch, the host checks if `d_flag` was set by any thread.  
  If set, the valid nonce and corresponding block hash are copied back to the host. Otherwise,
  the next batch is processed.

- **Memory Freeing:**  
  Finally, all allocated GPU memory is freed.

### Optimizations:
- Shared memory is used to reduce global memory reads by copying `block_content` once per block.  
- `atomicExch` ensures only the first thread to find a valid nonce saves the result, preventing race conditions.

## Helper functions:
- `kernel_hash_txs`
	- Receives the raw transactions array.
	- Each thread calculates the SHA256 hash of one transaction and stores the result in out.
	- Used in the first step of building the Merkle tree, hashing all leaf transactions in parallel.

- `kernel_hash_pairs`
	- Each thread takes two hashes from the current level (or duplicates the last one if odd count).
	- Concatenates the two hashes and applies SHA256.
	- Produces the next level of the merkle tree and it is called alternativelly until one hash remains.

- `find_nonce_kernel`
	- Each thread tries a unique nonce and Copies block_content into shared memory once per block for
	fast access.
	- Each thread creates a local copy, appends its nonce (as string), computes SHA256, and checks if
	the hash meets the difficulty.
	- If a valid nonce is found, uses atomicExch(&d_flag, 1) to atomically set a flag and save the
	result only once.

The performance test results for the test4 showcase the following times:
	- merkle root computation: 0.03693 on GPU vs 0.89863 on CPU (x24 sppedup)
	- nonce search: 0.01762 on GPU vs 2.09675 on CPU (X119 speedup)
	- total: 0.05455 on GPU vs 1.99539 on CPU (overall speedup of x55)
During building the solution, I have had some problems with optimizing and obtaining the desired times, but each
step that I have explained above helped me reach the goal times, as each optimizing procedure decreased the times.
I have come to these conclusions that are supported by the computation times:
	- CUDA parallelism transformed expensive parts (computationally speaking) of mining into highly 
	efficient GPU computations
	- some key optimizations include: persistent GPU memory allocation to avoid repeated allocations,
	parallel execution of hashing on many threads ,use of shared memory to reduce latency ,atomic
	operations to synchronize between threads and avoid race conditions.
