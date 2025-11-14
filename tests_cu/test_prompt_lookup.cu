#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <climits>

#include "cuda_runtime.h"
#include "../include/mirage/persistent_kernel/tasks/speculative_decoding/prompt_lookup.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %d: %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel configuration
constexpr int NGRAM_SIZE = 5;
constexpr int NUM_BLOCKS = 96;
constexpr int VOCAB_SIZE = 321467;
constexpr int SEQUENCE_LENGTH = 1024 * 1024 * 2; // 2M tokens
// constexpr int SEQUENCE_LENGTH = 1024 * 8; // 8K tokens
constexpr int NUM_WARMUP_ITERATIONS = 10;
constexpr int NUM_ITERATIONS = 500;

__global__ void test_wrapper_kernel(int const *__restrict__ input_ptr,
                                    int *__restrict__ ngram_id_ptr,
                                    long long *__restrict__ output_id_ptr,
                                    long long *__restrict__ final_output_ptr,
                                    int input_token_num) {
    kernel::find_ngram_partial_kernel<int, NGRAM_SIZE, NUM_BLOCKS>(
        input_ptr, ngram_id_ptr, output_id_ptr + blockIdx.x, input_token_num);

    kernel::find_ngram_global_kernel<NUM_BLOCKS>(output_id_ptr, final_output_ptr);
    // kernel::find_ngram_global_kernel_sequential<NUM_BLOCKS>(output_id_ptr, final_output_ptr);
}

// CPU-side verification function
long long find_first_ngram_cpu(const std::vector<int>& sequence, const std::vector<int>& ngram) {
    for (size_t i = 0; i <= sequence.size() - ngram.size(); ++i) {
        bool match = true;
        for (size_t j = 0; j < ngram.size(); ++j) {
            if (sequence[i + j] != ngram[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            return i;
        }
    }
    return INT_MAX;
}

int main() {
    // --- 1. Setup Host Data ---
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<> distrib(0, VOCAB_SIZE - 1);

    std::vector<int> h_sequence(SEQUENCE_LENGTH);
    for (int& val : h_sequence) {
        val = distrib(gen);
    }

    std::vector<int> h_ngram(NGRAM_SIZE);
    for (int& val : h_ngram) {
        val = distrib(gen);
    }

    // Inject the n-gram at a random position to ensure it's found
    std::uniform_int_distribution<> pos_distrib(0, SEQUENCE_LENGTH - NGRAM_SIZE);
    int inject_pos = pos_distrib(gen);
    std::copy(h_ngram.begin(), h_ngram.end(), h_sequence.begin() + inject_pos);

    std::vector<long long> h_output(NUM_BLOCKS, INT_MAX);

    // --- 2. Setup Device Data ---
    int *d_sequence, *d_ngram;
    long long* d_output;
    long long* d_final_result;

    CUDA_CHECK(cudaMalloc(&d_sequence, SEQUENCE_LENGTH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ngram, NGRAM_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, NUM_BLOCKS * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_final_result, sizeof(long long)));

    CUDA_CHECK(cudaMemcpy(d_sequence, h_sequence.data(), SEQUENCE_LENGTH * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ngram, h_ngram.data(), NGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    
    // --- 3. Run Kernel and Benchmark ---

    for (int i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
        test_wrapper_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_sequence, d_ngram, d_output, d_final_result, SEQUENCE_LENGTH);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Running benchmark for " << NUM_ITERATIONS << " iterations..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        test_wrapper_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_sequence, d_ngram, d_output, d_final_result, SEQUENCE_LENGTH);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    
    std::cout << "Total time for " << NUM_ITERATIONS << " iterations: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average time per iteration: " << elapsed.count() / NUM_ITERATIONS << " ms" << std::endl;

    // --- 4. Verify Result ---
    std::cout << "\nVerifying result..." << std::endl;
    long long h_final_result;
    CUDA_CHECK(cudaMemcpy(&h_final_result, d_final_result, sizeof(long long), cudaMemcpyDeviceToHost));
    
    long long expected_idx = find_first_ngram_cpu(h_sequence, h_ngram);
    
    std::cout << "N-gram was injected at position: " << inject_pos << std::endl;
    std::cout << "Earliest position found by CPU: " << expected_idx << std::endl;
    std::cout << "Earliest position found by GPU: " << h_final_result << std::endl;

    if (expected_idx == h_final_result) {
        std::cout << "SUCCESS: GPU result matches CPU result." << std::endl;
    } else {
        std::cout << "FAILURE: GPU result does not match CPU result." << std::endl;
    }

    // --- 5. Cleanup ---
    CUDA_CHECK(cudaFree(d_sequence));
    CUDA_CHECK(cudaFree(d_ngram));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_final_result));

    return 0;
} 