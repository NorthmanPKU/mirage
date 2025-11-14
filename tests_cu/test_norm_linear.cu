#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib> // For rand()

#include "../include/mirage/persistent_kernel/tasks/bfloat16.h"
#include "../include/mirage/config.h"
// #include "../include/mirage/persistent_kernel/tasks/norm_linear_old.cuh"
// #include "../include/mirage/persistent_kernel/tasks/norm_linear.cuh"

// 宏：用于检查CUDA API调用是否成功
#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                  \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// 辅助函数：用随机数据初始化主机内存
template <typename T>
void initialize_data(std::vector<T>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    }
}

// 内核启动器：这个 __global__ 函数是GPU执行的入口点
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE, int O_STRIDE, int K_PIPE_MAX>
__global__ void norm_linear_kernel_launcher(void const *input_ptr,
                                          void const *norm_weight_ptr,
                                          void const *weight_ptr, float eps,
                                          void *output_ptr) {
  kernel::norm_linear_task_impl<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

int main() {
    // std::cout << "Starting Nsight performance profiling run..." << std::endl;

    // --- 1. 定义问题维度和参数 ---
    // 您可以在这里修改这些值来测试不同的配置
    constexpr int BATCH_SIZE = 1;
    constexpr int OUTPUT_SIZE = 128;
    constexpr int REDUCTION_SIZE = 4096;
    constexpr int O_STRIDE = OUTPUT_SIZE;
    constexpr int K_PIPE_MAX = 2; // 必须与内核中的定义匹配
    constexpr float EPS = 1e-5f;
    using T = type::bfloat16_t;

    // --- 2. 在主机端分配和初始化输入数据 ---
    srand(123); // 使用固定的种子以保证可复现性
    std::vector<T> h_input(BATCH_SIZE * REDUCTION_SIZE);
    std::vector<T> h_norm_weight(REDUCTION_SIZE);
    std::vector<T> h_weight(REDUCTION_SIZE * OUTPUT_SIZE);

    initialize_data(h_input);
    initialize_data(h_norm_weight);
    initialize_data(h_weight);

    // --- 3. 在设备端分配内存 ---
    T *d_input, *d_norm_weight, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(T) * h_input.size()));
    CUDA_CHECK(cudaMalloc(&d_norm_weight, sizeof(T) * h_norm_weight.size()));
    CUDA_CHECK(cudaMalloc(&d_weight, sizeof(T) * h_weight.size()));
    // 输出缓冲区仍然需要被分配，因为内核会向它写入数据
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T) * BATCH_SIZE * OUTPUT_SIZE));

    // --- 4. 将输入数据从主机复制到设备 ---
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), sizeof(T) * h_input.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm_weight, h_norm_weight.data(), sizeof(T) * h_norm_weight.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), sizeof(T) * h_weight.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError()); // 检查启动是否立即产生了错误
    // --- 5. 配置并启动内核 ---
    dim3 gridDim(1); // 内核似乎是为单个线程块设计的
    dim3 blockDim(128); // 根据内核分析，warp总数为4，即128个线程

    // 动态计算内核所需的共享内存大小
    // using KernelSpec = kernel::NormLinearKernelSpec<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>;
    // size_t shared_mem_size = KernelSpec::SMEM_OFFSETS::SHARED_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_SIZE;

    std::cout << "\nKernel Configuration:" << std::endl;
    std::cout << "  Batch Size:     " << BATCH_SIZE << std::endl;
    std::cout << "  Output Size:    " << OUTPUT_SIZE << std::endl;
    std::cout << "  Reduction Size: " << REDUCTION_SIZE << std::endl;
    // std::cout << "  Shared Memory:  " << shared_mem_size << " bytes" << std::endl;
    // size_t shared_mem_size = 160 * 1024;
    size_t shared_mem_size = mirage::runtime::MAX_SHARE_MEMORY_SIZE;
    std::cout << "  Shared Memory:  " << shared_mem_size << " bytes" << std::endl;
    // std::cout << "  CUDA Arch:      " << mirage::runtime::__CUDA_ARCH__ << std::endl;
    // --- 6. 运行内核以进行性能分析 ---
    // 为了在Nsight中获得清晰的性能数据，即使只运行一次也可以。
    // 多次运行有助于测量更稳定的平均执行时间。
    int num_runs = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaGetLastError()); // 检查启动是否立即产生了错误

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    // Launcher persistent kernel
    cudaFuncSetAttribute(norm_linear_kernel_launcher<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        shared_mem_size);
    // 预热运行：第一次内核启动会包含JIT编译等一次性开销，应将其排除在计时之外
    norm_linear_kernel_launcher<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX><<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_norm_weight, d_weight, EPS, d_output);
    // CUDA_CHECK(cudaGetLastError()); // 检查启动是否立即产生了错误
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时运行
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        norm_linear_kernel_launcher<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX><<<gridDim, blockDim, shared_mem_size>>>(
            d_input, d_norm_weight, d_weight, EPS, d_output);
        // CUDA_CHECK(cudaGetLastError()); // 检查启动是否立即产生了错误
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "  Average kernel execution time over " << num_runs << " runs: "
              << milliseconds / num_runs << " ms" << std::endl << std::endl;

    // --- 7. Copy back the result and print one element to prevent dead code elimination ---
    std::vector<T> h_output(BATCH_SIZE * OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, sizeof(T) * h_output.size(), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; i++) {
    //     std::cout << "Result verification (element " << i << "): " << (float)h_output[i] << std::endl;
    // }
    std::cout << std::endl;


    // --- 8. 清理和释放资源 ---
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_norm_weight));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Profiling run complete. Use Nsight Systems to view the report." << std::endl;
    return 0;
}