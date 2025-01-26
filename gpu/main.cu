#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaSetDevice(0);

    // メモリを適当に確保して解放
    const size_t size = 1 << 20;  // 1 MB
    void* d_ptr = nullptr;
    cudaMalloc(&d_ptr, size);
    cudaMemset(d_ptr, 0, size);
    cudaFree(d_ptr);

    // 別ターミナルでnvidia-smiで確認
    std::cout << "Check nvidia-smi in another terminal. Press Enter to exit." << std::endl;
    std::cin.get();

    return 0;
}
