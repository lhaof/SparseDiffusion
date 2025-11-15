#include "common_cuda.cu"
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <atomic>
__global__ void scatter_cuda_kernel(
        int total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const float *__restrict__ x, float *__restrict__ output,
        const int *__restrict__ activeIndices,
        const float *__restrict__ residual,
        int residualB, int residualC, int residualH, int residualW) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total)
        return;
    int t = index;
    int intraBw = t % S;
    t /= S;
    int intraBh = t % R;
    t /= R;
    int cc = t % C;
    t /= C;
    int ib = t % numActive, bb = t / numActive;
    int biH = (offsetH + activeIndices[ib << 1]) / strideH;
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
    int ww = biW + intraBw;
    if (ww >= W)
        return;
    auto p = bb * C * H * W + cc * H * W + hh * W + ww;
    auto z = x[index];
    z = binary_op_array_cuda<ADD>(
            residual, z,
            residualB, residualC, residualH, residualW,
            bb, cc, hh, ww);
    output[p] = z;
}


__global__ void scatter_avg_cuda_kernel(
        int total, int numActive,
        int B, int C, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const float *__restrict__ x, float *__restrict__ temp,
        int *__restrict__ count, // 用于存储写入次数
        const int *__restrict__ activeIndices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total)
        return;
    int t = index;
    int intraBw = t % S;
    t /= S;
    int intraBh = t % R;
    t /= R;
    int cc = t % C;
    t /= C;
    int ib = t % numActive, bb = t / numActive;
    int biH = (offsetH + activeIndices[ib << 1]) / strideH;
    int hh = biH + intraBh;
    if (hh >= H)
        return;
    int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
    int ww = biW + intraBw;
    if (ww >= W)
        return;
    auto p = bb * C * H * W + cc * H * W + hh * W + ww;
    auto z = x[index];
    atomicAdd(&temp[p], z); // 使用 atomicAdd 累加值
    atomicAdd(&count[p], 1);  // 使用 atomicAdd 增加计数
}

torch::Tensor scatter_cuda(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices, // Indices [N, 2], dim 0 is h, dim 1 is w,
        const torch::optional<torch::Tensor> &residual) {
    const int numActive = activeIndices.size(0);
    auto xData = x.data_ptr<float>();
    auto activeIndicesData = activeIndices.data_ptr<int>();

    const int C = x.size(1), R = x.size(2), S = x.size(3);
    const int B = y.size(0), H = y.size(2), W = y.size(3);
    auto output = y.clone();
    auto outputData = output.data_ptr<float>();

    const float *residualData = nullptr;
    int residualB = 0, residualC = 0, residualH = 0, residualW = 0;
    if (residual.has_value()) {
        assert(broadcastable(y, residual.value()));
        residualData = residual.value().data_ptr<float>();
        residualB = residual.value().size(0);
        residualC = residual.value().size(1);
        residualH = residual.value().size(2);
        residualW = residual.value().size(3);
    }

    const int total = x.numel();
    const dim3 blocks((total + threads - 1) / threads, 1);
    scatter_cuda_kernel<<<blocks, threads>>>(
            total, numActive,
            B, C, H, W,
            R, S,
            offsetH, offsetW,
            strideH, strideW,
            xData, outputData,
            activeIndicesData,
            residualData,
            residualB, residualC, residualH, residualW);

    return output;
}


 //将输入张量 x 中的值按照 activeIndices 指定的位置散射到输出张量 output 中
torch::Tensor scatter_avg_cuda(
    const torch::Tensor &x,
    const torch::Tensor &y,
    int offsetH, int offsetW,
    int strideH, int strideW,// Indices [N, 2], dim 0 is h, dim 1 is w,
    const torch::Tensor &activeIndices) {

    const int numActive = activeIndices.size(0);
    auto xData = x.data_ptr<float>();
    auto activeIndicesData = activeIndices.data_ptr<int>();

    const int C = x.size(1), R = x.size(2), S = x.size(3);
    const int B = y.size(0), H = y.size(2), W = y.size(3);
    
    auto output = y.clone();
    auto outputData = output.data_ptr<float>();

    auto temp = torch::zeros_like(y);
    auto tempData = temp.data_ptr<float>();

    // Create a count tensor to store the number of writes to each position
    auto count = torch::zeros_like(y, y.options().dtype(torch::kInt));
    auto countData = count.data_ptr<int>();

    const int total = x.numel();
    const dim3 blocks((total + threads - 1) / threads, 1);
    scatter_avg_cuda_kernel<<<blocks, threads>>>(
            total, numActive,
            B, C, H, W,
            R, S,
            offsetH, offsetW,
            strideH, strideW,
            xData, tempData,
            countData,  // 传递 count 张量给内核
            activeIndicesData);
    // 计算平均值
    auto countDataFloat = count.to(output.scalar_type());
    auto mask = countDataFloat > 0;
    output.masked_scatter_(mask, temp.masked_select(mask) / countDataFloat.masked_select(mask));
    return output;
}
