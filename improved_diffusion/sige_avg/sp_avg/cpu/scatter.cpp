#include "common_cpu.cpp"
#include <torch/extension.h>
#include <atomic>
void scatter_kernel(
        int numActive,
        int B, int C, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const float *x, float *output,
        const int *activeIndices,
        const float *residual,
        int residualB, int residualC, int residualH, int residualW) {
#pragma omp parallel for collapse(3)
    for (int bb = 0; bb < B; ++bb)
        for (int ib = 0; ib < numActive; ++ib)
            for (int cc = 0; cc < C; ++cc) {
                int biH = (offsetH + activeIndices[ib << 1]) / strideH;
                int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
                for (int intraBh = 0; intraBh < R; ++intraBh) {
                    int hh = biH + intraBh;
                    if (hh >= H)
                        break;
                    for (int intraBw = 0; intraBw < S; ++intraBw) {
                        int ww = biW + intraBw;
                        if (ww >= W)
                            break;
                        int index = (bb * numActive + ib) * C * R * S + cc * R * S + intraBh * S + intraBw;
                        auto p = bb * C * H * W + cc * H * W + hh * W + ww;
                        auto z = x[index];
                        z = binary_op_array<ADD>(
                                residual, z,
                                residualB, residualC, residualH, residualW,
                                bb, cc, hh, ww);
                        output[p] = z;
                    }
                }
            }
}

void scatter_avg_kernel(
        int numActive,
        int B, int C, int H, int W,
        int R, int S,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const float *x, float *output,
        int *count,
        const int *activeIndices) {
#pragma omp parallel for collapse(3)
    for (int bb = 0; bb < B; ++bb)
        for (int ib = 0; ib < numActive; ++ib)
            for (int cc = 0; cc < C; ++cc) {
                int biH = (offsetH + activeIndices[ib << 1]) / strideH;
                int biW = (offsetW + activeIndices[ib << 1 | 1]) / strideW;
                for (int intraBh = 0; intraBh < R; ++intraBh) {
                    int hh = biH + intraBh;
                    if (hh >= H)
                        break;
                    for (int intraBw = 0; intraBw < S; ++intraBw) {
                        int ww = biW + intraBw;
                        if (ww >= W)
                            break;
                        int index = (bb * numActive + ib) * C * R * S + cc * R * S + intraBh * S + intraBw;
                        auto p = bb * C * H * W + cc * H * W + hh * W + ww;
                        auto z = x[index];
                        // atomicAdd(&output[p], z); // 使用 atomicAdd 累加值
                        // atomicAdd(&count[p], 1);
                        z = binary_op_array<ADD>(
                                residual, z,
                                residualB, residualC, residualH, residualW,
                                bb, cc, hh, ww);
                        output[p] = z;
                        
                        std::atomic<float> output_atomic(output[p]);
                        output_atomic.fetch_add(z, std::memory_order_relaxed);
                        output[p] = output_atomic.load();

                        std::atomic<int> count_atomic(count[p]);
                        count_atomic.fetch_add(1, std::memory_order_relaxed);
                        count[p] = count_atomic.load();
                    }
                }
            }
}

torch::Tensor scatter_cpu(
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

    scatter_kernel(
            numActive,
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

torch::Tensor scatter_avg_cpu(
        const torch::Tensor &x,
        const torch::Tensor &y,
        int offsetH, int offsetW,
        int strideH, int strideW,
        const torch::Tensor &activeIndices) {
    const int numActive = activeIndices.size(0);
    auto xData = x.data_ptr<float>();
    auto activeIndicesData = activeIndices.data_ptr<int>();

    const int C = x.size(1), R = x.size(2), S = x.size(3);
    const int B = y.size(0), H = y.size(2), W = y.size(3);
    auto output = y.clone();
    auto outputData = output.data_ptr<float>();

    auto count = torch::zeros_like(y, y.options().dtype(torch::kInt));
    auto countData = count.data_ptr<int>();

    scatter_avg_kernel(
            numActive,
            B, C, H, W,
            R, S,
            offsetH, offsetW,
            strideH, strideW,
            xData, outputData,
            countData,
            activeIndicesData);
    auto countDataFloat = count.to(output.scalar_type());
    auto mask = countDataFloat > 0;
    output.masked_scatter_(mask, output.masked_select(mask) / countDataFloat.masked_select(mask));
    return output;
}
