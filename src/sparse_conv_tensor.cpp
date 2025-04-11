#include "sparse_conv_tensor.h"
#include <torch/torch.h>
#include <stdexcept>

namespace trt_spconv {

SparseConvTensor::SparseConvTensor(
    torch::Tensor features,
    torch::Tensor indices,
    std::vector<int> spatial_shape,
    int batch_size,
    std::vector<int> grid)
    : features(features),
      indices(indices),
      spatial_shape(spatial_shape),
      batch_size(batch_size),
      grid(grid) {
    
    // 检查输入维度
    if (features.dim() != 2) {
        throw std::runtime_error("Features must be 2D tensor");
    }
    if (indices.dim() != 2) {
        throw std::runtime_error("Indices must be 2D tensor");
    }
    if (indices.size(1) != spatial_shape.size()) {
        throw std::runtime_error("Indices dimension mismatch with spatial shape");
    }
}

int SparseConvTensor::spatial_size() const {
    int size = 1;
    for (int dim : spatial_shape) {
        size *= dim;
    }
    return size;
}

std::shared_ptr<IndicePair> SparseConvTensor::find_indice_pair(const std::string& key) const {
    auto it = indice_pairs.find(key);
    if (it == indice_pairs.end()) {
        return nullptr;
    }
    return it->second;
}

torch::Tensor SparseConvTensor::dense() const {
    // 创建输出张量
    std::vector<int64_t> dense_shape = {batch_size};
    dense_shape.insert(dense_shape.end(), spatial_shape.begin(), spatial_shape.end());
    dense_shape.push_back(features.size(1));
    
    auto dense = torch::zeros(dense_shape, features.options());
    
    // 获取索引和特征数据
    auto indices_accessor = indices.accessor<int64_t, 2>();
    auto features_accessor = features.accessor<float, 2>();
    
    // 计算步长
    int64_t spatial_stride = spatial_shape[1] * spatial_shape[2];
    int64_t channel_stride = spatial_shape[0] * spatial_stride;
    int64_t batch_stride = features.size(1) * channel_stride;
    
    // 获取输出张量的原始指针
    float* dense_ptr = dense.data_ptr<float>();
    
    // 填充密集张量
    for (int64_t i = 0; i < indices.size(0); ++i) {
        int64_t batch_idx = indices_accessor[i][0];
        int64_t x = indices_accessor[i][1];
        int64_t y = indices_accessor[i][2];
        int64_t z = indices_accessor[i][3];
        
        // 计算输出位置
        int64_t output_idx = batch_idx * batch_stride + z * spatial_stride + y * spatial_shape[2] + x;
        
        // 复制特征
        for (int64_t c = 0; c < features.size(1); ++c) {
            dense_ptr[output_idx + c * channel_stride] = features_accessor[i][c];
        }
    }
    
    return dense;
}

float SparseConvTensor::sparsity() const {
    int64_t total_elements = batch_size;
    for (int dim : spatial_shape) {
        total_elements *= dim;
    }
    total_elements *= features.size(1);
    
    int64_t non_zero_elements = features.size(0) * features.size(1);
    
    return 1.0f - static_cast<float>(non_zero_elements) / total_elements;
}

} // namespace trt_spconv 