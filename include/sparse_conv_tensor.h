#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace trt_spconv {

class IndicePair {
public:
    torch::Tensor indices;
    torch::Tensor num_activations;
};

class SparseConvTensor {
public:
    SparseConvTensor(torch::Tensor features,
                    torch::Tensor indices,
                    std::vector<int> spatial_shape,
                    int batch_size,
                    std::vector<int> grid);
    
    // 获取空间大小
    int spatial_size() const;
    
    // 查找索引对
    std::shared_ptr<IndicePair> find_indice_pair(const std::string& key) const;
    
    // 转换为密集张量
    torch::Tensor dense() const;
    
    // 计算稀疏度
    float sparsity() const;
    
    // 成员变量
    torch::Tensor features;
    torch::Tensor indices;
    std::vector<int> spatial_shape;
    int batch_size;
    std::vector<int> grid;
    std::unordered_map<std::string, std::shared_ptr<IndicePair>> indice_pairs;
};

} // namespace trt_spconv 