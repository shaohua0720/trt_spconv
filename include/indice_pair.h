#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>

namespace trt_spconv {

class IndicePair {
public:
    IndicePair(torch::Tensor indices, int64_t num_out);
    
    // 获取输入索引
    torch::Tensor get_input_indices() const { return input_indices; }
    
    // 获取输出索引
    torch::Tensor get_output_indices() const { return output_indices; }
    
    // 获取输入索引数量
    int64_t get_num_input() const { return num_input; }
    
    // 获取输出索引数量
    int64_t get_num_output() const { return num_output; }
    
    // 获取索引对数量
    int64_t get_num_pairs() const { return num_pairs; }
    
    // 获取索引对
    std::pair<torch::Tensor, torch::Tensor> get_indice_pairs() const {
        return {input_indices, output_indices};
    }
    
    // 序列化
    std::vector<char> serialize() const;
    
    // 反序列化
    static std::shared_ptr<IndicePair> deserialize(const std::vector<char>& data);
    
private:
    torch::Tensor input_indices;  // 输入索引
    torch::Tensor output_indices; // 输出索引
    int64_t num_input;           // 输入索引数量
    int64_t num_output;          // 输出索引数量
    int64_t num_pairs;           // 索引对数量
};

} // namespace trt_spconv 