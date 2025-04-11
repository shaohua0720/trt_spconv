#pragma once

#include <NvInfer.h>
#include <torch/torch.h>
#include "sparse_conv_tensor.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <map>

namespace trt_spconv {

// CUDA核函数声明
extern "C" {
    void sparseConvolutionKernel(
        const float* input_features,
        const float* weight,
        const int64_t* indice_pairs,
        float* output_features,
        int64_t num_indices,
        int in_channels,
        int out_channels,
        int groups
    );

    void generateIndicePairsKernel(
        const int64_t* indices,
        int64_t* indice_pairs,
        const int* spatial_shape,
        const int* out_spatial_shape,
        const int* kernel_size,
        const int* stride,
        const int* padding,
        const int* dilation,
        int64_t num_indices,
        int64_t num_kernels
    );
}

class SparseConvolution {
public:
    SparseConvolution(int ndim,
                     int in_channels,
                     int out_channels,
                     std::vector<int> kernel_size,
                     std::vector<int> stride,
                     std::vector<int> padding,
                     std::vector<int> dilation,
                     int groups = 1,
                     bool bias = true,
                     bool subm = false,
                     std::string indice_key = "");

    ~SparseConvolution();

    // 前向传播
    SparseConvTensor forward(const SparseConvTensor& input);

    // 加载模型参数
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict);

    // 加载权重
    void load_weights(const torch::Tensor& weights, const torch::Tensor& bias = torch::Tensor());

    // 设备迁移
    void to(torch::Device device);

    // TensorRT引擎加载
    bool loadEngine(const std::string& engine_path);

private:
    // 获取卷积输出大小
    std::vector<int> get_conv_output_size(const std::vector<int>& spatial_shape);

    // 获取反卷积输出大小
    std::vector<int> get_deconv_output_size(const std::vector<int>& spatial_shape);

    // 获取索引对
    std::vector<torch::Tensor> get_indice_pairs(const SparseConvTensor& input);

    // 从缓存获取索引对
    std::vector<torch::Tensor> getIndicePairsFromCache(const SparseConvTensor& input);

    // 缓存索引对
    void cacheIndicePairs(const SparseConvTensor& input, 
                         const std::vector<torch::Tensor>& indice_pairs);

    // 生成缓存键
    std::string generateCacheKey(const SparseConvTensor& input) const;

    // 成员变量
    int ndim_;
    int in_channels_;
    int out_channels_;
    int groups_;
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
    std::vector<int> dilation_;
    bool bias_;
    bool subm_;
    std::string indice_key_;

    // TensorRT相关
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    // 权重和偏置
    torch::Tensor weight;
    torch::Tensor bias_tensor;

    // 索引对缓存
    std::unordered_map<std::string, std::vector<torch::Tensor>> indice_pairs_cache_;
};

} // namespace trt_spconv 