#include "sparse_convolution.h"
#include "spconv_plugin.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <torch/torch.h>
#include <iterator>
#include <algorithm>

namespace trt_spconv {

SparseConvolution::SparseConvolution(
    int ndim,
    int in_channels,
    int out_channels,
    std::vector<int> kernel_size,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    int groups,
    bool bias,
    bool subm,
    std::string indice_key)
    : ndim_(ndim),
      in_channels_(in_channels),
      out_channels_(out_channels),
      groups_(groups),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      bias_(bias),
      subm_(subm),
      indice_key_(indice_key),
      runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr) {
    
    // 初始化权重
    int64_t weight_size = out_channels * in_channels / groups;
    for (int k : kernel_size) {
        weight_size *= k;
    }
    weight = torch::zeros({weight_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 初始化偏置
    if (bias) {
        bias_tensor = torch::zeros({out_channels}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    }
}

SparseConvolution::~SparseConvolution() {
    if (context_) {
        context_->destroy();
    }
    if (engine_) {
        engine_->destroy();
    }
    if (runtime_) {
        runtime_->destroy();
    }
}

void SparseConvolution::load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) {
    auto weight_it = state_dict.find("weight");
    if (weight_it != state_dict.end()) {
        weight = weight_it->second;
    }
    
    if (bias_) {
        auto bias_it = state_dict.find("bias");
        if (bias_it != state_dict.end()) {
            bias_tensor = bias_it->second;
        }
    }
}

void SparseConvolution::load_weights(const torch::Tensor& weights, const torch::Tensor& bias) {
    weight = weights;
    if (bias_) {
        bias_tensor = bias;
    }
}

void SparseConvolution::to(torch::Device device) {
    weight = weight.to(device);
    if (bias_) {
        bias_tensor = bias_tensor.to(device);
    }
}

bool SparseConvolution::loadEngine(const std::string& engine_path) {
    // 读取引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // 创建引擎
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        return false;
    }

    // 创建执行上下文
    context_ = engine_->createExecutionContext();
    if (!context_) {
        return false;
    }

    return true;
}

SparseConvTensor SparseConvolution::forward(const SparseConvTensor& input) {
    // 获取索引对
    auto indice_pairs = get_indice_pairs(input);
    
    // 计算输出空间形状
    std::vector<int> out_spatial_shape = get_conv_output_size(input.spatial_shape);
    
    // 分配输出特征内存
    auto output_features = torch::zeros(
        {input.indices.size(0), out_channels_},
        torch::dtype(torch::kFloat32).device(torch::kCUDA)
    );
    
    // 调用CUDA核函数
    dim3 block(256);
    dim3 grid((input.indices.size(0) + block.x - 1) / block.x);
    
    // 获取所有需要的指针
    float* input_features_ptr = input.features.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    int64_t* indice_pairs_ptr = indice_pairs[0].data_ptr<int64_t>();
    float* output_features_ptr = output_features.data_ptr<float>();
    int64_t num_indices = input.indices.size(0);
    
    // 使用正确的CUDA语法调用核函数
    void* args[] = {
        &input_features_ptr,
        &weight_ptr,
        &indice_pairs_ptr,
        &output_features_ptr,
        &num_indices,
        &in_channels_,
        &out_channels_,
        &groups_
    };
    
    cudaLaunchKernel((void*)sparseConvolutionKernel, grid, block, args, 0, 0);
    cudaDeviceSynchronize();
    
    // 添加偏置
    if (bias_) {
        output_features += bias_tensor;
    }
    
    // 创建输出张量
    return SparseConvTensor(
        output_features,
        input.indices,
        out_spatial_shape,
        input.batch_size,
        input.grid
    );
}

std::string SparseConvolution::generateCacheKey(const SparseConvTensor& input) const {
    std::string key;
    
    // 添加空间形状
    for (int dim : input.spatial_shape) {
        key += std::to_string(dim) + "_";
    }
    
    // 添加批次大小
    key += std::to_string(input.batch_size) + "_";
    
    // 添加网格信息
    if (!input.grid.empty()) {
        for (int val : input.grid) {
            key += std::to_string(val) + "_";
        }
    }
    
    // 添加卷积参数
    for (int k : kernel_size_) {
        key += std::to_string(k) + "_";
    }
    for (int s : stride_) {
        key += std::to_string(s) + "_";
    }
    for (int p : padding_) {
        key += std::to_string(p) + "_";
    }
    for (int d : dilation_) {
        key += std::to_string(d) + "_";
    }
    
    return key;
}

std::vector<torch::Tensor> SparseConvolution::getIndicePairsFromCache(const SparseConvTensor& input) {
    std::string key = generateCacheKey(input);
    auto it = indice_pairs_cache_.find(key);
    if (it != indice_pairs_cache_.end()) {
        return it->second;
    }
    return {};
}

void SparseConvolution::cacheIndicePairs(const SparseConvTensor& input, 
                                       const std::vector<torch::Tensor>& indice_pairs) {
    std::string key = generateCacheKey(input);
    indice_pairs_cache_[key] = indice_pairs;
}

std::vector<torch::Tensor> SparseConvolution::get_indice_pairs(const SparseConvTensor& input) {
    // 首先尝试从缓存获取
    auto cached_pairs = getIndicePairsFromCache(input);
    if (!cached_pairs.empty()) {
        return cached_pairs;
    }
    
    // 计算输出空间形状
    std::vector<int> out_spatial_shape = get_conv_output_size(input.spatial_shape);
    
    // 分配内存
    int64_t num_indices = input.indices.size(0);
    int64_t num_kernels = 1;
    for (int k : kernel_size_) {
        num_kernels *= k;
    }
    
    auto indice_pairs = torch::zeros({2, num_kernels, num_indices}, 
                                   torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    // 调用CUDA核函数
    dim3 block(256);
    dim3 grid((num_indices + block.x - 1) / block.x);
    
    // 获取所有需要的指针
    int64_t* input_indices_ptr = input.indices.data_ptr<int64_t>();
    int64_t* indice_pairs_ptr = indice_pairs.data_ptr<int64_t>();
    const int* spatial_shape_ptr = input.spatial_shape.data();
    const int* out_spatial_shape_ptr = out_spatial_shape.data();
    const int* kernel_size_ptr = kernel_size_.data();
    const int* stride_ptr = stride_.data();
    const int* padding_ptr = padding_.data();
    const int* dilation_ptr = dilation_.data();
    
    // 使用正确的CUDA语法调用核函数
    void* args[] = {
        &input_indices_ptr,
        &indice_pairs_ptr,
        &spatial_shape_ptr,
        &out_spatial_shape_ptr,
        &kernel_size_ptr,
        &stride_ptr,
        &padding_ptr,
        &dilation_ptr,
        &num_indices,
        &num_kernels
    };
    
    cudaLaunchKernel((void*)generateIndicePairsKernel, grid, block, args, 0, 0);
    cudaDeviceSynchronize();
    
    // 将结果存入缓存
    cacheIndicePairs(input, {indice_pairs});
    
    return {indice_pairs};
}

std::vector<int> SparseConvolution::get_conv_output_size(const std::vector<int>& spatial_shape) {
    std::vector<int> out_shape(spatial_shape.size());
    for (size_t i = 0; i < spatial_shape.size(); ++i) {
        out_shape[i] = (spatial_shape[i] + 2 * padding_[i] - dilation_[i] * (kernel_size_[i] - 1) - 1) / stride_[i] + 1;
    }
    return out_shape;
}

std::vector<int> SparseConvolution::get_deconv_output_size(const std::vector<int>& spatial_shape) {
    std::vector<int> out_shape(spatial_shape.size());
    for (size_t i = 0; i < spatial_shape.size(); ++i) {
        out_shape[i] = (spatial_shape[i] - 1) * stride_[i] - 2 * padding_[i] + dilation_[i] * (kernel_size_[i] - 1) + 1;
    }
    return out_shape;
}

} // namespace trt_spconv 