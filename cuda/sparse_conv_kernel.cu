#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cub/cub.cuh>

namespace trt_spconv {
namespace kernel {

// 计算输出索引
__device__ int calculate_output_index(
    const int* indices,
    const int* spatial_shape,
    const int* kernel_size,
    const int* stride,
    const int* padding,
    const int* dilation,
    int ndim,
    int batch_idx
) {
    int output_idx = 0;
    int stride_accum = 1;
    
    for (int i = ndim - 1; i >= 0; --i) {
        int input_idx = indices[i];
        int output_pos = (input_idx + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1;
        output_idx += output_pos * stride_accum;
        stride_accum *= spatial_shape[i];
    }
    
    return output_idx + batch_idx * stride_accum;
}

// 生成索引对
__global__ void generateIndicePairs(
    const int* indices,
    int num_indices,
    int batch_size,
    const int* spatial_shape,
    const int* kernel_size,
    const int* stride,
    const int* padding,
    const int* dilation,
    int* out_indices,
    int* indice_pairs,
    int* indice_pair_num,
    int max_num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    int ndim = 3; // 3D卷积
    int kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2];
    
    // 计算当前索引的batch索引
    int batch_idx = idx / (num_indices / batch_size);
    
    // 计算输出索引
    int output_idx = calculate_output_index(
        indices + idx * ndim,
        spatial_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        ndim,
        batch_idx
    );
    
    // 存储输出索引
    out_indices[idx] = output_idx;
    
    // 生成卷积核的索引对
    for (int k = 0; k < kernel_volume; ++k) {
        int k_idx = idx * kernel_volume + k;
        if (k_idx >= max_num_pairs) return;
        
        // 计算卷积核位置
        int kx = k % kernel_size[0];
        int ky = (k / kernel_size[0]) % kernel_size[1];
        int kz = k / (kernel_size[0] * kernel_size[1]);
        
        // 计算输入索引
        int input_idx = idx;
        int output_pos = output_idx;
        
        // 存储索引对
        indice_pairs[k_idx * 2] = input_idx;
        indice_pairs[k_idx * 2 + 1] = output_pos;
        indice_pair_num[k_idx] = 1;
    }
}

// 稀疏卷积
__global__ void sparseConvolution(
    const float* features,
    const float* filters,
    const int* indices,
    const int* indice_pairs,
    const int* indice_pair_num,
    float* output,
    int out_size,
    int in_channels,
    int out_channels,
    int kernel_volume
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    
    // 获取当前输出位置的索引对
    int pair_idx = idx * kernel_volume;
    int num_pairs = indice_pair_num[idx];
    
    // 对每个输入特征进行卷积
    for (int k = 0; k < num_pairs; ++k) {
        int input_idx = indice_pairs[pair_idx + k * 2];
        int output_pos = indice_pairs[pair_idx + k * 2 + 1];
        
        // 对每个输出通道
        for (int oc = 0; oc < out_channels; ++oc) {
            float sum = 0.0f;
            
            // 对每个输入通道
            for (int ic = 0; ic < in_channels; ++ic) {
                // 获取输入特征和滤波器权重
                float feature = features[input_idx * in_channels + ic];
                float weight = filters[oc * in_channels * kernel_volume + k * in_channels + ic];
                
                sum += feature * weight;
            }
            
            // 累加到输出
            atomicAdd(&output[output_pos * out_channels + oc], sum);
        }
    }
}

// 特征聚合
__global__ void featureAggregation(
    const float* features,
    const int* indices,
    float* output,
    int num_features,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    
    int output_idx = indices[idx];
    
    // 对每个通道进行聚合
    for (int c = 0; c < num_channels; ++c) {
        float feature = features[idx * num_channels + c];
        atomicAdd(&output[output_idx * num_channels + c], feature);
    }
}

__global__ void generateIndicePairsKernel(
    const int64_t* indices,
    int64_t* indice_pairs,
    const int* spatial_shape,
    const int* out_spatial_shape,
    const int* kernel_size,
    const int* stride,
    const int* padding,
    const int* dilation,
    int64_t num_indices,
    int64_t num_kernels) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;
    
    // 获取当前点的坐标
    int64_t batch_idx = indices[idx * 4];
    int64_t z = indices[idx * 4 + 1];
    int64_t y = indices[idx * 4 + 2];
    int64_t x = indices[idx * 4 + 3];
    
    // 计算输出空间中的位置
    int64_t out_z = (z + padding[0] - dilation[0] * (kernel_size[0] - 1) / 2) / stride[0];
    int64_t out_y = (y + padding[1] - dilation[1] * (kernel_size[1] - 1) / 2) / stride[1];
    int64_t out_x = (x + padding[2] - dilation[2] * (kernel_size[2] - 1) / 2) / stride[2];
    
    // 检查是否在输出空间范围内
    if (out_z < 0 || out_z >= out_spatial_shape[0] ||
        out_y < 0 || out_y >= out_spatial_shape[1] ||
        out_x < 0 || out_x >= out_spatial_shape[2]) {
        return;
    }
    
    // 计算输出索引
    int64_t out_idx = batch_idx * out_spatial_shape[0] * out_spatial_shape[1] * out_spatial_shape[2] +
                     out_z * out_spatial_shape[1] * out_spatial_shape[2] +
                     out_y * out_spatial_shape[2] +
                     out_x;
    
    // 生成卷积核的索引对
    for (int64_t kz = 0; kz < kernel_size[0]; ++kz) {
        for (int64_t ky = 0; ky < kernel_size[1]; ++ky) {
            for (int64_t kx = 0; kx < kernel_size[2]; ++kx) {
                // 计算卷积核中的位置
                int64_t kernel_idx = kz * kernel_size[1] * kernel_size[2] + ky * kernel_size[2] + kx;
                
                // 计算输入空间中的位置
                int64_t in_z = out_z * stride[0] - padding[0] + kz * dilation[0];
                int64_t in_y = out_y * stride[1] - padding[1] + ky * dilation[1];
                int64_t in_x = out_x * stride[2] - padding[2] + kx * dilation[2];
                
                // 检查是否在输入空间范围内
                if (in_z < 0 || in_z >= spatial_shape[0] ||
                    in_y < 0 || in_y >= spatial_shape[1] ||
                    in_x < 0 || in_x >= spatial_shape[2]) {
                    continue;
                }
                
                // 计算输入索引
                int64_t in_idx = batch_idx * spatial_shape[0] * spatial_shape[1] * spatial_shape[2] +
                               in_z * spatial_shape[1] * spatial_shape[2] +
                               in_y * spatial_shape[2] +
                               in_x;
                
                // 存储索引对
                indice_pairs[kernel_idx * num_indices * 2 + idx * 2] = in_idx;
                indice_pairs[kernel_idx * num_indices * 2 + idx * 2 + 1] = out_idx;
            }
        }
    }
}

__global__ void sparseConvolutionKernel(
    const float* features,
    const float* weight,
    const int64_t* indice_pairs,
    float* output,
    int64_t num_features,
    int in_channels,
    int out_channels,
    int groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    
    // 获取当前特征对应的输入和输出索引
    int64_t in_idx = indice_pairs[idx * 2];
    int64_t out_idx = indice_pairs[idx * 2 + 1];
    
    // 计算每个组的通道数
    int channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    // 对每个组进行卷积
    for (int g = 0; g < groups; ++g) {
        // 计算当前组的输入和输出通道范围
        int in_start = g * channels_per_group;
        int out_start = g * out_channels_per_group;
        
        // 对每个输出通道
        for (int oc = 0; oc < out_channels_per_group; ++oc) {
            float sum = 0.0f;
            
            // 对每个输入通道
            for (int ic = 0; ic < channels_per_group; ++ic) {
                // 获取输入特征和权重
                float feature = features[in_idx * in_channels + in_start + ic];
                float w = weight[(out_start + oc) * in_channels + in_start + ic];
                
                // 累加乘积
                sum += feature * w;
            }
            
            // 使用原子操作累加到输出
            atomicAdd(&output[out_idx * out_channels + out_start + oc], sum);
        }
    }
}

__global__ void sparseConvolutionBackwardKernel(
    const float* features,
    const float* weight,
    const float* grad_output,
    float* grad_features,
    float* grad_weight,
    const int64_t* indice_pairs,
    int64_t num_features,
    int in_channels,
    int out_channels,
    int groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;
    
    // 获取当前特征对应的输入和输出索引
    int64_t in_idx = indice_pairs[idx * 2];
    int64_t out_idx = indice_pairs[idx * 2 + 1];
    
    // 计算每个组的通道数
    int channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    // 对每个组进行反向传播
    for (int g = 0; g < groups; ++g) {
        // 计算当前组的输入和输出通道范围
        int in_start = g * channels_per_group;
        int out_start = g * out_channels_per_group;
        
        // 对每个输出通道
        for (int oc = 0; oc < out_channels_per_group; ++oc) {
            float grad_out = grad_output[out_idx * out_channels + out_start + oc];
            
            // 对每个输入通道
            for (int ic = 0; ic < channels_per_group; ++ic) {
                // 计算特征梯度
                float w = weight[(out_start + oc) * in_channels + in_start + ic];
                atomicAdd(&grad_features[in_idx * in_channels + in_start + ic],
                         grad_out * w);
                
                // 计算权重梯度
                float f = features[in_idx * in_channels + in_start + ic];
                atomicAdd(&grad_weight[(out_start + oc) * in_channels + in_start + ic],
                         grad_out * f);
            }
        }
    }
}

} // namespace kernel
} // namespace trt_spconv 