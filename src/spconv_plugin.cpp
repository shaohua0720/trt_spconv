#include "spconv_plugin.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace trt_spconv {

SparseConvolutionPlugin::SparseConvolutionPlugin(
    const std::vector<int>& kernel_size,
    const std::vector<int>& stride,
    const std::vector<int>& padding,
    const std::vector<int>& dilation,
    int groups,
    bool bias)
    : kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      groups_(groups),
      bias_(bias) {
    
    // 检查参数
    if (kernel_size.size() != 3 || stride.size() != 3 ||
        padding.size() != 3 || dilation.size() != 3) {
        throw std::runtime_error("All parameters must have size 3");
    }
}

SparseConvolutionPlugin::SparseConvolutionPlugin(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    
    // 从序列化数据中恢复参数
    memcpy(kernel_size_.data(), d, 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(stride_.data(), d, 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(padding_.data(), d, 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(dilation_.data(), d, 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(&groups_, d, sizeof(int));
    d += sizeof(int);
    
    memcpy(&bias_, d, sizeof(bool));
}

int SparseConvolutionPlugin::getNbOutputs() const noexcept {
    return 1;
}

nvinfer1::Dims SparseConvolutionPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept {
    // 输出维度与输入维度相同
    return inputs[0];
}

bool SparseConvolutionPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept {
    return type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR;
}

void SparseConvolutionPlugin::configureWithFormat(
    const nvinfer1::Dims* inputDims, int nbInputs,
    const nvinfer1::Dims* outputDims, int nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format,
    int maxBatchSize) noexcept {}

int SparseConvolutionPlugin::initialize() noexcept {
    return 0;
}

void SparseConvolutionPlugin::terminate() noexcept {
    // 清理资源
}

size_t SparseConvolutionPlugin::getWorkspaceSize(int maxBatchSize) const noexcept {
    return 0;
}

int SparseConvolutionPlugin::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    // 实现稀疏卷积的前向传播
    return 0;
}

size_t SparseConvolutionPlugin::getSerializationSize() const noexcept {
    return 3 * sizeof(int) * 4 + sizeof(int) + sizeof(bool);
}

void SparseConvolutionPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    
    memcpy(d, kernel_size_.data(), 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(d, stride_.data(), 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(d, padding_.data(), 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(d, dilation_.data(), 3 * sizeof(int));
    d += 3 * sizeof(int);
    
    memcpy(d, &groups_, sizeof(int));
    d += sizeof(int);
    
    memcpy(d, &bias_, sizeof(bool));
}

void SparseConvolutionPlugin::destroy() noexcept {
    delete this;
}

nvinfer1::IPluginV2* SparseConvolutionPlugin::clone() const noexcept {
    return new SparseConvolutionPlugin(kernel_size_, stride_, padding_, dilation_, groups_, bias_);
}

void SparseConvolutionPlugin::setPluginNamespace(const char* libNamespace) noexcept {
    namespace_ = libNamespace;
}

const char* SparseConvolutionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

const char* SparseConvolutionPlugin::getPluginType() const noexcept {
    return "SparseConvolution_TRT";
}

const char* SparseConvolutionPlugin::getPluginVersion() const noexcept {
    return "1";
}

// Plugin Creator 实现
const char* SparseConvolutionPluginCreator::getPluginName() const noexcept {
    return "SparseConvolution_TRT";
}

const char* SparseConvolutionPluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const nvinfer1::PluginFieldCollection* SparseConvolutionPluginCreator::getFieldNames() noexcept {
    return &field_collection_;
}

nvinfer1::IPluginV2* SparseConvolutionPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
    std::vector<int> kernel_size;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    int groups = 1;
    bool bias = false;
    
    for (int i = 0; i < fc->nbFields; i++) {
        const char* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "kernel_size")) {
            kernel_size = std::vector<int>(static_cast<const int*>(fc->fields[i].data),
                                         static_cast<const int*>(fc->fields[i].data) + fc->fields[i].length);
        } else if (!strcmp(attrName, "stride")) {
            stride = std::vector<int>(static_cast<const int*>(fc->fields[i].data),
                                    static_cast<const int*>(fc->fields[i].data) + fc->fields[i].length);
        } else if (!strcmp(attrName, "padding")) {
            padding = std::vector<int>(static_cast<const int*>(fc->fields[i].data),
                                     static_cast<const int*>(fc->fields[i].data) + fc->fields[i].length);
        } else if (!strcmp(attrName, "dilation")) {
            dilation = std::vector<int>(static_cast<const int*>(fc->fields[i].data),
                                      static_cast<const int*>(fc->fields[i].data) + fc->fields[i].length);
        } else if (!strcmp(attrName, "groups")) {
            groups = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "bias")) {
            bias = *static_cast<const bool*>(fc->fields[i].data);
        }
    }
    
    return new SparseConvolutionPlugin(kernel_size, stride, padding, dilation, groups, bias);
}

nvinfer1::IPluginV2* SparseConvolutionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
    return new SparseConvolutionPlugin(serialData, serialLength);
}

void SparseConvolutionPluginCreator::setPluginNamespace(const char* libNamespace) noexcept {
    namespace_ = libNamespace;
}

const char* SparseConvolutionPluginCreator::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

SparseConvolutionPluginCreator::SparseConvolutionPluginCreator() {
    // 添加插件属性
    plugin_attributes_.emplace_back(nvinfer1::PluginField("kernel_size", nullptr, nvinfer1::PluginFieldType::kINT32, 3));
    plugin_attributes_.emplace_back(nvinfer1::PluginField("stride", nullptr, nvinfer1::PluginFieldType::kINT32, 3));
    plugin_attributes_.emplace_back(nvinfer1::PluginField("padding", nullptr, nvinfer1::PluginFieldType::kINT32, 3));
    plugin_attributes_.emplace_back(nvinfer1::PluginField("dilation", nullptr, nvinfer1::PluginFieldType::kINT32, 3));
    plugin_attributes_.emplace_back(nvinfer1::PluginField("groups", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    plugin_attributes_.emplace_back(nvinfer1::PluginField("bias", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    
    field_collection_.nbFields = plugin_attributes_.size();
    field_collection_.fields = plugin_attributes_.data();
}

// 注册插件
REGISTER_TENSORRT_PLUGIN(SparseConvolutionPluginCreator);

} // namespace trt_spconv 