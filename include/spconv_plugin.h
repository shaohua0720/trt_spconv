#pragma once

#include <NvInfer.h>
#include <string>
#include <vector>

namespace trt_spconv {

class SpconvPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    SpconvPlugin(const std::vector<int>& kernel_size,
                const std::vector<int>& stride,
                const std::vector<int>& padding,
                const std::vector<int>& dilation,
                int groups);

    SpconvPlugin(const void* data, size_t length);

    ~SpconvPlugin() override = default;

    // IPluginV2DynamicExt 方法
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                        int nbInputs,
                        const nvinfer1::DynamicPluginTensorDesc* out,
                        int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                           int nbInputs,
                           const nvinfer1::PluginTensorDesc* outputs,
                           int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override;

    // IPluginV2Ext 方法
    nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const noexcept override;
    void attachToContext(cudnnContext* cudnn,
                        cublasContext* cublas,
                        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

    // IPluginV2 方法
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
    std::vector<int> dilation_;
    int groups_;
    std::string namespace_;
};

class SpconvPluginCreator : public nvinfer1::IPluginCreator {
public:
    SpconvPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection field_collection_;
    static std::vector<nvinfer1::PluginField> fields_;
    std::string namespace_;
};

class SparseConvolutionPlugin : public nvinfer1::IPluginV2 {
public:
    SparseConvolutionPlugin(
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation,
        int groups,
        bool bias);
    
    SparseConvolutionPlugin(const void* data, size_t length);
    
    // IPluginV2 方法
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override;
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override;
    void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                           const nvinfer1::Dims* outputDims, int nbOutputs,
                           nvinfer1::DataType type, nvinfer1::PluginFormat format,
                           int maxBatchSize) noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
    int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
               void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    nvinfer1::IPluginV2* clone() const noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    
private:
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
    std::vector<int> dilation_;
    int groups_;
    bool bias_;
    std::string namespace_;
};

class SparseConvolutionPluginCreator : public nvinfer1::IPluginCreator {
public:
    SparseConvolutionPluginCreator();
    
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    
private:
    std::string namespace_;
    nvinfer1::PluginFieldCollection field_collection_;
    std::vector<nvinfer1::PluginField> plugin_attributes_;
};

} // namespace trt_spconv 