
## 安装

1. 确保您已安装 [CUDA](https://developer.nvidia.com/cuda-downloads) 和 [TensorRT](https://developer.nvidia.com/tensorrt)。
2. 克隆该库：

   ```bash
   git clone <repository-url>
   cd trt_spconv
   ```

3. 创建构建目录并编译：

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## 使用

### 依赖

- CUDA 10.0 或更高版本
- TensorRT 7.0 或更高版本
- PyTorch 1.7 或更高版本

### 示例

以下是如何使用 `SparseConvolution` 类的示例：

```cpp
#include "sparse_convolution.h"

// 创建稀疏卷积对象
trt_spconv::SparseConvolution sparse_conv(
    ndim,          // 维度
    in_channels,   // 输入通道数
    out_channels,  // 输出通道数
    kernel_size,   // 卷积核大小
    stride,        // 步幅
    padding,       // 填充
    dilation       // 膨胀
);

// 前向传播
trt_spconv::SparseConvTensor output = sparse_conv.forward(input);
```

## 贡献

欢迎任何形式的贡献！请提交问题、功能请求或拉取请求。

## 许可证

该项目采用 MIT 许可证，详细信息请查看 [LICENSE](LICENSE) 文件。