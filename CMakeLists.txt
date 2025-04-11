cmake_minimum_required(VERSION 3.10)
project(trt_spconv)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置TensorRT路径
set(TENSORRT_ROOT "/root/TensorRT-8.6.1.6")
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARIES 
    "${TENSORRT_ROOT}/lib/libnvinfer.so"
    "${TENSORRT_ROOT}/lib/libnvinfer_plugin.so"
    "${TENSORRT_ROOT}/lib/libnvonnxparser.so"
    "${TENSORRT_ROOT}/lib/libnvparsers.so"
)

# 设置Torch路径
set(TORCH_ROOT "/opt/conda/lib/python3.8/site-packages/torch")
set(TORCH_INCLUDE_DIRS 
    "${TORCH_ROOT}/include"
    "${TORCH_ROOT}/include/torch/csrc/api/include"
)
set(TORCH_LIBRARIES
    "${TORCH_ROOT}/lib/libtorch.so"
    "${TORCH_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_ROOT}/lib/libtorch_cuda.so"
    "${TORCH_ROOT}/lib/libc10.so"
    "${TORCH_ROOT}/lib/libc10_cuda.so"
)

# 查找必要的包
find_package(CUDA REQUIRED)

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
)

# 设置CUDA编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O3")

# 添加源文件
set(SOURCES
    src/sparse_convolution.cpp
    src/sparse_conv_tensor.cpp
    src/spconv_plugin.cpp
)

# 添加CUDA源文件
set(CUDA_SOURCES
    cuda/sparse_conv_kernel.cu
)

# 创建spconv动态库
cuda_add_library(spconv SHARED
    ${SOURCES}
    ${CUDA_SOURCES}
)

# 设置库的版本号
set_target_properties(spconv PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# 链接库
target_link_libraries(spconv
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
)

# 创建SCN动态库
cuda_add_library(scn SHARED
    ${SOURCES}
    ${CUDA_SOURCES}
)

# 设置库的版本号
set_target_properties(scn PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# 链接库
target_link_libraries(scn
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
)

# 安装
install(TARGETS spconv scn
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# 导出头文件
install(FILES
    include/sparse_convolution.h
    include/sparse_conv_tensor.h
    include/spconv_plugin.h
    DESTINATION include/trt_spconv
) 