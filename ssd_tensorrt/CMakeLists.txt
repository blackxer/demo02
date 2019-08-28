project(ssd_tensorrt)
cmake_minimum_required(VERSION 3.15.1 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_BUILD_TYPE "Debug")

# opencv config
set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 使CUDA NVCC 编译器支持C++11特性
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)

LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

#nvcuvid
set(CUDA_nvcuvid_LIB_DIR /usr/lib/nvidia-418)
set(CUDA_nvcuvid_LIBRARY nvcuvid nvidia-ml)
link_directories(${CUDA_nvcuvid_LIB_DIR})
message(STATUS "nvcuvid: ${CUDA_nvcuvid_LIBRARY}")

#cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    libraries: ${CUDA_CUDA_LIBRARY}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/local/cuda-10.1/targets/x86_64-linux/include)

#tensorrt
set(tensorRT_INCLUDE_DIR /media/zw/DL/ly/software/TensorRT-5.1.5.0/include)
set(tensorRT_LIB_DIR /media/zw/DL/ly/software/TensorRT-5.1.5.0/lib)
set(tensorRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser nvonnxparser_runtime)
include_directories(${tensorRT_INCLUDE_DIR})
include_directories(/media/zw/DL/ly/software/TensorRT-5.1.5.0/samples/common)
link_directories(${tensorRT_LIB_DIR})


# torch config
set(CMAKE_PREFIX_PATH /media/zw/DL/ly/software/libtorch/libtorch)
find_package(Torch REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

include_directories(SRC_LIST)

aux_source_directory(. SRC_LIST)
set(logger_file "/media/zw/DL/ly/software/TensorRT-5.1.5.0/samples/common/logger.cpp")
add_executable(${PROJECT_NAME} ${SRC_LIST} ${logger_file})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${CUDA_nvcuvid_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${tensorRT_LIBS} ${OpenCV_LIBS})

