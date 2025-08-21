cmake_minimum_required(VERSION 3.16)
project(FunASR_GPU_Engine)

set(CMAKE_CXX_STANDARD 17)

# 查找依赖
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 源文件
add_executable(funasr_gpu_engine 
    src/main.cpp
    src/funasr_engine.cpp
)

# 链接库
target_link_libraries(funasr_gpu_engine 
    ${Python3_LIBRARIES}
    pybind11::embed
    stdc++fs  # C++17 filesystem
)

# 编译选项
target_compile_options(funasr_gpu_engine PRIVATE -O3 -Wall)
