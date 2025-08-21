cmake_minimum_required(VERSION 3.16)
project(FunASR_GPU_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找pybind11
find_package(pybind11 REQUIRED)

# 源文件 (添加utils.cpp)
add_executable(funasr_gpu_engine 
    src/main.cpp
    src/funasr_engine.cpp
    src/utils.cpp
)

# 链接库
target_link_libraries(funasr_gpu_engine 
    ${Python3_LIBRARIES}
    pybind11::embed
    stdc++fs
)

# 编译选项
target_compile_options(funasr_gpu_engine PRIVATE 
    -O3 -Wall -Wno-attributes
)
