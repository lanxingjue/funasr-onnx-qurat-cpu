cmake_minimum_required(VERSION 3.16)
project(FunASR_CPU_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找pybind11
find_package(pybind11 REQUIRED)

# 源文件
add_executable(funasr_cpu_engine 
    src/main.cpp
    src/funasr_engine.cpp
    src/utils.cpp
)

# 链接库（不再依赖 GPU/CUDA 库，仅用 Python3 + pybind11 + 标准库）
target_link_libraries(funasr_cpu_engine 
    ${Python3_LIBRARIES}
    pybind11::embed
    stdc++fs
)

# 编译选项
target_compile_options(funasr_cpu_engine PRIVATE
    -O3 -Wall -Wno-attributes
)
