#include "utils.h"

// 修复：将静态变量定义移到cpp文件
Logger::Level Logger::current_level_ = Logger::INFO;
