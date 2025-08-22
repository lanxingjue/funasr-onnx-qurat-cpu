#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <filesystem>
#include <fstream>

/**
 * 轻量 Logger（日志输出类），支持流式格式化，中文注释，全局线程安全
 */
class Logger {
public:
    enum Level { DEBUG, INFO, WARN, ERROR };
    static void SetLevel(Level level) { current_level_ = level; }

    // 信息日志流式拼接，兼容中文与多变量
    template <typename... Args>
    static void Info(const std::string& format, Args&&... args) {
        if (current_level_ <= INFO) {
            Print("INFO", format, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static void Error(const std::string& format, Args&&... args) {
        if (current_level_ <= ERROR) {
            Print("ERROR", format, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static void Warn(const std::string& format, Args&&... args) {
        if (current_level_ <= WARN) {
            Print("WARN", format, std::forward<Args>(args)...);
        }
    }

private:
    static Level current_level_;
    template <typename... Args>
    static void Print(const std::string& level, const std::string& format, Args&&... args) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

        std::string message = FormatString(format, std::forward<Args>(args)...);
        std::cout << "[" << oss.str() << "] [" << level << "] " << message << std::endl;
    }

    // 递归的 C++11 格式化字符串
    template <typename T>
    static std::string FormatString(const std::string& format, T&& value) {
        size_t pos = format.find("{}");
        if (pos == std::string::npos) return format;
        std::ostringstream oss;
        oss << value;
        std::string result = format;
        result.replace(pos, 2, oss.str());
        return result;
    }
    template <typename T, typename... Args>
    static std::string FormatString(const std::string& format, T&& value, Args&&... args) {
        std::string partial = FormatString(format, std::forward<T>(value));
        return FormatString(partial, std::forward<Args>(args)...);
    }
    static std::string FormatString(const std::string& format) { return format; }
};

/**
 * 高精度计时器
 */
class Timer {
public:
    Timer() { Reset(); }
    void Reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double ElapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * 音频文件工具类：支持扫描、读取、转单声道等，全程兼容 CPU
 */
class AudioFileReader {
public:
    struct AudioData {
        std::vector<float> samples; // 归一化数据
        int sample_rate = 16000;
        int channels = 1;
        double duration_seconds = 0.0;
        bool IsValid() const {
            return !samples.empty() && sample_rate > 0 && channels > 0;
        }
    };
    // 读取标准 WAV 文件
    static AudioData ReadWavFile(const std::string& file_path);
    // 扫描目录下所有 WAV 文件
    static std::vector<std::string> ScanWavFiles(const std::string& directory);
};

/**
 * 性能指标结构体，包含所有测试统计，兼容中文 ToString 输出
 */
struct PerformanceMetrics {
    double streaming_rtf = 0.0;
    double offline_rtf = 0.0;
    double two_pass_rtf = 0.0;
    double end_to_end_latency_ms = 0.0;
    double gpu_memory_gb = 0.0;

    double online_latency_ms = 0.0;
    double offline_refinement_ms = 0.0;
    double vad_processing_ms = 0.0;
    double punctuation_ms = 0.0;

    int concurrent_sessions = 0;
    double total_audio_processed_hours = 0.0;
    uint64_t total_requests = 0;
    uint64_t success_requests = 0;
    int test_files_count = 0;

    double GetSuccessRate() const {
        return total_requests > 0 ? (double(success_requests) / total_requests) * 100.0 : 100.0;
    }

    std::string ToString() const {
        std::ostringstream oss;
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        oss << " 🏆 FunASR CPU 性能测试报告\n";
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        oss << "📊 核心性能指标:\n";
        oss << " 流式识别RTF: " << std::fixed << std::setprecision(4) << streaming_rtf << "\n";
        oss << " 离线识别RTF: " << std::fixed << std::setprecision(4) << offline_rtf << "\n";
        oss << " 2Pass模式RTF: " << std::fixed << std::setprecision(4) << two_pass_rtf << "\n";
        oss << " 端到端延迟: " << std::fixed << std::setprecision(1) << end_to_end_latency_ms << "ms\n";
        oss << " 并发会话数: " << concurrent_sessions << "\n";
        oss << " GPU显存/CPU内存使用: " << std::fixed << std::setprecision(1) << gpu_memory_gb << "GB\n";
        oss << " 测试文件数: " << test_files_count << " 个WAV文件\n";
        oss << " 处理音频总时长: " << std::fixed << std::setprecision(1) << total_audio_processed_hours << " 小时\n";
        oss << " 成功率: " << std::fixed << std::setprecision(1) << GetSuccessRate() << "%\n";
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        return oss.str();
    }
};
