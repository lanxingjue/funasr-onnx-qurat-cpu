#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>

/**
 * 简化的日志工具
 */
class Logger {
public:
    enum Level { DEBUG, INFO, WARN, ERROR };
    
    static void SetLevel(Level level) { current_level_ = level; }
    
    template<typename... Args>
    static void Info(const std::string& format, Args&&... args) {
        if (current_level_ <= INFO) {
            Print("INFO", format, std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    static void Error(const std::string& format, Args&&... args) {
        if (current_level_ <= ERROR) {
            Print("ERROR", format, std::forward<Args>(args)...);
        }
    }
    
    template<typename... Args>
    static void Warn(const std::string& format, Args&&... args) {
        if (current_level_ <= WARN) {
            Print("WARN", format, std::forward<Args>(args)...);
        }
    }

private:
    // 修复：将定义移到cpp文件，这里只声明
    static Level current_level_;
    
    template<typename... Args>
    static void Print(const std::string& level, const std::string& format, Args&&... args) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        
        std::string message = FormatString(format, std::forward<Args>(args)...);
        std::cout << "[" << oss.str() << "] [" << level << "] " << message << std::endl;
    }
    
    template<typename T>
    static std::string FormatString(const std::string& format, T&& value) {
        size_t pos = format.find("{}");
        if (pos == std::string::npos) return format;
        
        std::ostringstream oss;
        oss << value;
        std::string result = format;
        result.replace(pos, 2, oss.str());
        return result;
    }
    
    template<typename T, typename... Args>
    static std::string FormatString(const std::string& format, T&& value, Args&&... args) {
        std::string partial = FormatString(format, std::forward<T>(value));
        return FormatString(partial, std::forward<Args>(args)...);
    }
    
    static std::string FormatString(const std::string& format) {
        return format;
    }
};

// 修复：移除这行定义，避免多重定义
// Logger::Level Logger::current_level_ = Logger::INFO;

/**
 * 高精度计时器
 */
class Timer {
public:
    Timer() { Reset(); }
    
    void Reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double ElapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            end - start_).count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * WAV音频文件读取器
 * 支持16位PCM格式的WAV文件
 */
class AudioFileReader {
public:
    struct AudioData {
        std::vector<float> samples;     // 音频样本数据 (归一化到[-1,1])
        int sample_rate = 16000;        // 采样率
        int channels = 1;               // 声道数
        double duration_seconds = 0.0;  // 音频时长
        
        bool IsValid() const {
            return !samples.empty() && sample_rate > 0 && channels > 0;
        }
    };
    
    /**
     * 读取WAV文件
     */
    static AudioData ReadWavFile(const std::string& file_path) {
        AudioData audio_data;
        
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            Logger::Error("无法打开音频文件: {}", file_path);
            return audio_data;
        }
        
        try {
            // 读取WAV头部 (44字节)
            char header[44];
            file.read(header, 44);
            
            if (file.gcount() != 44) {
                Logger::Error("WAV文件头部读取失败: {}", file_path);
                return audio_data;
            }
            
            // 验证WAV文件格式
            if (std::string(header, 4) != "RIFF" || std::string(header + 8, 4) != "WAVE") {
                Logger::Error("不是有效的WAV文件: {}", file_path);
                return audio_data;
            }
            
            // 解析WAV头部信息
            audio_data.channels = *reinterpret_cast<short*>(header + 22);
            audio_data.sample_rate = *reinterpret_cast<int*>(header + 24);
            int bits_per_sample = *reinterpret_cast<short*>(header + 34);
            int data_size = *reinterpret_cast<int*>(header + 40);
            
            Logger::Info("读取WAV: {}, 采样率={}Hz, 声道={}, 位深={}bit, 数据大小={}bytes", 
                        std::filesystem::path(file_path).filename().string(),
                        audio_data.sample_rate, audio_data.channels, bits_per_sample, data_size);
            
            // 目前只支持16位PCM格式
            if (bits_per_sample != 16) {
                Logger::Error("暂不支持{}位音频，请转换为16位PCM格式: {}", bits_per_sample, file_path);
                return audio_data;
            }
            
            // 读取音频数据
            std::vector<int16_t> raw_data(data_size / 2);  // 16位 = 2字节
            file.read(reinterpret_cast<char*>(raw_data.data()), data_size);
            
            if (file.gcount() != data_size) {
                Logger::Error("音频数据读取不完整: {}", file_path);
                return audio_data;
            }
            
            // 转换为float并归一化到[-1, 1]
            audio_data.samples.resize(raw_data.size());
            for (size_t i = 0; i < raw_data.size(); ++i) {
                audio_data.samples[i] = static_cast<float>(raw_data[i]) / 32768.0f;
            }
            
            // 如果是立体声，转换为单声道 (取平均值)
            if (audio_data.channels == 2) {
                std::vector<float> mono_samples;
                mono_samples.reserve(audio_data.samples.size() / 2);
                
                for (size_t i = 0; i < audio_data.samples.size(); i += 2) {
                    float left = audio_data.samples[i];
                    float right = audio_data.samples[i + 1];
                    mono_samples.push_back((left + right) / 2.0f);
                }
                
                audio_data.samples = std::move(mono_samples);
                audio_data.channels = 1;
                Logger::Info("立体声转单声道完成");
            }
            
            // 重采样到16kHz (如果需要)
            if (audio_data.sample_rate != 16000) {
                Logger::Warn("音频采样率为{}Hz，FunASR需要16kHz，建议预处理音频文件", audio_data.sample_rate);
            }
            
            // 计算音频时长
            audio_data.duration_seconds = static_cast<double>(audio_data.samples.size()) / audio_data.sample_rate;
            
            Logger::Info("音频读取成功: 时长={:.2f}秒, 样本数={}", 
                        audio_data.duration_seconds, audio_data.samples.size());
            
        } catch (const std::exception& e) {
            Logger::Error("读取WAV文件异常: {} - {}", file_path, e.what());
        }
        
        return audio_data;
    }
    
    /**
     * 扫描目录中的所有WAV文件
     */
    static std::vector<std::string> ScanWavFiles(const std::string& directory) {
        std::vector<std::string> wav_files;
        
        try {
            if (!std::filesystem::exists(directory)) {
                Logger::Error("音频目录不存在: {}", directory);
                return wav_files;
            }
            
            for (const auto& entry : std::filesystem::directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    std::string file_path = entry.path().string();
                    std::string extension = entry.path().extension().string();
                    
                    // 转换为小写进行比较
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                    
                    if (extension == ".wav") {
                        wav_files.push_back(file_path);
                    }
                }
            }
            
            // 按文件名排序
            std::sort(wav_files.begin(), wav_files.end());
            
            Logger::Info("扫描到{}个WAV文件，目录: {}", wav_files.size(), directory);
            
        } catch (const std::exception& e) {
            Logger::Error("扫描音频目录异常: {} - {}", directory, e.what());
        }
        
        return wav_files;
    }
};

/**
 * 性能指标结构体
 */
struct PerformanceMetrics {
    // ============ 核心性能指标 ============
    double streaming_rtf = 0.0;           // 流式实时因子
    double offline_rtf = 0.0;             // 离线实时因子  
    double two_pass_rtf = 0.0;            // 2Pass模式RTF
    double end_to_end_latency_ms = 0.0;   // 端到端延迟
    double gpu_memory_gb = 0.0;           // GPU显存使用
    
    // ============ 2Pass模式专用指标 ============
    double online_latency_ms = 0.0;       // 在线识别延迟
    double offline_refinement_ms = 0.0;   // 离线精化时间
    double vad_processing_ms = 0.0;       // VAD处理时间
    double punctuation_ms = 0.0;          // 标点符号处理时间
    
    // ============ 吞吐量指标 ============
    int concurrent_sessions = 0;          // 并发会话数
    double total_audio_processed_hours = 0.0; // 已处理音频总时长
    
    // ============ 统计指标 ============
    uint64_t total_requests = 0;          // 总请求数
    uint64_t success_requests = 0;        // 成功请求数
    int test_files_count = 0;             // 测试文件数量
    
    double GetSuccessRate() const {
        return total_requests > 0 ? (double(success_requests) / total_requests) * 100.0 : 100.0;
    }
    
    double GetSpeedupFactor() const {
        return streaming_rtf > 0 ? (1.0 / streaming_rtf) : 0.0;
    }
    
    std::string ToString() const {
        std::ostringstream oss;
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        oss << "                            🏆 FunASR GPU 性能测试报告                            \n";
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        oss << "📊 核心性能指标:\n";
        oss << "   流式识别RTF:      " << std::fixed << std::setprecision(4) << streaming_rtf 
            << " (实时倍数: " << std::setprecision(1) << GetSpeedupFactor() << "x)\n";
        oss << "   离线识别RTF:      " << std::fixed << std::setprecision(4) << offline_rtf 
            << " (实时倍数: " << std::setprecision(1) << (offline_rtf > 0 ? 1.0/offline_rtf : 0) << "x)\n";
        oss << "   2Pass模式RTF:     " << std::fixed << std::setprecision(4) << two_pass_rtf 
            << " (实时倍数: " << std::setprecision(1) << (two_pass_rtf > 0 ? 1.0/two_pass_rtf : 0) << "x)\n";
        oss << "   端到端延迟:       " << std::fixed << std::setprecision(1) << end_to_end_latency_ms << "ms\n";
        oss << "\n";
        oss << "⚡ 2Pass模式详细指标:\n";
        oss << "   在线识别延迟:     " << std::fixed << std::setprecision(1) << online_latency_ms << "ms\n";
        oss << "   离线精化时间:     " << std::fixed << std::setprecision(1) << offline_refinement_ms << "ms\n";
        oss << "   VAD处理时间:      " << std::fixed << std::setprecision(1) << vad_processing_ms << "ms\n";
        oss << "   标点符号时间:     " << std::fixed << std::setprecision(1) << punctuation_ms << "ms\n";
        oss << "\n";
        oss << "💾 资源使用:\n";
        oss << "   GPU显存使用:      " << std::fixed << std::setprecision(1) << gpu_memory_gb << "GB\n";
        oss << "   并发会话数:       " << concurrent_sessions << "\n";
        oss << "\n";
        oss << "📈 统计信息:\n";
        oss << "   测试文件数:       " << test_files_count << " 个WAV文件\n";
        oss << "   处理音频总时长:   " << std::fixed << std::setprecision(1) << total_audio_processed_hours << " 小时\n";
        oss << "   成功率:           " << std::fixed << std::setprecision(1) << GetSuccessRate() << "%\n";
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        return oss.str();
    }
};
