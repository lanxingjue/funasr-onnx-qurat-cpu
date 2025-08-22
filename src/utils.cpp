#include "utils.h"

/**
 * Logger 日志等级实现（建议放头文件后专门实现）
 */
Logger::Level Logger::current_level_ = Logger::INFO;

/**
 * 读取 WAV 文件，16位 PCM 格式，附详细日志输出
 */
AudioFileReader::AudioData AudioFileReader::ReadWavFile(const std::string& file_path) {
    AudioData audio_data;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        Logger::Error("无法打开音频文件: {}", file_path);
        return audio_data;
    }
    try {
        // 读取 WAV 头部
        char header[44];
        file.read(header, 44);
        if (file.gcount() != 44) {
            Logger::Error("WAV文件头部读取失败: {}", file_path);
            return audio_data;
        }
        // 验证并解析
        if (std::string(header, 4) != "RIFF" || std::string(header + 8, 4) != "WAVE") {
            Logger::Error("不是有效的WAV文件: {}", file_path);
            return audio_data;
        }
        audio_data.channels = *reinterpret_cast<short*>(header + 22);
        audio_data.sample_rate = *reinterpret_cast<int*>(header + 24);
        int bits_per_sample = *reinterpret_cast<short*>(header + 34);
        int data_size = *reinterpret_cast<int*>(header + 40);
        if (bits_per_sample != 16) {
            Logger::Error("暂不支持{}位音频，请转换为16位PCM格式: {}", bits_per_sample, file_path);
            return audio_data;
        }
        // 读取音频数据
        std::vector<short> raw_data(data_size / 2);
        file.read(reinterpret_cast<char*>(raw_data.data()), data_size);
        if (file.gcount() != data_size) {
            Logger::Error("音频数据读取不完整: {}", file_path);
            return audio_data;
        }
        // 格式归一化
        audio_data.samples.resize(raw_data.size());
        for (size_t i = 0; i < raw_data.size(); ++i) {
            audio_data.samples[i] = static_cast<float>(raw_data[i]) / 32768.0f;
        }
        // 立体声转换
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
        // 音频时长计算
        audio_data.duration_seconds = static_cast<double>(audio_data.samples.size()) / audio_data.sample_rate;
        Logger::Info("音频读取成功: 时长={:.2f}秒, 样本数={}", audio_data.duration_seconds, audio_data.samples.size());
    } catch (const std::exception& e) {
        Logger::Error("读取WAV文件异常: {} - {}", file_path, e.what());
    }
    return audio_data;
}

/**
 * 扫描目录下所有 WAV 文件
 */
std::vector<std::string> AudioFileReader::ScanWavFiles(const std::string& directory) {
    std::vector<std::string> wav_files;
    try {
        if (!std::filesystem::exists(directory)) {
            Logger::Error("音频目录不存在: {}", directory);
            return wav_files;
        }
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".wav")
                    wav_files.push_back(entry.path().string());
            }
        }
        std::sort(wav_files.begin(), wav_files.end());
        Logger::Info("扫描到{}个WAV文件，目录: {}", wav_files.size(), directory);
    } catch (const std::exception& e) {
        Logger::Error("扫描音频目录异常: {} - {}", directory, e.what());
    }
    return wav_files;
}
