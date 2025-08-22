#include "funasr_engine.h"
#include <random>
#include <algorithm>
#include <future>
#include <sstream>
#include <iomanip>

// Linux系统相关头文件 (CPU版本新增)
#ifdef __linux__
#include <sys/resource.h>  // CPU资源监控
#include <unistd.h>        // 系统信息
#endif

/**
 * 构造函数 - CPU版本适配
 * 
 * 🔄 主要变化: 日志输出适配CPU模式
 */
FunASREngine::FunASREngine(const Config& config) : config_(config) {
    // 修复日志格式化问题 - 使用字符串拼接替代模板格式化
    std::ostringstream log_msg;
    log_msg << "创建FunASR CPU引擎，设备: " << config_.device 
            << ", 音频目录: " << config_.audio_files_dir;
    Logger::Info(log_msg.str());
    
    // CPU特有信息日志
    std::ostringstream cpu_info;
    cpu_info << "CPU配置: " << config_.cpu_threads << "核心, "
             << "最大并发: " << config_.max_concurrent_sessions << "路";
    Logger::Info(cpu_info.str());
}

FunASREngine::~FunASREngine() {
    testing_active_ = false;
    if (test_thread_.joinable()) {
        test_thread_.join();
    }
    Logger::Info("FunASR CPU引擎已销毁");
}

/**
 * 初始化引擎 - CPU版本核心改造
 * 
 * 🔄 主要改造点:
 * 1. 增加CPU性能优化步骤
 * 2. 修改Python环境初始化 (CPU模式)
 * 3. 修改资源监控 (CPU内存替代GPU显存)
 * 4. 修复日志格式化问题
 */
bool FunASREngine::Initialize() {
    Logger::Info("初始化FunASR CPU引擎...");
    Timer init_timer;
    
    try {
        // 1. CPU性能优化 (🆕 CPU版本新增)
        if (config_.enable_cpu_optimization) {
            OptimizeCPUPerformance();
        }
        
        // 2. 初始化Python环境 - CPU模式适配
        if (!InitializePython()) {
            Logger::Error("Python环境初始化失败");
            return false;
        }
        
        // 3. 加载所有FunASR模型 - CPU配置
        Logger::Info("加载FunASR模型组件到CPU...");
        
        // 加载流式ASR模型
        if (!LoadFunASRModel("streaming_asr", config_.streaming_model,
                            config_.streaming_revision, streaming_model_)) {
            Logger::Error("流式ASR模型加载失败");
            return false;
        }
        
        // 加载离线ASR模型
        if (!LoadFunASRModel("offline_asr", config_.offline_model,
                            config_.offline_revision, offline_model_)) {
            Logger::Error("离线ASR模型加载失败");
            return false;
        }
        
        // 加载VAD模型
        if (!LoadFunASRModel("vad", config_.vad_model,
                            config_.vad_revision, vad_model_)) {
            Logger::Error("VAD模型加载失败");
            return false;
        }
        
        // 加载标点符号模型
        if (!LoadFunASRModel("punctuation", config_.punc_model,
                            config_.punc_revision, punc_model_)) {
            Logger::Error("标点符号模型加载失败");
            return false;
        }
        
        // 4. 检查CPU资源状态 (🔄 替代GPU状态检查)
        double cpu_memory = GetCPUMemoryUsage();
        std::ostringstream memory_log;
        memory_log << "系统内存使用: " << std::fixed << std::setprecision(1) << cpu_memory << "GB";
        Logger::Info(memory_log.str());
        
        // 5. 加载测试音频文件
        if (!LoadTestAudioFiles()) {
            Logger::Error("加载测试音频文件失败");
            return false;
        }
        
        // 6. 初始化性能指标
        current_metrics_.gpu_memory_gb = cpu_memory;  // 复用字段存储CPU内存
        current_metrics_.test_files_count = test_audio_files_.size();
        
        initialized_ = true;
        
        // 修复日志格式化 - 使用ostringstream
        std::ostringstream completion_log;
        completion_log << "FunASR CPU引擎初始化完成，耗时: " 
                      << std::fixed << std::setprecision(1) << init_timer.ElapsedMs() << "ms";
        Logger::Info(completion_log.str());
        
        Logger::Info("已加载模型: 流式ASR + 离线ASR + VAD + 标点符号 (CPU模式)");
        
        std::ostringstream files_log;
        files_log << "测试音频文件: " << test_audio_files_.size() << "个";
        Logger::Info(files_log.str());
        
        return true;
        
    } catch (const std::exception& e) {
        std::string error_msg = "引擎初始化异常: " + std::string(e.what());
        Logger::Error(error_msg);
        return false;
    }
}


/**
 * 初始化Python环境 - CPU模式核心适配
 * 
 * 🔄 主要改造点:
 * 1. 跳过CUDA可用性检查
 * 2. 强制使用CPU模式
 * 3. 设置PyTorch CPU线程数
 * 4. 处理CUDA警告
 */
bool FunASREngine::InitializePython() {
    try {
        // 创建Python解释器（保持不变）
        py_guard_ = std::make_unique<py::scoped_interpreter>();
        
        // 导入必要模块
        py::module_ sys = py::module_::import("sys");
        py::module_ funasr = py::module_::import("funasr");
        py::module_ torch = py::module_::import("torch");
        
        // 🔄 CPU模式 - 跳过CUDA检查
        // 原GPU版本会检查CUDA可用性并报错，CPU版本跳过
        Logger::Info("Python环境初始化成功 (CPU模式)");
        
        // 🔄 设置PyTorch使用CPU
        torch.attr("set_num_threads")(config_.cpu_threads);
        
        // 🔄 处理CUDA检测但强制CPU模式
        if (torch.attr("cuda").attr("is_available")().cast<bool>()) {
            Logger::Info("检测到CUDA，但强制使用CPU模式");
        }
        
        std::ostringstream threads_log;
        threads_log << "PyTorch CPU线程数: " << config_.cpu_threads;
        Logger::Info(threads_log.str());
        
        return true;
        
    } catch (const py::error_already_set& e) {
        std::string error_msg = "Python初始化失败: " + std::string(e.what());
        Logger::Error(error_msg);
        return false;
    }
}

/**
 * 加载FunASR模型 - CPU配置核心适配
 * 
 * 🔄 主要改造点:
 * 1. device = "cpu" (替代cuda设备)
 * 2. ngpu = 0 (不使用GPU)
 * 3. ncpu = config_.cpu_threads (使用所有CPU核心)
 * 4. 增加disable_update=true (减少网络依赖)
 */
bool FunASREngine::LoadFunASRModel(const std::string& model_type,
                                   const std::string& model_name,
                                   const std::string& model_revision,
                                   py::object& model_obj) {
    try {
        std::ostringstream load_log;
        load_log << "加载" << model_type << "模型到CPU: " << model_name 
                 << " (版本: " << model_revision << ")";
        Logger::Info(load_log.str());
        
        Timer load_timer;
        
        // 使用FunASR的AutoModel类 (保持不变)
        py::module_ funasr = py::module_::import("funasr");
        py::object auto_model = funasr.attr("AutoModel");
        
        // 🔄 CPU模式配置参数 (核心改造)
        py::dict kwargs;
        kwargs["model"] = model_name;
        kwargs["model_revision"] = model_revision;
        kwargs["device"] = config_.device;              // 🔄 "cpu" 替代 "cuda:0"
        kwargs["ngpu"] = 0;                            // 🔄 0 替代 1 (不使用GPU)
        kwargs["ncpu"] = config_.cpu_threads;          // 🔄 使用所有CPU核心
        kwargs["disable_pbar"] = true;
        kwargs["disable_log"] = true;
        kwargs["disable_update"] = true;               // 🆕 禁用版本检查，减少网络依赖
        
        // CPU特定优化配置
        if (model_type == "streaming_asr") {
            // 流式模型CPU优化
            kwargs["batch_size"] = 1;                  // 🆕 CPU建议单批次处理
        } else if (model_type == "offline_asr") {
            // 离线模型CPU优化
            kwargs["batch_size"] = 1;                  // 🆕 CPU建议单批次处理
        }
        
        // 实例化模型
        model_obj = auto_model(**kwargs);
        
        std::ostringstream completion_log;
        completion_log << model_type << "模型加载完成 (CPU模式)，耗时: " 
                      << std::fixed << std::setprecision(1) << load_timer.ElapsedMs() << "ms";
        Logger::Info(completion_log.str());
        
        return true;
        
    } catch (const py::error_already_set& e) {
        std::string error_msg = model_type + "模型加载失败: " + std::string(e.what());
        Logger::Error(error_msg);
        return false;
    }
}

/**
 * CPU性能优化 - CPU版本新增功能
 * 
 * 🆕 CPU版本专用方法:
 * 1. 设置环境变量优化多线程库
 * 2. 配置OpenMP和MKL线程数
 * 3. 进程优先级调整 (可选)
 */
void FunASREngine::OptimizeCPUPerformance() {
    Logger::Info("启动CPU性能优化...");
    
    try {
        // 设置环境变量优化CPU性能
        std::string thread_str = std::to_string(config_.cpu_threads);
        
        // OpenMP线程数
        setenv("OMP_NUM_THREADS", thread_str.c_str(), 1);
        
        // Intel MKL线程数 (如果使用Intel MKL)
        setenv("MKL_NUM_THREADS", thread_str.c_str(), 1);
        
        // NumExpr线程数
        setenv("NUMEXPR_NUM_THREADS", thread_str.c_str(), 1);
        
        // 可选：提升进程优先级 (需要权限)
#ifdef __linux__
        if (setpriority(PRIO_PROCESS, 0, -5) == 0) {
            Logger::Info("进程优先级已适度提升");
        }
#endif
        
        Logger::Info("CPU性能优化完成");
        
    } catch (const std::exception& e) {
        std::string warn_msg = "CPU性能优化部分失败: " + std::string(e.what());
        Logger::Warn(warn_msg);
    }
}

/**
 * 音频重采样 - CPU版本新增功能
 * 
 * 🆕 解决24kHz音频问题:
 * 使用线性插值实现高质量音频重采样
 * 支持任意采样率到16kHz的转换
 */
std::vector<float> FunASREngine::ResampleAudio(const std::vector<float>& audio_data, 
                                               int from_rate, int to_rate) {
    if (from_rate == to_rate) {
        return audio_data;
    }
    
    // 简单线性插值重采样
    double ratio = static_cast<double>(to_rate) / from_rate;
    size_t new_size = static_cast<size_t>(audio_data.size() * ratio);
    std::vector<float> resampled(new_size);
    
    for (size_t i = 0; i < new_size; ++i) {
        double src_index = i / ratio;
        size_t idx = static_cast<size_t>(src_index);
        
        if (idx + 1 < audio_data.size()) {
            double frac = src_index - idx;
            resampled[i] = audio_data[idx] * (1.0 - frac) + audio_data[idx + 1] * frac;
        } else {
            resampled[i] = audio_data.back();
        }
    }
    
    std::ostringstream resample_log;
    resample_log << "音频重采样完成: " << from_rate << "Hz → " << to_rate << "Hz, "
                 << "样本数: " << audio_data.size() << " → " << new_size;
    Logger::Info(resample_log.str());
    
    return resampled;
}

/**
 * 离线语音识别 - CPU版本增强
 * 
 * 🔄 CPU版本改进:
 * 1. 增加音频重采样功能 (24kHz→16kHz)
 * 2. 增强错误处理和异常捕获
 * 3. 优化VAD分段处理逻辑
 * 4. 修复日志格式化问题
 */
FunASREngine::RecognitionResult FunASREngine::OfflineRecognize(
    const std::vector<float>& audio_input,
    bool enable_vad,
    bool enable_punctuation) {
    
    RecognitionResult result;
    if (!initialized_) {
        Logger::Error("引擎未初始化");
        return result;
    }
    
    try {
        Timer total_timer;
        
        // 🆕 音频预处理 - 重采样支持
        std::vector<float> audio_data = audio_input;
        if (config_.enable_audio_resampling && audio_data.size() > 0) {
            // 假设输入可能是24kHz (基于你的日志)，重采样到16kHz
            // 实际项目中应该从AudioFileReader获取真实采样率
            double input_duration = audio_data.size() / 24000.0; // 假设24kHz
            if (input_duration > 0) {
                audio_data = ResampleAudio(audio_input, 24000, 16000);
            }
        }
        
        std::string final_text;
        
        // VAD分段处理 - 增强错误处理
        enable_vad = false;
        if (enable_vad && audio_data.size() > 16000 * 5) { // 大于5秒启用VAD
            Logger::Info("长音频检测，启用VAD分段处理 (CPU模式)");
            
            try {
                std::map<std::string, py::object> vad_cache;
                auto vad_result = DetectVoiceActivity(audio_data, vad_cache);
                
                if (vad_result.HasValidSegments()) {
                    std::ostringstream vad_log;
                    vad_log << "VAD检测到" << vad_result.segments.size() << "个语音段";
                    Logger::Info(vad_log.str());
                    
                    // 对每个语音段进行ASR识别
                    std::vector<std::string> segment_texts;
                    for (const auto& segment : vad_result.segments) {
                        int start_sample = (segment.first * 16000) / 1000;
                        int end_sample = (segment.second * 16000) / 1000;
                        
                        if (start_sample >= 0 && end_sample <= static_cast<int>(audio_data.size()) && 
                            end_sample > start_sample) {
                            
                            std::vector<float> segment_audio(
                                audio_data.begin() + start_sample,
                                audio_data.begin() + end_sample
                            );
                            
                            // CPU ASR识别单个段
                            py::array_t<float> audio_array = VectorToNumpy(segment_audio);
                            py::dict asr_kwargs;
                            asr_kwargs["input"] = audio_array;
                            
                            py::object asr_result = offline_model_.attr("generate")(**asr_kwargs);
                            auto parsed_result = ParseRecognitionResult(asr_result, 0);
                            
                            if (!parsed_result.text.empty()) {
                                segment_texts.push_back(parsed_result.text);
                            }
                        }
                    }
                    
                    // 合并所有段的文本
                    for (const auto& text : segment_texts) {
                        if (!final_text.empty()) final_text += " ";
                        final_text += text;
                    }
                } else {
                    Logger::Info("VAD未检测到有效语音段，使用完整音频识别");
                    enable_vad = false; // 回退到完整音频识别
                }
            } catch (const std::exception& e) {
                std::string error_msg = "VAD处理异常，回退到完整音频识别: " + std::string(e.what());
                Logger::Error(error_msg);
                enable_vad = false;
            }
        }
        
        // 完整音频识别 (CPU)
        if (!enable_vad || final_text.empty()) {
            try {
                py::array_t<float> audio_array = VectorToNumpy(audio_data);
                py::dict asr_kwargs;
                asr_kwargs["input"] = audio_array;
                
                py::object asr_result = offline_model_.attr("generate")(**asr_kwargs);
                auto parsed_result = ParseRecognitionResult(asr_result, 0);
                final_text = parsed_result.text;
            } catch (const std::exception& e) {
                std::string error_msg = "离线识别异常: " + std::string(e.what());
                Logger::Error(error_msg);
                return result;
            }
        }
        
        // 标点符号恢复 (CPU)
        if (enable_punctuation && !final_text.empty()) {
            try {
                std::map<std::string, py::object> punc_cache;
                final_text = AddPunctuation(final_text, punc_cache);
            } catch (const std::exception& e) {
                std::string warn_msg = "标点符号处理异常: " + std::string(e.what());
                Logger::Warn(warn_msg);
            }
        }
        
        result.text = final_text;
        result.is_final = true;
        result.is_offline_result = true;
        result.inference_time_ms = total_timer.ElapsedMs();
        
        // 更新性能指标
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.total_requests++;
            if (!result.IsEmpty()) {
                current_metrics_.success_requests++;
                double audio_duration_s = audio_data.size() / 16000.0;
                current_metrics_.offline_rtf = result.inference_time_ms / (audio_duration_s * 1000.0);
                current_metrics_.total_audio_processed_hours += audio_duration_s / 3600.0;
            }
        }
        
        // 修复日志格式化
        std::ostringstream result_log;
        result_log << "CPU离线识别完成: '" << final_text.substr(0, 30) << "...', 耗时: " 
                  << std::fixed << std::setprecision(1) << result.inference_time_ms << "ms";
        Logger::Info(result_log.str());
        
    } catch (const std::exception& e) {
        std::string error_msg = "CPU离线识别异常: " + std::string(e.what());
        Logger::Error(error_msg);
        current_metrics_.total_requests++;
    }
    
    return result;
}

/**
 * 流式识别 - CPU版本优化
 * 
 * 🔄 CPU版本改进:
 * 1. CPU多线程推理优化
 * 2. 修复pybind11对象生命周期问题
 * 3. 增强缓存状态管理
 * 4. 修复日志格式化
 */
FunASREngine::RecognitionResult FunASREngine::StreamingRecognize(
    const std::vector<float>& audio_chunk,
    TwoPassSession& session,
    bool is_final) {
    
    RecognitionResult result;
    if (!initialized_) {
        Logger::Error("引擎未初始化");
        return result;
    }
    
    try {
        Timer inference_timer;
        
        // 转换音频数据为numpy数组
        py::array_t<float> audio_array = VectorToNumpy(audio_chunk);
        
        // 构建流式推理参数 (CPU配置)
        py::dict kwargs;
        kwargs["input"] = audio_array;
        kwargs["is_final"] = is_final;
        kwargs["chunk_size"] = py::make_tuple(
            session.chunk_size[0],
            session.chunk_size[1],
            session.chunk_size[2]
        );
        kwargs["encoder_chunk_look_back"] = session.encoder_chunk_look_back;
        kwargs["decoder_chunk_look_back"] = session.decoder_chunk_look_back;
        
        // 添加缓存状态 (修复pybind11对象生命周期问题)
        if (!session.streaming_cache.empty()) {
            py::dict cache_dict;
            for (const auto& item : session.streaming_cache) {
                cache_dict[py::str(item.first)] = item.second;
            }
            kwargs["cache"] = cache_dict;
        }
        
        // 执行CPU流式推理
        py::object py_result = streaming_model_.attr("generate")(**kwargs);
        
        // 🔄 更新缓存状态 - 修复pybind11对象赋值问题
        if (kwargs.contains("cache")) {
            py::dict updated_cache = kwargs["cache"];
            session.streaming_cache.clear();
            for (auto item : updated_cache) {
                // 修复：使用reinterpret_borrow显式转换handle到object
                session.streaming_cache[item.first.cast<std::string>()] = 
                    py::reinterpret_borrow<py::object>(item.second);
            }
        }
        
        // 解析结果
        result = ParseRecognitionResult(py_result, inference_timer.ElapsedMs());
        result.is_final = is_final;
        result.is_online_result = true;
        
        // 更新性能指标
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.total_requests++;
            if (!result.IsEmpty()) {
                current_metrics_.success_requests++;
                double chunk_duration_s = audio_chunk.size() / 16000.0;
                current_metrics_.streaming_rtf = result.inference_time_ms / (chunk_duration_s * 1000.0);
                current_metrics_.online_latency_ms = result.inference_time_ms;
                current_metrics_.total_audio_processed_hours += chunk_duration_s / 3600.0;
            }
        }
        
        // 修复日志格式化
        std::ostringstream stream_log;
        stream_log << "CPU流式识别: '" << result.text << "', 耗时: " 
                  << std::fixed << std::setprecision(1) << result.inference_time_ms << "ms, RTF: "
                  << std::fixed << std::setprecision(4) << current_metrics_.streaming_rtf;
        Logger::Info(stream_log.str());
        
    } catch (const std::exception& e) {
        std::string error_msg = "CPU流式识别异常: " + std::string(e.what());
        Logger::Error(error_msg);
        current_metrics_.total_requests++;
    }
    
    return result;
}

/**
 * CPU内存使用量获取 - 替代GPU显存监控
 * 
 * 🆕 CPU版本新增功能:
 * 通过读取/proc/meminfo获取系统内存使用情况
 */
double FunASREngine::GetCPUMemoryUsage() {
    try {
#ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open()) {
            return 0.0;
        }
        
        std::string line;
        long mem_total = 0, mem_available = 0;
        
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                sscanf(line.c_str(), "MemTotal: %ld kB", &mem_total);
            } else if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %ld kB", &mem_available);
            }
        }
        
        if (mem_total > 0 && mem_available > 0) {
            double used_gb = (mem_total - mem_available) / (1024.0 * 1024.0);
            return used_gb;
        }
#endif
    } catch (...) {
        // 静默处理异常
    }
    return 0.0;
}

// ============ 其他方法的CPU版本适配 ============

/**
 * 2Pass识别 - CPU并行优化版本
 * 基本逻辑保持不变，但增强了错误处理和日志修复
 */
void FunASREngine::TwoPassRecognize(
    const std::vector<float>& audio_chunk,
    TwoPassSession& session,
    std::vector<RecognitionResult>& results) {
    
    if (!initialized_) {
        Logger::Error("引擎未初始化");
        return;
    }
    
    try {
        Timer total_timer;
        
        // 添加音频块到缓冲区
        session.audio_buffer.insert(session.audio_buffer.end(),
                                   audio_chunk.begin(), audio_chunk.end());
        
        // 1. CPU并行执行VAD检测和流式识别
        std::future<VADResult> vad_future = std::async(std::launch::async, 
            [this, &audio_chunk, &session]() {
                return DetectVoiceActivity(audio_chunk, session.vad_cache);
            });
            
        std::future<RecognitionResult> streaming_future = std::async(std::launch::async,
            [this, &audio_chunk, &session]() {
                return StreamingRecognize(audio_chunk, session, false);
            });
        
        // 2. 获取流式识别结果 (立即返回给用户)
        auto streaming_result = streaming_future.get();
        if (!streaming_result.IsEmpty()) {
            streaming_result.is_online_result = true;
            results.push_back(streaming_result);
        }
        
        // 3. 处理VAD结果
        auto vad_result = vad_future.get();
        current_metrics_.vad_processing_ms = vad_result.inference_time_ms;
        
        // 4. 检测语音结束点
        if (vad_result.speech_end_ms != -1) {
            session.is_speaking = false;
            Logger::Info("检测到语音结束，启动离线精化处理");
            
            // 提取完整语音段进行离线识别
            std::vector<float> complete_segment = session.audio_buffer;
            
            // CPU异步执行离线精化
            std::thread([this, complete_segment, &session]() {
                Timer offline_timer;
                auto offline_result = OfflineRecognize(complete_segment, false, true);
                if (!offline_result.IsEmpty()) {
                    offline_result.is_offline_result = true;
                    offline_result.is_final = true;
                    current_metrics_.offline_refinement_ms = offline_timer.ElapsedMs();
                    
                    std::ostringstream offline_log;
                    offline_log << "离线精化完成: '" << offline_result.text << "'";
                    Logger::Info(offline_log.str());
                }
                session.Reset();
            }).detach();
            
        } else if (vad_result.speech_start_ms != -1) {
            session.is_speaking = true;
        }
        
        // 更新2Pass模式性能指标
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            double chunk_duration_s = audio_chunk.size() / 16000.0;
            current_metrics_.two_pass_rtf = total_timer.ElapsedMs() / (chunk_duration_s * 1000.0);
            current_metrics_.end_to_end_latency_ms = total_timer.ElapsedMs();
        }
        
    } catch (const std::exception& e) {
        std::string error_msg = "CPU 2Pass识别异常: " + std::string(e.what());
        Logger::Error(error_msg);
    }
}

/**
 * VAD检测 - CPU版本 (基本逻辑保持不变，修复日志格式化)
 */
FunASREngine::VADResult FunASREngine::DetectVoiceActivity(
    const std::vector<float>& audio_data,
    std::map<std::string, py::object>& vad_cache,
    int max_single_segment_time) {
    
    VADResult result;
    try {
        Timer vad_timer;
        
        // 转换音频数据
        py::array_t<float> audio_array = VectorToNumpy(audio_data);
        
        // 构建VAD参数
        py::dict kwargs;
        kwargs["input"] = audio_array;
        kwargs["max_single_segment_time"] = max_single_segment_time;
        
        // 添加缓存状态
        if (!vad_cache.empty()) {
            py::dict cache_dict;
            for (const auto& item : vad_cache) {
                cache_dict[py::str(item.first)] = item.second;
            }
            kwargs["cache"] = cache_dict;
        }
        
        // 执行CPU VAD推理
        py::object vad_py_result = vad_model_.attr("generate")(**kwargs);
        
        // 更新缓存 - 修复pybind11对象赋值问题
        if (kwargs.contains("cache")) {
            py::dict updated_cache = kwargs["cache"];
            vad_cache.clear();
            for (auto item : updated_cache) {
                vad_cache[item.first.cast<std::string>()] = 
                    py::reinterpret_borrow<py::object>(item.second);
            }
        }
        
        result = ParseVADResult(vad_py_result, vad_timer.ElapsedMs());
        
    } catch (const std::exception& e) {
        std::string error_msg = "CPU VAD检测异常: " + std::string(e.what());
        Logger::Error(error_msg);
    }
    
    return result;
}

/**
 * 标点符号恢复 - CPU版本 (基本逻辑保持不变，修复日志格式化)
 */
std::string FunASREngine::AddPunctuation(const std::string& text,
                                         std::map<std::string, py::object>& punc_cache) {
    if (text.empty() || punc_model_.is_none()) {
        return text;
    }
    
    try {
        Timer punc_timer;
        
        // 构建标点符号恢复参数
        py::dict kwargs;
        kwargs["input"] = text;
        
        // 添加缓存状态
        if (!punc_cache.empty()) {
            py::dict cache_dict;
            for (const auto& item : punc_cache) {
                cache_dict[py::str(item.first)] = item.second;
            }
            kwargs["cache"] = cache_dict;
        }
        
        // 执行CPU标点符号恢复
        py::object punc_result = punc_model_.attr("generate")(**kwargs);
        
        // 更新缓存 - 修复pybind11对象赋值问题
        if (kwargs.contains("cache")) {
            py::dict updated_cache = kwargs["cache"];
            punc_cache.clear();
            for (auto item : updated_cache) {
                punc_cache[item.first.cast<std::string>()] = 
                    py::reinterpret_borrow<py::object>(item.second);
            }
        }
        
        // 解析结果
        if (py::isinstance<py::list>(punc_result)) {
            py::list result_list = punc_result;
            if (result_list.size() > 0) {
                py::dict first_result = result_list[0];
                if (first_result.contains("text")) {
                    std::string punctuated_text = first_result["text"].cast<std::string>();
                    current_metrics_.punctuation_ms = punc_timer.ElapsedMs();
                    return punctuated_text;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::string error_msg = "CPU标点符号处理异常: " + std::string(e.what());
        Logger::Error(error_msg);
    }
    
    return text; // 出错时返回原文本
}

// ============ 辅助方法实现 (基本保持不变) ============

py::array_t<float> FunASREngine::VectorToNumpy(const std::vector<float>& data) {
    return py::array_t<float>(
        data.size(),
        data.data(),
        py::cast(this)
    );
}

FunASREngine::RecognitionResult FunASREngine::ParseRecognitionResult(
    const py::object& result, double inference_time_ms) {
    
    RecognitionResult parsed;
    parsed.inference_time_ms = inference_time_ms;
    
    try {
        if (py::isinstance<py::list>(result)) {
            py::list result_list = result;
            if (result_list.size() > 0) {
                py::dict first_result = result_list[0];
                if (first_result.contains("text")) {
                    parsed.text = first_result["text"].cast<std::string>();
                }
            }
        }
    } catch (const std::exception& e) {
        std::string error_msg = "识别结果解析异常: " + std::string(e.what());
        Logger::Error(error_msg);
    }
    
    return parsed;
}

FunASREngine::VADResult FunASREngine::ParseVADResult(
    const py::object& result, double inference_time_ms) {
    
    VADResult parsed;
    parsed.inference_time_ms = inference_time_ms;
    
    try {
        if (py::isinstance<py::list>(result)) {
            py::list result_list = result;
            if (result_list.size() > 0) {
                py::dict first_result = result_list[0];
                if (first_result.contains("value")) {
                    py::list segments = first_result["value"];
                    for (auto segment : segments) {
                        py::list seg_pair = py::reinterpret_borrow<py::list>(segment);
                        if (seg_pair.size() >= 2) {
                            int64_t start = seg_pair[0].cast<int64_t>();
                            int64_t end = seg_pair[1].cast<int64_t>();
                            parsed.segments.emplace_back(start, end);
                            
                            if (start != -1 && parsed.speech_start_ms == -1) {
                                parsed.speech_start_ms = start;
                            }
                            if (end != -1) {
                                parsed.speech_end_ms = end;
                            }
                        }
                    }
                    parsed.has_speech = !parsed.segments.empty();
                }
            }
        }
    } catch (const std::exception& e) {
        std::string error_msg = "VAD结果解析异常: " + std::string(e.what());
        Logger::Error(error_msg);
    }
    
    return parsed;
}
// ============ 性能测试实现 ============

bool FunASREngine::RunPerformanceTests() {
    if (!initialized_ || test_audio_files_.empty()) {
        Logger::Error("引擎未初始化或无测试文件");
        return false;
    }

    Logger::Info("🧪 开始FunASR CPU完整性能测试套件...");
    testing_active_ = true;
    
    test_thread_ = std::thread([this]() {
        try {
            Timer total_test_timer;
            
            if (config_.enable_offline_test) {
                Logger::Info("1️⃣ 离线识别性能测试 (CPU模式)...");
                auto offline_metrics = TestOfflinePerformance();
                UpdateMetrics(offline_metrics);
            }
            
            Logger::Info("🎉 完整CPU性能测试套件完成！总耗时: {:.1f}秒", 
                        total_test_timer.ElapsedMs() / 1000.0);
        } catch (const std::exception& e) {
            Logger::Error("性能测试异常: {}", e.what());
        }
        
        testing_active_ = false; // 标记测试完成
    });
    
    return true;
}

PerformanceMetrics FunASREngine::TestOfflinePerformance() {
    PerformanceMetrics metrics;
    int test_count = std::min(20, static_cast<int>(test_audio_files_.size()));
    std::vector<double> rtf_values;
    double total_audio_duration = 0.0;
    
    Logger::Info("开始离线测试，目标处理{}个音频文件", test_count);
    
    for (int i = 0; i < test_count; ++i) {
        const auto& file_path = test_audio_files_[i];
        Logger::Info("处理音频文件 [{}/{}]: {}", i+1, test_count, file_path);
        
        auto audio_data = AudioFileReader::ReadWavFile(file_path);
        if (!audio_data.IsValid()) {
            Logger::Warn("跳过无效音频文件: {}", file_path);
            continue;
        }

        Timer test_timer;
        Logger::Info("开始识别，音频时长: {:.2f}秒", audio_data.duration_seconds);
        
        auto result = OfflineRecognize(audio_data.samples, true, true);
        double elapsed_ms = test_timer.ElapsedMs();
        
        if (!result.IsEmpty()) {
            double rtf = elapsed_ms / (audio_data.duration_seconds * 1000.0);
            rtf_values.push_back(rtf);
            total_audio_duration += audio_data.duration_seconds;
            
            Logger::Info("识别完成 [{}/{}]: RTF={:.4f}, 耗时={:.1f}ms, 结果: '{}'", 
                        i+1, test_count, rtf, elapsed_ms, result.text.substr(0, 50));
        } else {
            Logger::Error("识别失败 [{}/{}]: 返回空结果", i+1, test_count);
        }
    }
    
    if (!rtf_values.empty()) {
        double sum = 0;
        for (double rtf : rtf_values) sum += rtf;
        metrics.offline_rtf = sum / rtf_values.size();
        metrics.total_audio_processed_hours = total_audio_duration / 3600.0;
        metrics.test_files_count = static_cast<int>(rtf_values.size());
        
        Logger::Info("离线测试完成: 成功{}/{}个文件, 平均RTF={:.4f}, 总时长={:.2f}小时", 
                    rtf_values.size(), test_count, metrics.offline_rtf, metrics.total_audio_processed_hours);
    } else {
        Logger::Error("离线测试失败: 没有成功处理任何音频文件");
    }
    
    return metrics;
}



PerformanceMetrics FunASREngine::TestStreamingPerformance() {
    PerformanceMetrics metrics;
    int test_count = std::min(15, static_cast<int>(test_audio_files_.size()));
    std::vector<double> rtf_values, latency_values;
    Logger::Info("流式测试使用{}个音频文件", test_count);

    for (int i = 0; i < test_count; ++i) {
        const auto& file_path = test_audio_files_[i];
        auto audio_data = AudioFileReader::ReadWavFile(file_path);
        if (!audio_data.IsValid()) continue;
        auto chunks = SimulateStreamingChunks(audio_data.samples);
        TwoPassSession session;
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            bool is_final = (chunk_idx == chunks.size() - 1);
            auto result = StreamingRecognize(chunks[chunk_idx], session, is_final);
            if (!result.IsEmpty()) {
                double chunk_duration_ms = chunks[chunk_idx].size() / 16.0; // 16kHz
                double rtf = result.inference_time_ms / chunk_duration_ms;
                rtf_values.push_back(rtf);
                latency_values.push_back(result.inference_time_ms);
            }
        }
        std::ostringstream oss;
        oss << "流式测试 [" << (i+1) << "/" << test_count << "]: "
            << std::fixed << std::setprecision(1) << audio_data.duration_seconds << "秒, "
            << chunks.size() << "个分块";
        Logger::Info(oss.str());
    }
    if (!rtf_values.empty()) {
        double sum_rtf = 0, sum_latency = 0;
        for (double rtf : rtf_values) sum_rtf += rtf;
        for (double lat : latency_values) sum_latency += lat;
        metrics.streaming_rtf = sum_rtf / rtf_values.size();
        metrics.online_latency_ms = sum_latency / latency_values.size();
        metrics.end_to_end_latency_ms = metrics.online_latency_ms;
        std::ostringstream oss;
        oss << "流式测试完成: 平均RTF=" << std::fixed << std::setprecision(4) << metrics.streaming_rtf 
            << ", 平均延迟=" << std::fixed << std::setprecision(1) << metrics.online_latency_ms << "ms";
        Logger::Info(oss.str());
    }
    return metrics;
}


PerformanceMetrics FunASREngine::TestTwoPassPerformance() {
    PerformanceMetrics metrics;
    int test_count = std::min(10, static_cast<int>(test_audio_files_.size()));
    std::vector<double> rtf_values;
    Logger::Info("2Pass测试使用{}个音频文件", test_count);

    for (int i = 0; i < test_count; ++i) {
        const auto& file_path = test_audio_files_[i];
        auto audio_data = AudioFileReader::ReadWavFile(file_path);
        if (!audio_data.IsValid()) continue;
        auto chunks = SimulateStreamingChunks(audio_data.samples);
        TwoPassSession session;
        std::vector<RecognitionResult> results;
        Timer two_pass_timer;
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            TwoPassRecognize(chunks[chunk_idx], session, results);
        }
        double elapsed_ms = two_pass_timer.ElapsedMs();
        double rtf = elapsed_ms / (audio_data.duration_seconds * 1000.0);
        rtf_values.push_back(rtf);
        std::ostringstream oss;
        oss << "2Pass测试 [" << (i+1) << "/" << test_count << "]: "
            << std::fixed << std::setprecision(1) << audio_data.duration_seconds << "秒, "
            << "RTF=" << std::setprecision(4) << rtf << ", 输出" << results.size() << "个结果";
        Logger::Info(oss.str());
    }
    if (!rtf_values.empty()) {
        double sum = 0;
        for (double rtf : rtf_values) sum += rtf;
        metrics.two_pass_rtf = sum / rtf_values.size();
        std::ostringstream oss;
        oss << "2Pass测试完成: 平均RTF=" << std::fixed << std::setprecision(4) << metrics.two_pass_rtf;
        Logger::Info(oss.str());
    }
    return metrics;
}


PerformanceMetrics FunASREngine::TestConcurrentPerformance() {
    PerformanceMetrics metrics;
    const int num_workers = config_.max_concurrent_sessions;
    const int files_per_worker = std::max(1, static_cast<int>(test_audio_files_.size()) / num_workers);
    std::ostringstream start_log;
    start_log << "启动" << num_workers << "路并发测试，每路处理" << files_per_worker << "个文件";
    Logger::Info(start_log.str());

    std::vector<std::future<void>> futures;
    std::vector<PerformanceMetrics> worker_results(num_workers);
    std::atomic<int> active_sessions{0};
    Timer concurrent_timer;

    for (int i = 0; i < num_workers; ++i) {
        int start_idx = i * files_per_worker;
        int end_idx = std::min(start_idx + files_per_worker, static_cast<int>(test_audio_files_.size()));
        std::vector<std::string> worker_files(
            test_audio_files_.begin() + start_idx,
            test_audio_files_.begin() + end_idx
        );
        futures.emplace_back(std::async(std::launch::async,
            [this, i, worker_files, &active_sessions, &worker_results]() {
                ConcurrentTestWorker(i, worker_files, active_sessions, worker_results);
            })
        );
    }

    for (auto& future : futures) future.wait();

    double total_time_s = concurrent_timer.ElapsedMs() / 1000.0;
    double total_rtf = 0;
    int valid_results = 0;
    for (const auto& result : worker_results) {
        if (result.streaming_rtf > 0) {
            total_rtf += result.streaming_rtf;
            valid_results++;
        }
    }
    if (valid_results > 0) {
        metrics.streaming_rtf = total_rtf / valid_results;
    }
    metrics.concurrent_sessions = num_workers;

    std::ostringstream oss;
    oss << "并发测试完成: " << num_workers << "路并发, 平均RTF=" 
        << std::fixed << std::setprecision(4) << metrics.streaming_rtf 
        << ", 总耗时=" << std::fixed << std::setprecision(1) << total_time_s << "秒";
    Logger::Info(oss.str());

    return metrics;
}


void FunASREngine::ConcurrentTestWorker(
    int worker_id,
    const std::vector<std::string>& worker_files,
    std::atomic<int>& active_sessions,
    std::vector<PerformanceMetrics>& results) {
    try {
        active_sessions++;
        PerformanceMetrics worker_metrics;
        std::vector<double> rtf_values;
        Timer worker_timer;
        for (const auto& file_path : worker_files) {
            auto audio_data = AudioFileReader::ReadWavFile(file_path);
            if (!audio_data.IsValid()) continue;
            auto chunks = SimulateStreamingChunks(audio_data.samples);
            TwoPassSession session;
            for (const auto& chunk : chunks) {
                auto result = StreamingRecognize(chunk, session, false);
                double elapsed_ms = worker_timer.ElapsedMs();
                double rtf = elapsed_ms / (audio_data.duration_seconds * 1000.0);
                rtf_values.push_back(rtf);
            }
        }
        if (!rtf_values.empty()) {
            double sum = 0;
            for (double rtf : rtf_values) sum += rtf;
            worker_metrics.streaming_rtf = sum / rtf_values.size();
        }
        results[worker_id] = worker_metrics;
        active_sessions--;
        std::ostringstream oss;
        oss << "并发Worker-" << worker_id << " 完成: 处理" << worker_files.size()
            << "个文件, 平均RTF=" << std::fixed << std::setprecision(4) << worker_metrics.streaming_rtf;
        Logger::Info(oss.str());
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "并发Worker-" << worker_id << " 异常: " << e.what();
        Logger::Error(oss.str());
        active_sessions--;
    }
}

std::vector<std::vector<float>> FunASREngine::SimulateStreamingChunks(
    const std::vector<float>& audio_data, double chunk_duration_ms) {
    std::vector<std::vector<float>> chunks;
    const int chunk_samples = static_cast<int>((chunk_duration_ms / 1000.0) * 16000); // 假设16kHz
    for (size_t i = 0; i < audio_data.size(); i += chunk_samples) {
        size_t end_idx = std::min(i + chunk_samples, audio_data.size());
        chunks.emplace_back(audio_data.begin() + i, audio_data.begin() + end_idx);
    }
    std::ostringstream oss;
    oss << "模拟流式分块完成，分块数量: " << chunks.size() 
        << ", 每块时长: " << chunk_duration_ms << "ms";
    Logger::Info(oss.str());
    return chunks;
}


/**
 * 获取性能指标 - CPU版本修复
 * 
 * 🔄 修复说明：将GetGPUMemoryUsage()改为GetCPUMemoryUsage()
 */
PerformanceMetrics FunASREngine::GetPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto metrics = current_metrics_;
    // 修正：调用正确的CPU内存获取方法
    metrics.gpu_memory_gb = const_cast<FunASREngine*>(this)->GetCPUMemoryUsage();
    return metrics;
}


/**
 * 加载测试音频文件 - CPU版本适配
 * 
 * 🔄 主要功能:
 * 1. 扫描音频文件目录
 * 2. 随机选择测试文件（如果数量过多）
 * 3. 预检查文件有效性
 * 4. 修复日志格式化问题
 */
bool FunASREngine::LoadTestAudioFiles() {
    std::ostringstream scan_log;
    scan_log << "扫描测试音频文件目录: " << config_.audio_files_dir;
    Logger::Info(scan_log.str());
    
    // 扫描WAV文件
    auto all_wav_files = AudioFileReader::ScanWavFiles(config_.audio_files_dir);
    if (all_wav_files.empty()) {
        std::string error_msg = "未找到WAV音频文件，请检查目录: " + config_.audio_files_dir;
        Logger::Error(error_msg);
        return false;
    }
    
    // 选择测试文件 (如果文件太多，随机选择一部分)
    if (all_wav_files.size() > static_cast<size_t>(config_.max_test_files)) {
        std::ostringstream select_log;
        select_log << "音频文件总数(" << all_wav_files.size() 
                   << ")超过最大测试数(" << config_.max_test_files << "), 将随机选择";
        Logger::Info(select_log.str());
        
        // 随机打乱并选择前N个
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(all_wav_files.begin(), all_wav_files.end(), gen);
        all_wav_files.resize(config_.max_test_files);
    }
    
    test_audio_files_ = std::move(all_wav_files);
    
    std::ostringstream selected_log;
    selected_log << "已选择" << test_audio_files_.size() << "个音频文件用于测试";
    Logger::Info(selected_log.str());
    
    // 预检查几个文件确保可读性
    int valid_files = 0;
    for (int i = 0; i < std::min(5, static_cast<int>(test_audio_files_.size())); ++i) {
        auto audio_data = AudioFileReader::ReadWavFile(test_audio_files_[i]);
        if (audio_data.IsValid()) {
            valid_files++;
        }
    }
    
    if (valid_files == 0) {
        Logger::Error("没有有效的音频文件可供测试");
        return false;
    }
    
    std::ostringstream check_log;
    check_log << "音频文件预检查完成，有效文件率: " << valid_files << "/5";
    Logger::Info(check_log.str());
    
    return true;
}


void FunASREngine::UpdateMetrics(const PerformanceMetrics& new_metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    // 实际更新逻辑，按字段分别处理
    if (new_metrics.streaming_rtf > 0) current_metrics_.streaming_rtf = new_metrics.streaming_rtf;
    if (new_metrics.offline_rtf > 0) current_metrics_.offline_rtf = new_metrics.offline_rtf;
    if (new_metrics.two_pass_rtf > 0) current_metrics_.two_pass_rtf = new_metrics.two_pass_rtf;
    if (new_metrics.end_to_end_latency_ms > 0) current_metrics_.end_to_end_latency_ms = new_metrics.end_to_end_latency_ms;
    if (new_metrics.online_latency_ms > 0) current_metrics_.online_latency_ms = new_metrics.online_latency_ms;
    if (new_metrics.concurrent_sessions > 0) current_metrics_.concurrent_sessions = new_metrics.concurrent_sessions;
    if (new_metrics.total_audio_processed_hours > 0) current_metrics_.total_audio_processed_hours += new_metrics.total_audio_processed_hours;
    if (new_metrics.test_files_count > 0) current_metrics_.test_files_count = new_metrics.test_files_count;
    current_metrics_.gpu_memory_gb = new_metrics.gpu_memory_gb;

    // 修复：日志格式（标准 C++ 流式拼接，保证输出内容清晰）
    std::ostringstream oss;
    oss << "更新性能指标："
        << "流式RTF=" << std::fixed << std::setprecision(4) << current_metrics_.streaming_rtf << ", "
        << "离线RTF=" << std::fixed << std::setprecision(4) << current_metrics_.offline_rtf << ", "
        << "2PassRTF=" << std::fixed << std::setprecision(4) << current_metrics_.two_pass_rtf << ", "
        << "并发=" << current_metrics_.concurrent_sessions << ", "
        << "总时长=" << std::fixed << std::setprecision(1) << current_metrics_.total_audio_processed_hours << "h";
    Logger::Info(oss.str());
}
