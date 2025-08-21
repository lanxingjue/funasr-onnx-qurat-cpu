#include "funasr_engine.h"
#include <random>
#include <cmath>
#include <future>

FunASREngine::FunASREngine(const Config& config) : config_(config) {
    Logger::Info("创建FunASR引擎，设备: {}, 音频目录: {}", config_.device, config_.audio_files_dir);
}

FunASREngine::~FunASREngine() {
    testing_active_ = false;
    if (test_thread_.joinable()) {
        test_thread_.join();
    }
    Logger::Info("FunASR引擎已销毁");
}

bool FunASREngine::Initialize() {
    Logger::Info("初始化FunASR引擎...");
    Timer init_timer;
    
    try {
        // 1. 初始化Python环境
        if (!InitializePython()) {
            Logger::Error("Python环境初始化失败");
            return false;
        }
        
        // 2. 加载所有FunASR模型 (按照FunASR WebSocket服务器的标准配置)
        Logger::Info("加载FunASR模型组件...");
        
        // 加载流式ASR模型 (对应FunASR WebSocket中的model_asr_streaming)
        if (!LoadFunASRModel("streaming_asr", config_.streaming_model, 
                            config_.streaming_revision, streaming_model_)) {
            Logger::Error("流式ASR模型加载失败");
            return false;
        }
        
        // 加载离线ASR模型 (对应FunASR WebSocket中的model_asr)
        if (!LoadFunASRModel("offline_asr", config_.offline_model, 
                            config_.offline_revision, offline_model_)) {
            Logger::Error("离线ASR模型加载失败");
            return false;
        }
        
        // 加载VAD模型 (对应FunASR WebSocket中的model_vad)
        if (!LoadFunASRModel("vad", config_.vad_model, 
                            config_.vad_revision, vad_model_)) {
            Logger::Error("VAD模型加载失败");
            return false;
        }
        
        // 加载标点符号模型 (对应FunASR WebSocket中的model_punc)
        if (!LoadFunASRModel("punctuation", config_.punc_model, 
                            config_.punc_revision, punc_model_)) {
            Logger::Error("标点符号模型加载失败");
            return false;
        }
        
        // 3. 检查GPU状态
        double gpu_memory = GetGPUMemoryUsage();
        Logger::Info("GPU显存使用: {:.1f}GB", gpu_memory);
        
        // 4. 加载测试音频文件
        if (!LoadTestAudioFiles()) {
            Logger::Error("加载测试音频文件失败");
            return false;
        }
        
        // 5. 初始化性能指标
        current_metrics_.gpu_memory_gb = gpu_memory;
        current_metrics_.test_files_count = test_audio_files_.size();
        
        initialized_ = true;
        Logger::Info("FunASR引擎初始化完成，耗时: {:.1f}ms", init_timer.ElapsedMs());
        Logger::Info("已加载模型: 流式ASR + 离线ASR + VAD + 标点符号");
        Logger::Info("测试音频文件: {}个", test_audio_files_.size());
        
        return true;
        
    } catch (const std::exception& e) {
        Logger::Error("引擎初始化异常: {}", e.what());
        return false;
    }
}

bool FunASREngine::InitializePython() {
    try {
        // 创建Python解释器（pybind11核心功能）
        py_guard_ = std::make_unique<py::scoped_interpreter>();
        
        // 导入必要模块
        py::module_ sys = py::module_::import("sys");
        py::module_ funasr = py::module_::import("funasr");
        py::module_ torch = py::module_::import("torch");
        
        // 检查CUDA可用性
        bool cuda_available = torch.attr("cuda").attr("is_available")().cast<bool>();
        if (!cuda_available) {
            Logger::Error("CUDA不可用！请检查PyTorch CUDA安装");
            return false;
        }
        
        // 设置CUDA设备
        int device_id = 0;
        if (config_.device.find("cuda:") == 0) {
            device_id = std::stoi(config_.device.substr(5));
        }
        torch.attr("cuda").attr("set_device")(device_id);
        
        // 检查GPU设备信息
        py::object device_name = torch.attr("cuda").attr("get_device_name")(device_id);
        Logger::Info("Python环境初始化成功");
        Logger::Info("CUDA设备: {} - {}", device_id, device_name.cast<std::string>());
        
        return true;
        
    } catch (const py::error_already_set& e) {
        Logger::Error("Python初始化失败: {}", e.what());
        return false;
    }
}

bool FunASREngine::LoadFunASRModel(const std::string& model_type, 
                                   const std::string& model_name,
                                   const std::string& model_revision,
                                   py::object& model_obj) {
    try {
        Logger::Info("加载{}模型: {} (版本: {})", model_type, model_name, model_revision);
        Timer load_timer;
        
        // 使用FunASR的AutoModel类 (对应FunASR标准用法)
        py::module_ funasr = py::module_::import("funasr");
        py::object auto_model = funasr.attr("AutoModel");
        
        // 构建模型参数 (严格按照FunASR WebSocket服务器的配置)
        py::dict kwargs;
        kwargs["model"] = model_name;
        kwargs["model_revision"] = model_revision;
        kwargs["device"] = config_.device;
        kwargs["ngpu"] = 1;               // 使用1个GPU
        kwargs["ncpu"] = 4;               // 使用4个CPU核心 (对应FunASR默认)
        kwargs["disable_pbar"] = true;    // 禁用进度条
        kwargs["disable_log"] = true;     // 禁用冗余日志
        
        // 针对不同模型类型的特殊配置
        if (model_type == "streaming_asr") {
            // 流式ASR模型无需特殊配置，使用默认参数
        } else if (model_type == "offline_asr") {
            // 离线ASR模型无需特殊配置
        } else if (model_type == "vad") {
            // VAD模型可以设置分块大小
            // kwargs["chunk_size"] = 60;  // 可选参数
        } else if (model_type == "punctuation") {
            // 标点符号模型无需特殊配置
        }
        
        // 实例化模型
        model_obj = auto_model(**kwargs);
        
        Logger::Info("{}模型加载完成，耗时: {:.1f}ms", model_type, load_timer.ElapsedMs());
        return true;
        
    } catch (const py::error_already_set& e) {
        Logger::Error("{}模型加载失败: {}", model_type, e.what());
        return false;
    }
}

bool FunASREngine::LoadTestAudioFiles() {
    Logger::Info("扫描测试音频文件目录: {}", config_.audio_files_dir);
    
    // 扫描WAV文件
    auto all_wav_files = AudioFileReader::ScanWavFiles(config_.audio_files_dir);
    
    if (all_wav_files.empty()) {
        Logger::Error("未找到WAV音频文件，请检查目录: {}", config_.audio_files_dir);
        return false;
    }
    
    // 选择测试文件 (如果文件太多，随机选择一部分)
    if (all_wav_files.size() > static_cast<size_t>(config_.max_test_files)) {
        Logger::Info("音频文件总数({})超过最大测试数({}), 将随机选择", 
                    all_wav_files.size(), config_.max_test_files);
        
        // 随机打乱并选择前N个
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(all_wav_files.begin(), all_wav_files.end(), gen);
        all_wav_files.resize(config_.max_test_files);
    }
    
    test_audio_files_ = std::move(all_wav_files);
    
    Logger::Info("已选择{}个音频文件用于测试", test_audio_files_.size());
    
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
    
    Logger::Info("音频文件预检查完成，有效文件率: {}/5", valid_files);
    return true;
}

// ============ 核心识别功能实现 ============

FunASREngine::RecognitionResult FunASREngine::OfflineRecognize(
    const std::vector<float>& audio_data,
    bool enable_vad,
    bool enable_punctuation) {
    
    RecognitionResult result;
    if (!initialized_) {
        Logger::Error("引擎未初始化");
        return result;
    }
    
    try {
        Timer total_timer;
        std::string final_text;
        
        if (enable_vad && audio_data.size() > 16000 * 5) {  // 大于5秒的音频启用VAD
            Logger::Info("长音频检测，启用VAD分段处理");
            
            // 1. VAD分段
            std::map<std::string, py::object> vad_cache;
            auto vad_result = DetectVoiceActivity(audio_data, vad_cache);
            
            if (vad_result.HasValidSegments()) {
                Logger::Info("VAD检测到{}个语音段", vad_result.segments.size());
                
                // 2. 对每个语音段进行ASR识别
                std::vector<std::string> segment_texts;
                for (const auto& segment : vad_result.segments) {
                    int start_sample = (segment.first * 16000) / 1000;  // ms转换为样本
                    int end_sample = (segment.second * 16000) / 1000;
                    
                    if (start_sample >= 0 && end_sample <= static_cast<int>(audio_data.size()) && 
                        end_sample > start_sample) {
                        
                        std::vector<float> segment_audio(
                            audio_data.begin() + start_sample,
                            audio_data.begin() + end_sample
                        );
                        
                        // ASR识别单个段
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
                enable_vad = false;  // 回退到完整音频识别
            }
        }
        
        // 如果未启用VAD或VAD未检测到语音，使用完整音频识别
        if (!enable_vad || final_text.empty()) {
            py::array_t<float> audio_array = VectorToNumpy(audio_data);
            py::dict asr_kwargs;
            asr_kwargs["input"] = audio_array;
            
            py::object asr_result = offline_model_.attr("generate")(**asr_kwargs);
            auto parsed_result = ParseRecognitionResult(asr_result, 0);
            final_text = parsed_result.text;
        }
        
        // 3. 标点符号恢复
        if (enable_punctuation && !final_text.empty()) {
            std::map<std::string, py::object> punc_cache;
            final_text = AddPunctuation(final_text, punc_cache);
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
        
        Logger::Info("离线识别完成: '{}...', 耗时: {:.1f}ms", 
                    result.text.substr(0, 30), result.inference_time_ms);
        
    } catch (const std::exception& e) {
        Logger::Error("离线识别异常: {}", e.what());
        current_metrics_.total_requests++;
    }
    
    return result;
}

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
        
        // 1. 转换音频数据为numpy数组
        py::array_t<float> audio_array = VectorToNumpy(audio_chunk);
        
        // 2. 构建流式推理参数 (对应FunASR streaming generate参数)
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
        
        // 3. 添加缓存状态 (维持流式上下文)
        if (!session.streaming_cache.empty()) {
            py::dict cache_dict;
            for (const auto& item : session.streaming_cache) {
                cache_dict[py::str(item.first)] = item.second;
            }
            kwargs["cache"] = cache_dict;
        }
        
        // 4. 执行流式推理
        py::object py_result = streaming_model_.attr("generate")(**kwargs);
        
        // 5. 更新缓存状态 (重要：保持流式连续性) - 修复pybind11对象赋值问题
        if (kwargs.contains("cache")) {
            py::dict updated_cache = kwargs["cache"];
            session.streaming_cache.clear();
            for (auto item : updated_cache) {
                // 修复：使用reinterpret_borrow显式转换handle到object
                session.streaming_cache[item.first.cast<std::string>()] = 
                    py::reinterpret_borrow<py::object>(item.second);
            }
        }
        
        // 6. 解析结果
        result = ParseRecognitionResult(py_result, inference_timer.ElapsedMs());
        result.is_final = is_final;
        result.is_online_result = true;
        
        // 7. 更新性能指标
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
        
        Logger::Info("流式识别: '{}', 耗时: {:.1f}ms, RTF: {:.4f}", 
                    result.text, result.inference_time_ms, current_metrics_.streaming_rtf);
                    
    } catch (const std::exception& e) {
        Logger::Error("流式识别异常: {}", e.what());
        current_metrics_.total_requests++;
    }
    
    return result;
}

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
        
        // 1. 并行执行VAD检测和流式识别
        std::future<VADResult> vad_future = std::async(std::launch::async, [this, &audio_chunk, &session]() {
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
            
            // 触发离线精化识别
            Logger::Info("检测到语音结束，启动离线精化处理");
            
            // 提取完整语音段进行离线识别
            std::vector<float> complete_segment = session.audio_buffer;
            
            // 异步执行离线精化 (避免阻塞实时流)
            std::thread([this, complete_segment, &session, &results]() {
                Timer offline_timer;
                
                auto offline_result = OfflineRecognize(complete_segment, false, true);
                if (!offline_result.IsEmpty()) {
                    offline_result.is_offline_result = true;
                    offline_result.is_final = true;
                    
                    // 计算离线精化时间
                    current_metrics_.offline_refinement_ms = offline_timer.ElapsedMs();
                    
                    // 注意: 在实际应用中，需要通过回调或队列机制返回结果
                    Logger::Info("离线精化完成: '{}'", offline_result.text);
                }
                
                // 重置会话状态
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
        Logger::Error("2Pass识别异常: {}", e.what());
    }
}

FunASREngine::VADResult FunASREngine::DetectVoiceActivity(
    const std::vector<float>& audio_data,
    std::map<std::string, py::object>& vad_cache,
    int max_single_segment_time) {
    
    VADResult result;
    
    try {
        Timer vad_timer;
        
        // 转换音频数据
        py::array_t<float> audio_array = VectorToNumpy(audio_data);
        
        // 构建VAD参数 (对应FunASR VAD generate参数)
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
        
        // 执行VAD推理
        py::object vad_py_result = vad_model_.attr("generate")(**kwargs);
        
        // 更新缓存 - 修复pybind11对象赋值问题
        if (kwargs.contains("cache")) {
            py::dict updated_cache = kwargs["cache"];
            vad_cache.clear();
            for (auto item : updated_cache) {
                // 修复：使用reinterpret_borrow显式转换
                vad_cache[item.first.cast<std::string>()] = 
                    py::reinterpret_borrow<py::object>(item.second);
            }
        }
        
        result = ParseVADResult(vad_py_result, vad_timer.ElapsedMs());
        
    } catch (const std::exception& e) {
        Logger::Error("VAD检测异常: {}", e.what());
    }
    
    return result;
}

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
        
        // 执行标点符号恢复
        py::object punc_result = punc_model_.attr("generate")(**kwargs);
        
        // 更新缓存 - 修复pybind11对象赋值问题
        if (kwargs.contains("cache")) {
            py::dict updated_cache = kwargs["cache"];
            punc_cache.clear();
            for (auto item : updated_cache) {
                // 修复：使用reinterpret_borrow显式转换
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
                    
                    // 更新标点符号处理时间
                    current_metrics_.punctuation_ms = punc_timer.ElapsedMs();
                    
                    return punctuated_text;
                }
            }
        }
        
    } catch (const std::exception& e) {
        Logger::Error("标点符号处理异常: {}", e.what());
    }
    
    return text;  // 出错时返回原文本
}

// ============ 辅助方法实现 ============

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
        Logger::Error("识别结果解析异常: {}", e.what());
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
                        // 修复：显式转换handle到list
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
        Logger::Error("VAD结果解析异常: {}", e.what());
    }
    
    return parsed;
}

double FunASREngine::GetGPUMemoryUsage() {
    try {
        py::module_ torch = py::module_::import("torch");
        if (torch.attr("cuda").attr("is_available")().cast<bool>()) {
            py::object memory_allocated = torch.attr("cuda").attr("memory_allocated")(0);
            return memory_allocated.cast<size_t>() / (1024.0 * 1024.0 * 1024.0);
        }
    } catch (...) {
        // 静默处理异常
    }
    return 0.0;
}

// ============ 性能测试实现 ============

bool FunASREngine::RunPerformanceTests() {
    if (!initialized_ || test_audio_files_.empty()) {
        Logger::Error("引擎未初始化或无测试文件");
        return false;
    }
    
    Logger::Info("🧪 开始FunASR GPU完整性能测试套件...");
    Logger::Info("测试文件数: {}, GPU设备: {}", test_audio_files_.size(), config_.device);
    
    testing_active_ = true;
    
    // 在后台线程运行测试
    test_thread_ = std::thread([this]() {
        try {
            Timer total_test_timer;
            
            // 1. 离线识别性能测试
            if (config_.enable_offline_test) {
                Logger::Info("1️⃣ 离线识别性能测试...");
                auto offline_metrics = TestOfflinePerformance();
                UpdateMetrics(offline_metrics);
            }
            
            // 2. 流式识别性能测试
            if (config_.enable_streaming_test) {
                Logger::Info("2️⃣ 流式识别性能测试...");
                auto streaming_metrics = TestStreamingPerformance();
                UpdateMetrics(streaming_metrics);
            }
            
            // 3. 2Pass模式性能测试
            if (config_.enable_two_pass_test) {
                Logger::Info("3️⃣ 2Pass模式性能测试...");
                auto two_pass_metrics = TestTwoPassPerformance();
                UpdateMetrics(two_pass_metrics);
            }
            
            // 4. 并发性能测试
            if (config_.enable_concurrent_test) {
                Logger::Info("4️⃣ 并发性能测试...");
                auto concurrent_metrics = TestConcurrentPerformance();
                UpdateMetrics(concurrent_metrics);
            }
            
            Logger::Info("🎉 完整性能测试套件完成！总耗时: {:.1f}秒", total_test_timer.ElapsedMs() / 1000.0);
            
        } catch (const std::exception& e) {
            Logger::Error("性能测试异常: {}", e.what());
        }
        
        testing_active_ = false;
    });
    
    return true;
}

PerformanceMetrics FunASREngine::TestOfflinePerformance() {
    PerformanceMetrics metrics;
    
    // 选择前20个文件进行离线测试
    int test_count = std::min(20, static_cast<int>(test_audio_files_.size()));
    std::vector<double> rtf_values;
    double total_audio_duration = 0.0;
    
    Logger::Info("离线测试使用{}个音频文件", test_count);
    
    for (int i = 0; i < test_count; ++i) {
        const auto& file_path = test_audio_files_[i];
        
        // 读取音频文件
        auto audio_data = AudioFileReader::ReadWavFile(file_path);
        if (!audio_data.IsValid()) {
            Logger::Warn("跳过无效音频文件: {}", file_path);
            continue;
        }
        
        // 执行离线识别
        Timer test_timer;
        auto result = OfflineRecognize(audio_data.samples, true, true);
        double elapsed_ms = test_timer.ElapsedMs();
        
        if (!result.IsEmpty()) {
            double rtf = elapsed_ms / (audio_data.duration_seconds * 1000.0);
            rtf_values.push_back(rtf);
            total_audio_duration += audio_data.duration_seconds;
            
            Logger::Info("离线测试 [{}/{}]: {:.1f}秒音频, RTF={:.4f}, 结果: '{}'", 
                        i+1, test_count, audio_data.duration_seconds, rtf, 
                        result.text.substr(0, 30));
        }
    }
    
    // 计算平均RTF
    if (!rtf_values.empty()) {
        double sum = 0;
        for (double rtf : rtf_values) sum += rtf;
        metrics.offline_rtf = sum / rtf_values.size();
        metrics.total_audio_processed_hours = total_audio_duration / 3600.0;
        metrics.test_files_count = rtf_values.size();
    }
    
    Logger::Info("离线测试完成: 平均RTF={:.4f}, 处理音频={:.2f}小时", 
                metrics.offline_rtf, metrics.total_audio_processed_hours);
    
    return metrics;
}

PerformanceMetrics FunASREngine::TestStreamingPerformance() {
    PerformanceMetrics metrics;
    
    // 选择前15个文件进行流式测试
    int test_count = std::min(15, static_cast<int>(test_audio_files_.size()));
    std::vector<double> rtf_values;
    std::vector<double> latency_values;
    
    Logger::Info("流式测试使用{}个音频文件", test_count);
    
    for (int i = 0; i < test_count; ++i) {
        const auto& file_path = test_audio_files_[i];
        
        auto audio_data = AudioFileReader::ReadWavFile(file_path);
        if (!audio_data.IsValid()) continue;
        
        // 模拟流式处理：将音频分成600ms的块
        auto chunks = SimulateStreamingChunks(audio_data.samples);
        TwoPassSession session;
        
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            bool is_final = (chunk_idx == chunks.size() - 1);
            auto result = StreamingRecognize(chunks[chunk_idx], session, is_final);
            
            if (!result.IsEmpty()) {
                double chunk_duration_ms = chunks[chunk_idx].size() / 16.0;  // 16kHz
                double rtf = result.inference_time_ms / chunk_duration_ms;
                rtf_values.push_back(rtf);
                latency_values.push_back(result.inference_time_ms);
            }
        }
        
        Logger::Info("流式测试 [{}/{}]: {:.1f}秒音频, {}个分块", 
                    i+1, test_count, audio_data.duration_seconds, chunks.size());
    }
    
    // 计算统计指标
    if (!rtf_values.empty()) {
        double sum_rtf = 0, sum_latency = 0;
        for (double rtf : rtf_values) sum_rtf += rtf;
        for (double lat : latency_values) sum_latency += lat;
        
        metrics.streaming_rtf = sum_rtf / rtf_values.size();
        metrics.online_latency_ms = sum_latency / latency_values.size();
        metrics.end_to_end_latency_ms = metrics.online_latency_ms;
    }
    
    Logger::Info("流式测试完成: 平均RTF={:.4f}, 平均延迟={:.1f}ms", 
                metrics.streaming_rtf, metrics.online_latency_ms);
    
    return metrics;
}

PerformanceMetrics FunASREngine::TestTwoPassPerformance() {
    PerformanceMetrics metrics;
    
    // 选择10个文件进行2Pass测试
    int test_count = std::min(10, static_cast<int>(test_audio_files_.size()));
    std::vector<double> rtf_values;
    
    Logger::Info("2Pass测试使用{}个音频文件", test_count);
    
    for (int i = 0; i < test_count; ++i) {
        const auto& file_path = test_audio_files_[i];
        
        auto audio_data = AudioFileReader::ReadWavFile(file_path);
        if (!audio_data.IsValid()) continue;
        
        // 模拟2Pass处理
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
        
        Logger::Info("2Pass测试 [{}/{}]: {:.1f}秒音频, RTF={:.4f}, 输出{}个结果", 
                    i+1, test_count, audio_data.duration_seconds, rtf, results.size());
    }
    
    // 计算平均RTF
    if (!rtf_values.empty()) {
        double sum = 0;
        for (double rtf : rtf_values) sum += rtf;
        metrics.two_pass_rtf = sum / rtf_values.size();
    }
    
    Logger::Info("2Pass测试完成: 平均RTF={:.4f}", metrics.two_pass_rtf);
    
    return metrics;
}

PerformanceMetrics FunASREngine::TestConcurrentPerformance() {
    PerformanceMetrics metrics;
    
    const int num_workers = config_.max_concurrent_sessions;
    const int files_per_worker = std::max(1, static_cast<int>(test_audio_files_.size()) / num_workers);
    
    Logger::Info("启动{}路并发测试，每路处理{}个文件", num_workers, files_per_worker);
    
    std::vector<std::future<void>> futures;
    std::vector<PerformanceMetrics> worker_results(num_workers);
    std::atomic<int> active_sessions{0};
    
    Timer concurrent_timer;
    
    // 启动并发工作线程
    for (int i = 0; i < num_workers; ++i) {
        int start_idx = i * files_per_worker;
        int end_idx = std::min(start_idx + files_per_worker, static_cast<int>(test_audio_files_.size()));
        
        std::vector<std::string> worker_files(
            test_audio_files_.begin() + start_idx,
            test_audio_files_.begin() + end_idx
        );
        
        futures.emplace_back(std::async(std::launch::async, [this, i, worker_files, &active_sessions, &worker_results]() {
            ConcurrentTestWorker(i, worker_files, active_sessions, worker_results);
        }));
    }
    
    // 等待所有线程完成
    for (auto& future : futures) {
        future.wait();
    }
    
    double total_time_s = concurrent_timer.ElapsedMs() / 1000.0;
    
    // 聚合结果
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
    
    Logger::Info("并发测试完成: {}路并发, 平均RTF={:.4f}, 总耗时={:.1f}秒", 
                num_workers, metrics.streaming_rtf, total_time_s);
    
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
        
        for (const auto& file_path : worker_files) {
            auto audio_data = AudioFileReader::ReadWavFile(file_path);
            if (!audio_data.IsValid()) continue;
            
            // 执行流式识别测试
            auto chunks = SimulateStreamingChunks(audio_data.samples);
            TwoPassSession session;
            
            Timer worker_timer;
            for (const auto& chunk : chunks) {
                auto result = StreamingRecognize(chunk, session, false);
            }
            double elapsed_ms = worker_timer.ElapsedMs();
            
            double rtf = elapsed_ms / (audio_data.duration_seconds * 1000.0);
            rtf_values.push_back(rtf);
        }
        
        // 计算该worker的平均RTF
        if (!rtf_values.empty()) {
            double sum = 0;
            for (double rtf : rtf_values) sum += rtf;
            worker_metrics.streaming_rtf = sum / rtf_values.size();
        }
        
        results[worker_id] = worker_metrics;
        active_sessions--;
        
        Logger::Info("并发Worker-{} 完成: 处理{}个文件, 平均RTF={:.4f}", 
                    worker_id, worker_files.size(), worker_metrics.streaming_rtf);
        
    } catch (const std::exception& e) {
        Logger::Error("并发Worker-{} 异常: {}", worker_id, e.what());
        active_sessions--;
    }
}

std::vector<std::vector<float>> FunASREngine::SimulateStreamingChunks(
    const std::vector<float>& audio_data, double chunk_duration_ms) {
    
    std::vector<std::vector<float>> chunks;
    const int chunk_samples = static_cast<int>((chunk_duration_ms / 1000.0) * 16000);  // 16kHz
    
    for (size_t i = 0; i < audio_data.size(); i += chunk_samples) {
        size_t end_idx = std::min(i + chunk_samples, audio_data.size());
        chunks.emplace_back(audio_data.begin() + i, audio_data.begin() + end_idx);
    }
    
    return chunks;
}

PerformanceMetrics FunASREngine::GetPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto metrics = current_metrics_;
    metrics.gpu_memory_gb = const_cast<FunASREngine*>(this)->GetGPUMemoryUsage();
    return metrics;
}

void FunASREngine::UpdateMetrics(const PerformanceMetrics& new_metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // 更新非零值
    if (new_metrics.streaming_rtf > 0) {
        current_metrics_.streaming_rtf = new_metrics.streaming_rtf;
    }
    if (new_metrics.offline_rtf > 0) {
        current_metrics_.offline_rtf = new_metrics.offline_rtf;
    }
    if (new_metrics.two_pass_rtf > 0) {
        current_metrics_.two_pass_rtf = new_metrics.two_pass_rtf;
    }
    if (new_metrics.end_to_end_latency_ms > 0) {
        current_metrics_.end_to_end_latency_ms = new_metrics.end_to_end_latency_ms;
    }
    if (new_metrics.online_latency_ms > 0) {
        current_metrics_.online_latency_ms = new_metrics.online_latency_ms;
    }
    if (new_metrics.concurrent_sessions > 0) {
        current_metrics_.concurrent_sessions = new_metrics.concurrent_sessions;
    }
    if (new_metrics.total_audio_processed_hours > 0) {
        current_metrics_.total_audio_processed_hours += new_metrics.total_audio_processed_hours;
    }
    if (new_metrics.test_files_count > 0) {
        current_metrics_.test_files_count = new_metrics.test_files_count;
    }
    
    current_metrics_.gpu_memory_gb = new_metrics.gpu_memory_gb;
}
