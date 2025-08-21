#pragma once
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "utils.h"

namespace py = pybind11;

/**
 * FunASR GPU引擎 - 完整的离线+实时2Pass语音识别系统
 * 
 * 功能架构:
 * 1. 离线高精度识别: VAD分段 → ASR识别 → 标点符号恢复
 * 2. 实时流式识别: 流式VAD → 实时ASR → 实时输出
 * 3. 2Pass混合模式: 实时输出 + 离线精化
 * 4. 完整性能测试: 基于真实音频文件的全面性能评估
 * 
 * 模型配置 (对应FunASR标准模型):
 * - paraformer-zh-streaming: 实时流式ASR模型
 * - paraformer-zh: 离线高精度ASR模型  
 * - fsmn-vad: 语音活动检测模型
 * - ct-punc: 标点符号恢复模型
 */
class FunASREngine {
public:
    /**
     * 识别结果结构体
     */
    struct RecognitionResult {
        std::string text;                     // 识别文本
        bool is_final = false;                // 是否最终结果
        double inference_time_ms = 0.0;       // 推理时间
        
        // 2Pass模式专用字段
        bool is_online_result = false;        // 是否为在线结果
        bool is_offline_result = false;       // 是否为离线精化结果
        
        bool IsEmpty() const { return text.empty(); }
    };
    
    /**
     * VAD检测结果
     */
    struct VADResult {
        std::vector<std::pair<int64_t, int64_t>> segments;  // 语音段 [开始ms, 结束ms]
        int64_t speech_start_ms = -1;         // 当前语音开始位置
        int64_t speech_end_ms = -1;           // 当前语音结束位置
        double inference_time_ms = 0.0;       // VAD推理时间
        bool has_speech = false;              // 是否检测到语音
        
        bool HasValidSegments() const { return !segments.empty(); }
    };
    
    /**
     * 2Pass会话状态 (对应FunASR WebSocket服务器的会话管理)
     */
    struct TwoPassSession {
        // 流式状态 (对应FunASR streaming cache)
        std::map<std::string, py::object> streaming_cache;
        std::map<std::string, py::object> vad_cache;
        std::map<std::string, py::object> punc_cache;
        
        // 音频缓冲区
        std::vector<float> audio_buffer;      // 完整音频缓冲
        std::vector<float> current_segment;   // 当前语音段
        
        // 状态控制 (对应FunASR WebSocket协议)
        bool is_speaking = false;             // 是否正在说话
        bool is_final = false;                // 是否结束
        int vad_pre_idx = 0;                 // VAD预处理索引
        
        // 流式配置 (对应FunASR streaming参数)
        std::vector<int> chunk_size = {0, 10, 5};  // [0,10,5] = 600ms实时显示
        int encoder_chunk_look_back = 4;     // 编码器回看块数
        int decoder_chunk_look_back = 1;     // 解码器回看块数
        int chunk_interval = 10;             // 分块间隔
        
        void Reset() {
            streaming_cache.clear();
            vad_cache.clear();
            punc_cache.clear();
            audio_buffer.clear();
            current_segment.clear();
            is_speaking = false;
            is_final = false;
            vad_pre_idx = 0;
        }
    };
    
    /**
     * 引擎配置
     */
    struct Config {
        // GPU配置
        std::string device = "cuda:0";              // CUDA设备
        
        // 音频文件配置
        std::string audio_files_dir = "./audio_files";  // 音频文件目录
        int max_test_files = 100;                   // 最大测试文件数 (从500个中选择)
        
        // 测试配置
        bool enable_offline_test = true;            // 启用离线识别测试
        bool enable_streaming_test = true;          // 启用流式识别测试
        bool enable_two_pass_test = true;           // 启用2Pass模式测试
        bool enable_concurrent_test = true;         // 启用并发测试
        int max_concurrent_sessions = 4;            // 最大并发数
        
        // FunASR模型配置 (标准ModelScope路径)
        std::string streaming_model = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online";
        std::string streaming_revision = "v2.0.4";
        std::string offline_model = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch";
        std::string offline_revision = "v2.0.4";
        std::string vad_model = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";
        std::string vad_revision = "v2.0.4";
        std::string punc_model = "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727";
        std::string punc_revision = "v2.0.4";
    };

    explicit FunASREngine(const Config& config = Config{});
    ~FunASREngine();

    /**
     * 初始化引擎 - 加载所有必要的FunASR模型
     */
    bool Initialize();

    /**
     * 离线语音识别 (完整流程)
     * 流程: 音频输入 → VAD分段 → ASR识别 → 标点符号恢复
     * 
     * @param audio_data 完整音频数据 (16kHz, 单声道)
     * @param enable_vad 是否启用VAD分段 (长音频推荐开启)
     * @param enable_punctuation 是否添加标点符号
     */
    RecognitionResult OfflineRecognize(
        const std::vector<float>& audio_data,
        bool enable_vad = true,
        bool enable_punctuation = true
    );

    /**
     * 实时流式识别 (对应FunASR WebSocket streaming模式)
     * 
     * @param audio_chunk 音频块数据 (通常600ms)
     * @param session 会话状态 (维持流式上下文)
     * @param is_final 是否最后一块音频
     */
    RecognitionResult StreamingRecognize(
        const std::vector<float>& audio_chunk,
        TwoPassSession& session,
        bool is_final = false
    );

    /**
     * 2Pass混合识别 (对应FunASR WebSocket 2pass模式)
     * 流程: 实时流式识别(快速反馈) + VAD端点检测 + 离线精化识别(高精度)
     * 
     * @param audio_chunk 音频块数据
     * @param session 2Pass会话状态
     * @param results 输出结果列表 (可能包含多个结果)
     */
    void TwoPassRecognize(
        const std::vector<float>& audio_chunk,
        TwoPassSession& session,
        std::vector<RecognitionResult>& results
    );

    /**
     * VAD语音活动检测 (对应FunASR VAD模型)
     */
    VADResult DetectVoiceActivity(
        const std::vector<float>& audio_data,
        std::map<std::string, py::object>& vad_cache,
        int max_single_segment_time = 30000  // 最大分段时长(毫秒)
    );

    /**
     * 标点符号恢复 (对应FunASR标点符号模型)
     */
    std::string AddPunctuation(
        const std::string& text,
        std::map<std::string, py::object>& punc_cache
    );

    /**
     * 运行完整性能测试套件 (基于真实音频文件)
     */
    bool RunPerformanceTests();

    /**
     * 获取当前性能指标
     */
    PerformanceMetrics GetPerformanceMetrics() const;

    /**
     * 检查是否已初始化
     */
    bool IsInitialized() const { return initialized_; }

private:
    Config config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> testing_active_{false};
    
    // Python解释器和模型实例
    std::unique_ptr<py::scoped_interpreter> py_guard_;
    py::object streaming_model_;     // 流式ASR模型
    py::object offline_model_;       // 离线ASR模型
    py::object vad_model_;          // VAD模型
    py::object punc_model_;         // 标点符号模型
    
    // 性能数据
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics current_metrics_;
    
    // 测试相关
    std::thread test_thread_;
    std::vector<std::string> test_audio_files_;  // 测试音频文件列表

    /**
     * 初始化Python环境和FunASR模块
     */
    bool InitializePython();

    /**
     * 加载单个FunASR模型 (使用标准配置)
     */
    bool LoadFunASRModel(
        const std::string& model_type, 
        const std::string& model_name,
        const std::string& model_revision,
        py::object& model_obj
    );

    /**
     * C++ vector转numpy数组 (零拷贝)
     */
    py::array_t<float> VectorToNumpy(const std::vector<float>& data);

    /**
     * 解析FunASR识别结果
     */
    RecognitionResult ParseRecognitionResult(const py::object& result, double inference_time_ms);

    /**
     * 解析VAD结果
     */
    VADResult ParseVADResult(const py::object& result, double inference_time_ms);

    /**
     * 获取GPU内存使用量
     */
    double GetGPUMemoryUsage();

    /**
     * 更新性能指标
     */
    void UpdateMetrics(const PerformanceMetrics& new_metrics);

    /**
     * 加载测试音频文件
     */
    bool LoadTestAudioFiles();

    // ============ 性能测试方法 ============
    
    /**
     * 离线识别性能测试
     */
    PerformanceMetrics TestOfflinePerformance();

    /**
     * 流式识别性能测试  
     */
    PerformanceMetrics TestStreamingPerformance();

    /**
     * 2Pass模式性能测试
     */
    PerformanceMetrics TestTwoPassPerformance();

    /**
     * 并发性能测试
     */
    PerformanceMetrics TestConcurrentPerformance();

    /**
     * 并发测试工作线程
     */
    void ConcurrentTestWorker(
        int worker_id,
        const std::vector<std::string>& worker_files,
        std::atomic<int>& active_sessions,
        std::vector<PerformanceMetrics>& results
    );

    /**
     * 模拟流式音频处理 (将完整音频分块处理)
     */
    std::vector<std::vector<float>> SimulateStreamingChunks(
        const std::vector<float>& audio_data,
        double chunk_duration_ms = 600.0  // 默认600ms分块
    );
};
