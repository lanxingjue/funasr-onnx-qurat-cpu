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
 * FunASR CPU引擎 - 第二阶段完整CPU版本适配
 * 
 * 🔄 核心改造 (从GPU版本适配而来):
 * 1. 设备切换: GPU推理 → CPU多线程推理
 * 2. 并发提升: 4路并发 → 16路+并发
 * 3. 资源优化: GPU显存管理 → CPU内存管理
 * 4. 错误修复: 段错误修复 + 日志格式化修复
 * 5. 音频增强: 增加重采样功能 (24kHz→16kHz)
 * 
 * 🎯 适配目标:
 * - 保持原有功能完整性 (离线+流式+2Pass)
 * - 提升系统稳定性和并发能力
 * - 降低硬件门槛 (无需GPU)
 * - 为第三阶段ONNX优化打基础
 * 
 * 📊 预期性能对比:
 * GPU版本 → CPU版本:
 * ├── RTF: 0.08 → 0.15-0.25 (性能下降但可接受)
 * ├── 并发: 4路 → 16路+ (并发能力大幅提升)
 * ├── 稳定性: 段错误风险 → 稳定运行
 * └── 部署成本: 高(需GPU) → 低(仅需CPU)
 */
class FunASREngine {
public:
    /**
     * 识别结果结构体 (保持与GPU版本一致)
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
     * VAD检测结果结构体 (保持与GPU版本一致)
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
     * 2Pass会话状态 (保持与GPU版本一致)
     * 对应FunASR WebSocket服务器的会话管理
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
     * CPU版本引擎配置 - 核心改造点
     * 
     * 🔄 主要变化:
     * 1. device: "cuda:0" → "cpu"
     * 2. 增加CPU相关配置项
     * 3. 增加音频处理配置
     * 4. 提升并发能力上限
     */
    struct Config {
        // ============ 设备配置 (核心改造) ============
        std::string device;                       // "cpu" 替代 "cuda:0"
        int cpu_threads;                          // CPU线程数 (新增)
        bool enable_audio_resampling;             // 启用音频重采样 (新增)
        bool enable_cpu_optimization;             // 启用CPU优化 (新增)
        
        // ============ 音频文件配置 (保持不变) ============
        std::string audio_files_dir;              // 音频文件目录
        int max_test_files;                       // 最大测试文件数
        
        // ============ 测试配置 (并发能力提升) ============
        bool enable_offline_test;                 // 启用离线识别测试
        bool enable_streaming_test;               // 启用流式识别测试
        bool enable_two_pass_test;                // 启用2Pass模式测试
        bool enable_concurrent_test;              // 启用并发测试
        int max_concurrent_sessions;              // 最大并发数 (4→16)
        
        // ============ FunASR模型配置 (保持不变) ============
        std::string streaming_model;              // 流式ASR模型路径
        std::string streaming_revision;           // 流式ASR模型版本
        std::string offline_model;                // 离线ASR模型路径
        std::string offline_revision;             // 离线ASR模型版本
        std::string vad_model;                    // VAD模型路径
        std::string vad_revision;                 // VAD模型版本
        std::string punc_model;                   // 标点符号模型路径
        std::string punc_revision;                // 标点符号模型版本
        
        /**
         * CPU版本默认配置构造函数
         * 
         * 🎯 关键改造点:
         * 1. device = "cpu" (替代GPU)
         * 2. cpu_threads = 自动检测CPU核数
         * 3. max_concurrent_sessions = 16 (提升并发)
         * 4. 启用音频重采样和CPU优化
         */
        Config() :
            // 核心改造: 设备从GPU切换到CPU
            device("cpu"),                        // 🔄 "cuda:0" → "cpu"
            cpu_threads(std::thread::hardware_concurrency()), // 🆕 自动检测CPU核数
            enable_audio_resampling(true),        // 🆕 启用音频重采样 (解决24kHz问题)
            enable_cpu_optimization(true),        // 🆕 启用CPU性能优化
            
            // 音频文件配置 (保持不变)
            audio_files_dir("./audio_files"),
            max_test_files(100),
            
            // 测试配置 (提升并发能力)
            enable_offline_test(true),
            enable_streaming_test(true),
            enable_two_pass_test(true),
            enable_concurrent_test(true),
            max_concurrent_sessions(32),          // 🔄 4 → 16 (CPU可支持更多并发)
            
            // FunASR模型配置 (保持完全一致)
            streaming_model("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"),
            streaming_revision("v2.0.4"),
            offline_model("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
            offline_revision("v2.0.4"),
            vad_model("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            vad_revision("v2.0.4"),
            punc_model("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"),
            punc_revision("v2.0.4")
        {}
    };

    explicit FunASREngine(const Config& config = Config{});
    ~FunASREngine();

    /**
     * 初始化引擎 - 适配CPU模式
     * 
     * 🔄 主要改造:
     * 1. 跳过CUDA检查和GPU设备设置
     * 2. 设置PyTorch CPU线程数
     * 3. 强制禁用CUDA使用
     * 4. 增加CPU性能优化
     */
    bool Initialize();

    /**
     * 离线语音识别 - 增强版
     * 
     * 🔄 CPU版本改进:
     * 1. 增加音频重采样 (24kHz→16kHz)
     * 2. 增强错误处理和异常捕获
     * 3. 优化VAD分段处理逻辑
     * 4. CPU多线程优化
     * 
     * 流程: 音频预处理 → VAD分段 → ASR识别 → 标点符号恢复
     * 
     * @param audio_data 完整音频数据 (支持24kHz自动重采样到16kHz)
     * @param enable_vad 是否启用VAD分段 (长音频推荐开启)
     * @param enable_punctuation 是否添加标点符号
     */
    RecognitionResult OfflineRecognize(
        const std::vector<float>& audio_data,
        bool enable_vad = false,
        bool enable_punctuation = true
    );

    /**
     * 实时流式识别 - CPU多线程优化版
     * 
     * 🔄 CPU版本改进:
     * 1. CPU多线程推理替代GPU推理
     * 2. 优化内存管理和对象生命周期
     * 3. 增强缓存状态管理
     * 
     * 对应FunASR WebSocket streaming模式
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
     * 2Pass混合识别 - CPU并行优化版
     * 
     * 🔄 CPU版本改进:
     * 1. CPU多核心并行处理VAD和流式识别
     * 2. 优化异步离线精化处理
     * 3. 增强线程安全和资源管理
     * 
     * 流程: 实时流式识别(快速反馈) + VAD端点检测 + 离线精化识别(高精度)
     * 对应FunASR WebSocket 2pass模式
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
     * VAD语音活动检测 - CPU优化版
     * 
     * 🔄 CPU版本改进:
     * 1. CPU多线程VAD推理
     * 2. 优化内存使用和处理效率
     * 3. 增强错误处理
     * 
     * 对应FunASR VAD模型
     */
    VADResult DetectVoiceActivity(
        const std::vector<float>& audio_data,
        std::map<std::string, py::object>& vad_cache,
        int max_single_segment_time = 30000  // 最大分段时长(毫秒)
    );

    /**
     * 标点符号恢复 - CPU优化版
     * 
     * 🔄 CPU版本改进:
     * 1. CPU文本处理优化
     * 2. 增强异常处理
     * 3. 优化缓存管理
     * 
     * 对应FunASR标点符号模型
     */
    std::string AddPunctuation(
        const std::string& text,
        std::map<std::string, py::object>& punc_cache
    );

    /**
     * 运行完整性能测试套件 - CPU版本
     * 
     * 🔄 CPU版本改进:
     * 1. 测试CPU多线程并发能力
     * 2. 评估CPU资源使用效率
     * 3. 对比GPU版本性能差异
     * 
     * 基于真实音频文件进行全面测试
     */
    bool RunPerformanceTests();

    /**
     * 获取当前性能指标 - 适配CPU监控
     */
    PerformanceMetrics GetPerformanceMetrics() const;

    /**
     * 检查是否已初始化
     */
    bool IsInitialized() const { return initialized_; }

    /**
     * 检查测试是否正在运行 - 修复访问权限
     */
    bool IsTestingActive() const { return testing_active_; }

private:
    Config config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> testing_active_{false};
    
    // Python解释器和模型实例 (保持不变)
    std::unique_ptr<py::scoped_interpreter> py_guard_;
    py::object streaming_model_;     // 流式ASR模型
    py::object offline_model_;       // 离线ASR模型
    py::object vad_model_;          // VAD模型
    py::object punc_model_;         // 标点符号模型
    
    // 性能数据 (保持不变)
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics current_metrics_;
    
    // 测试相关 (保持不变)
    std::thread test_thread_;
    std::vector<std::string> test_audio_files_;  // 测试音频文件列表

    // ============ 核心私有方法 - CPU版本适配 ============

    /**
     * 初始化Python环境 - CPU模式适配
     * 
     * 🔄 主要改造:
     * 1. 跳过CUDA可用性检查
     * 2. 设置PyTorch使用CPU
     * 3. 配置CPU线程数
     * 4. 禁用CUDA警告
     */
    bool InitializePython();

    /**
     * 加载单个FunASR模型 - CPU配置
     * 
     * 🔄 主要改造:
     * 1. device="cpu" (替代cuda设备)
     * 2. ngpu=0 (不使用GPU)
     * 3. ncpu=auto (使用所有CPU核心)
     * 4. 增加disable_update=true (减少网络依赖)
     */
    bool LoadFunASRModel(
        const std::string& model_type, 
        const std::string& model_name,
        const std::string& model_revision,
        py::object& model_obj
    );

    /**
     * C++ vector转numpy数组 - 零拷贝 (保持不变)
     */
    py::array_t<float> VectorToNumpy(const std::vector<float>& data);

    /**
     * 解析FunASR识别结果 (保持不变)
     */
    RecognitionResult ParseRecognitionResult(const py::object& result, double inference_time_ms);

    /**
     * 解析VAD结果 (保持不变)
     */
    VADResult ParseVADResult(const py::object& result, double inference_time_ms);

    // ============ CPU版本专用方法 (新增) ============

    /**
     * 获取CPU内存使用量 - 替代GetGPUMemoryUsage()
     * 
     * 🆕 CPU版本新增功能:
     * 读取/proc/meminfo获取系统内存使用情况
     */
    double GetCPUMemoryUsage();

    /**
     * 音频重采样 - 解决24kHz音频问题
     * 
     * 🆕 CPU版本新增功能:
     * 支持任意采样率到16kHz的重采样转换
     * 使用线性插值算法实现高质量重采样
     * 
     * @param audio_data 原始音频数据
     * @param from_rate 源采样率
     * @param to_rate 目标采样率
     * @return 重采样后的音频数据
     */
    std::vector<float> ResampleAudio(
        const std::vector<float>& audio_data, 
        int from_rate, 
        int to_rate
    );

    /**
     * CPU性能优化 - 系统级优化
     * 
     * 🆕 CPU版本新增功能:
     * 1. 设置环境变量优化多线程库
     * 2. 配置OpenMP线程数
     * 3. 设置MKL线程数
     * 4. 进程优先级调整
     */
    void OptimizeCPUPerformance();

    /**
     * 更新性能指标 (保持不变)
     */
    void UpdateMetrics(const PerformanceMetrics& new_metrics);

    /**
     * 加载测试音频文件 (保持不变)
     */
    bool LoadTestAudioFiles();

    // ============ 性能测试方法 - CPU版本优化 ============

    /**
     * 离线识别性能测试 - CPU版本
     * 
     * 🔄 CPU版本改进:
     * 1. 测试CPU多核心处理能力
     * 2. 评估重采样功能性能影响
     * 3. 对比不同音频时长的处理效率
     */
    PerformanceMetrics TestOfflinePerformance();

    /**
     * 流式识别性能测试 - CPU版本
     * 
     * 🔄 CPU版本改进:
     * 1. 测试CPU实时处理能力
     * 2. 评估流式缓存管理效率
     * 3. 测试不同分块大小的性能影响
     */
    PerformanceMetrics TestStreamingPerformance();

    /**
     * 2Pass模式性能测试 - CPU版本
     * 
     * 🔄 CPU版本改进:
     * 1. 测试CPU并行处理VAD和ASR
     * 2. 评估异步离线精化性能
     * 3. 测试复杂场景下的稳定性
     */
    PerformanceMetrics TestTwoPassPerformance();

    /**
     * 并发性能测试 - CPU版本重点功能
     * 
     * 🔄 CPU版本改进:
     * 1. 测试16路+高并发处理能力
     * 2. 评估CPU多核心利用率
     * 3. 测试内存使用和资源竞争
     * 4. 对比GPU版本的并发优势
     */
    PerformanceMetrics TestConcurrentPerformance();

    /**
     * 并发测试工作线程 - CPU线程亲和性优化
     * 
     * 🔄 CPU版本改进:
     * 1. 可选的CPU核心绑定
     * 2. 优化线程间资源竞争
     * 3. 增强异常处理和资源清理
     */
    void ConcurrentTestWorker(
        int worker_id,
        const std::vector<std::string>& worker_files,
        std::atomic<int>& active_sessions,
        std::vector<PerformanceMetrics>& results
    );

    /**
     * 模拟流式音频处理 - 将完整音频分块处理 (保持不变)
     */
    std::vector<std::vector<float>> SimulateStreamingChunks(
        const std::vector<float>& audio_data,
        double chunk_duration_ms = 600.0  // 默认600ms分块
    );
};
