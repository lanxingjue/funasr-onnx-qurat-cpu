#include <iostream>
#include <signal.h>
#include <memory>
#include <thread>
#include "funasr_engine.h"

std::unique_ptr<FunASREngine> g_engine;

void SignalHandler(int signal) {
    Logger::Info("接收到停止信号，正在退出...");
    g_engine.reset();
    exit(0);
}

void PrintBanner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║      FunASR GPU 高性能语音转写系统 - 第一阶段 GPU原型版本                        ║
║                           基于真实音频文件测试                                   ║
║                                                                               ║
║  🎯 核心功能:                                                                  ║
║    📝 离线高精度识别: VAD分段 → ASR识别 → 标点符号恢复                           ║
║    ⚡ 实时流式识别: 600ms分块 → 实时ASR → 即时输出                              ║
║    🔄 2Pass混合模式: 实时反馈 + 离线精化 (对应FunASR WebSocket)                 ║
║    🧪 完整性能测试: 基于500个真实WAV文件的全面评估                               ║
║                                                                               ║
║  📊 测试指标:                                                                  ║
║    • 离线RTF (Real Time Factor)                                               ║
║    • 流式RTF 和端到端延迟                                                      ║
║    • 2Pass模式综合性能                                                        ║
║    • GPU显存使用和并发能力                                                     ║
║                                                                               ║
║  🎪 FunASR模型配置:                                                           ║
║    • paraformer-zh-streaming (实时流式)                                       ║
║    • paraformer-zh (离线精化)                                                 ║
║    • fsmn-vad (语音活动检测)                                                  ║
║    • ct-punc (标点符号恢复)                                                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        PrintBanner();
        
        // 解析命令行参数
        FunASREngine::Config config;
        bool show_help = false;
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                show_help = true;
            }
            else if (arg == "--gpu-id" && i + 1 < argc) {
                config.device = "cuda:" + std::string(argv[++i]);
            }
            else if (arg == "--audio-dir" && i + 1 < argc) {
                config.audio_files_dir = argv[++i];
            }
            else if (arg == "--max-files" && i + 1 < argc) {
                config.max_test_files = std::stoi(argv[++i]);
            }
            else if (arg == "--concurrent" && i + 1 < argc) {
                config.max_concurrent_sessions = std::stoi(argv[++i]);
            }
            else if (arg == "--test-offline-only") {
                config.enable_offline_test = true;
                config.enable_streaming_test = false;
                config.enable_two_pass_test = false;
                config.enable_concurrent_test = false;
            }
            else if (arg == "--test-streaming-only") {
                config.enable_offline_test = false;
                config.enable_streaming_test = true;
                config.enable_two_pass_test = false;
                config.enable_concurrent_test = false;
            }
            else if (arg == "--test-2pass-only") {
                config.enable_offline_test = false;
                config.enable_streaming_test = false;
                config.enable_two_pass_test = true;
                config.enable_concurrent_test = false;
            }
        }
        
        if (show_help) {
            std::cout << "FunASR GPU语音转写系统 - 使用说明\n\n";
            std::cout << "用法: " << argv[0] << " [选项]\n\n";
            std::cout << "基本选项:\n";
            std::cout << "  --gpu-id <N>           GPU设备ID (默认: 0)\n";
            std::cout << "  --audio-dir <路径>     音频文件目录 (默认: ./audio_files)\n";
            std::cout << "  --max-files <N>        最大测试文件数 (默认: 100)\n";
            std::cout << "  --concurrent <N>       最大并发数 (默认: 4)\n\n";
            std::cout << "测试模式:\n";
            std::cout << "  --test-offline-only    仅测试离线识别\n";
            std::cout << "  --test-streaming-only  仅测试流式识别\n";
            std::cout << "  --test-2pass-only      仅测试2Pass模式\n";
            std::cout << "  (默认运行所有测试)\n\n";
            std::cout << "示例:\n";
            std::cout << "  " << argv[0] << " --gpu-id 0 --audio-dir ./my_wavs --max-files 50\n";
            std::cout << "  " << argv << " --test-offline-only --concurrent 2\n";
            return 0;
        }
        
        // 设置信号处理
        Logger::SetLevel(Logger::INFO);
        signal(SIGINT, SignalHandler);
        signal(SIGTERM, SignalHandler);
        
        Logger::Info("🚀 启动FunASR GPU引擎...");
        Logger::Info("配置信息:");
        Logger::Info("  GPU设备: {}", config.device);
        Logger::Info("  音频目录: {}", config.audio_files_dir);
        Logger::Info("  最大测试文件: {}", config.max_test_files);
        Logger::Info("  最大并发数: {}", config.max_concurrent_sessions);
        
        // 显示测试计划
        std::cout << "\n📋 测试计划:\n";
        if (config.enable_offline_test) std::cout << "  ✅ 离线识别性能测试\n";
        if (config.enable_streaming_test) std::cout << "  ✅ 流式识别性能测试\n";
        if (config.enable_two_pass_test) std::cout << "  ✅ 2Pass模式性能测试\n";
        if (config.enable_concurrent_test) std::cout << "  ✅ 并发性能测试\n";
        std::cout << std::endl;
        
        // 创建和初始化引擎
        g_engine = std::make_unique<FunASREngine>(config);
        
        Logger::Info("📥 正在初始化FunASR引擎和加载模型...");
        if (!g_engine->Initialize()) {
            Logger::Error("❌ 引擎初始化失败");
            return -1;
        }
        
        Logger::Info("✅ 引擎初始化成功!");
        
        // 运行性能测试
        Logger::Info("🧪 启动性能测试套件...");
        
        if (!g_engine->RunPerformanceTests()) {
            Logger::Error("性能测试启动失败");
            return -1;
        }
        
        // 监控测试进度
        Logger::Info("📊 性能测试进行中，按Ctrl+C可提前结束...");
        
        int progress_count = 0;
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            progress_count++;
            
            auto metrics = g_engine->GetPerformanceMetrics();
            
            Logger::Info("📈 进度报告 [{}0秒] - 流式RTF: {:.4f}, 离线RTF: {:.4f}, GPU: {:.1f}GB, 成功率: {:.1f}%",
                        progress_count, metrics.streaming_rtf, metrics.offline_rtf, 
                        metrics.gpu_memory_gb, metrics.GetSuccessRate());
            
            // 检查是否所有测试都完成了 (简化判断)
            if (progress_count >= 30) {  // 最多等待5分钟
                Logger::Info("测试时间到达，生成最终报告");
                break;
            }
        }
        
        // 输出最终性能报告
        auto final_metrics = g_engine->GetPerformanceMetrics();
        std::cout << "\n" << final_metrics.ToString() << std::endl;
        
        // 保存性能报告到文件
        std::ofstream report_file("funasr_gpu_performance_report.txt");
        if (report_file.is_open()) {
            report_file << final_metrics.ToString();
            report_file.close();
            Logger::Info("📄 性能报告已保存到: funasr_gpu_performance_report.txt");
        }
        
        Logger::Info("🎉 FunASR GPU第一阶段测试完成！");
        
        return 0;
        
    } catch (const std::exception& e) {
        Logger::Error("程序异常: {}", e.what());
        return -1;
    }
}
