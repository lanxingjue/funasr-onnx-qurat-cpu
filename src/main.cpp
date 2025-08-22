/**
 * FunASR CPU 语音识别系统 - 主程序入口
 * 
 * 🎯 核心功能:
 * - 完整的命令行参数解析
 * - 智能的配置管理和验证
 * - 全面的错误处理和异常捕获
 * - 实时进度监控和性能报告
 * - 优雅的程序退出和资源清理
 * 
 * 🔄 CPU版本特性:
 * - 自动检测CPU核心数
 * - 可配置的并发线程数
 * - CPU内存使用监控
 * - 音频重采样支持
 * - 多模式性能测试
 */

#include <iostream>
#include <signal.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include "funasr_engine.h"

// ============ 全局变量和工具函数 ============

std::unique_ptr<FunASREngine> g_engine;
std::atomic<bool> g_shutdown_requested{false};

/**
 * 信号处理函数 - 优雅退出
 * 支持 SIGINT (Ctrl+C) 和 SIGTERM 信号
 */
void SignalHandler(int signal) {
    std::ostringstream signal_log;
    signal_log << "接收到停止信号 (" << signal << ")，正在安全退出...";
    Logger::Info(signal_log.str());
    
    g_shutdown_requested = true;
    
    // 安全释放引擎资源
    if (g_engine) {
        Logger::Info("正在停止性能测试和清理资源...");
        g_engine.reset();
        Logger::Info("引擎资源已安全释放");
    }
    
    exit(0);
}

/**
 * 系统信息检测 - CPU版本专用
 * 检测系统CPU、内存等基本信息
 */
void DetectSystemInfo() {
    Logger::Info("========== 系统信息检测 ==========");
    
    // CPU核心数检测
    unsigned int cpu_cores = std::thread::hardware_concurrency();
    std::ostringstream cpu_log;
    cpu_log << "CPU核心数: " << cpu_cores << " 核";
    Logger::Info(cpu_log.str());
    
    // 可用内存检测 (Linux系统)
#ifdef __linux__
    try {
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            long mem_total = 0;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal:") == 0) {
                    sscanf(line.c_str(), "MemTotal: %ld kB", &mem_total);
                    break;
                }
            }
            if (mem_total > 0) {
                double mem_gb = mem_total / (1024.0 * 1024.0);
                std::ostringstream mem_log;
                mem_log << "系统总内存: " << std::fixed << std::setprecision(1) << mem_gb << "GB";
                Logger::Info(mem_log.str());
            }
        }
    } catch (...) {
        Logger::Warn("无法获取内存信息");
    }
#endif
    
    Logger::Info("==================================");
}

/**
 * 打印程序Banner - CPU版本专用设计
 */
void PrintBanner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                   🎙️  FunASR CPU 语音识别系统 v2.0                           ║
║                          高性能 · 低门槛 · 无GPU依赖                          ║
║                                                                               ║
║ 🎯 核心功能:                                                                  ║
║   📝 离线高精度识别: VAD分段 → ASR识别 → 标点符号恢复                         ║
║   ⚡ 实时流式识别: 600ms分块 → 实时ASR → 即时输出                             ║
║   🔄 2Pass混合模式: 实时反馈 + 离线精化                                        ║
║   🧪 完整性能测试: 基于真实音频文件的全面评估                                  ║
║                                                                               ║
║ 💡 CPU版本优势:                                                               ║
║   🚀 高并发支持: 最高144路并发处理                                            ║
║   💾 智能内存管理: 自动优化CPU/内存使用                                        ║
║   🔧 音频适配: 支持多种采样率自动重采样                                        ║
║   📊 详细监控: 实时RTF、延迟、成功率统计                                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;
}

/**
 * 打印帮助信息 - 完整的命令行参数说明
 */
void PrintHelp(const char* program_name) {
    std::cout << "\n🎙️  FunASR CPU语音识别系统 - 使用说明\n\n";
    
    std::cout << "📖 基本用法:\n";
    std::cout << "  " << program_name << " [选项]\n\n";
    
    std::cout << "🔧 设备配置选项:\n";
    std::cout << "  --cpu-threads <N>        设置CPU线程数 (默认: 自动检测)\n";
    std::cout << "  --concurrent <N>         设置最大并发会话数 (默认: 144)\n";
    std::cout << "  --enable-optimization    启用CPU性能优化 (默认: 开启)\n";
    std::cout << "  --disable-optimization   禁用CPU性能优化\n\n";
    
    std::cout << "📁 音频文件选项:\n";
    std::cout << "  --audio-dir <路径>       音频文件目录 (默认: ./audio_files)\n";
    std::cout << "  --max-files <N>          最大测试文件数 (默认: 100)\n";
    std::cout << "  --enable-resampling      启用音频重采样 (默认: 开启)\n";
    std::cout << "  --disable-resampling     禁用音频重采样\n\n";
    
    std::cout << "🧪 测试模式选项:\n";
    std::cout << "  --test-all               运行所有测试 (默认)\n";
    std::cout << "  --test-offline-only      仅测试离线识别\n";
    std::cout << "  --test-streaming-only    仅测试流式识别\n";
    std::cout << "  --test-2pass-only        仅测试2Pass模式\n";
    std::cout << "  --test-concurrent-only   仅测试并发性能\n\n";
    
    std::cout << "📊 输出控制选项:\n";
    std::cout << "  --report-file <文件>     性能报告输出文件 (默认: funasr_cpu_report.txt)\n";
    std::cout << "  --log-level <级别>       日志级别 [DEBUG|INFO|WARN|ERROR] (默认: INFO)\n";
    std::cout << "  --quiet                  静默模式，减少日志输出\n";
    std::cout << "  --verbose                详细模式，增加调试信息\n\n";
    
    std::cout << "ℹ️  其他选项:\n";
    std::cout << "  --help, -h               显示此帮助信息\n";
    std::cout << "  --version, -v            显示版本信息\n";
    std::cout << "  --system-info            显示系统信息并退出\n\n";
    
    std::cout << "💡 使用示例:\n";
    std::cout << "  # 基本使用 (使用默认配置)\n";
    std::cout << "  " << program_name << "\n\n";
    std::cout << "  # 自定义CPU线程数和音频目录\n";
    std::cout << "  " << program_name << " --cpu-threads 8 --audio-dir ./my_audio\n\n";
    std::cout << "  # 仅测试离线识别，启用详细日志\n";
    std::cout << "  " << program_name << " --test-offline-only --verbose\n\n";
    std::cout << "  # 高并发测试，自定义报告文件\n";
    std::cout << "  " << program_name << " --concurrent 32 --report-file performance.txt\n\n";
    
    std::cout << "📝 注意事项:\n";
    std::cout << "  • 音频文件须为16位PCM WAV格式\n";
    std::cout << "  • 建议CPU核心数 ≥ 4，内存 ≥ 8GB\n";
    std::cout << "  • 使用Ctrl+C可随时安全退出程序\n";
    std::cout << "  • 性能报告会自动保存到指定文件\n\n";
}

/**
 * 打印版本信息
 */
void PrintVersion() {
    std::cout << "\n🎙️  FunASR CPU语音识别系统\n";
    std::cout << "版本: 2.0.0 CPU Edition\n";
    std::cout << "构建日期: " << __DATE__ << " " << __TIME__ << "\n";
    std::cout << "支持的功能: 离线识别 | 流式识别 | 2Pass模式 | 并发处理\n";
    std::cout << "Python绑定: pybind11\n";
    std::cout << "模型支持: FunASR官方模型\n\n";
}

/**
 * 解析命令行参数 - 完整的参数处理
 * @param argc 参数数量
 * @param argv 参数数组
 * @param config 输出配置结构体
 * @param report_file 输出报告文件名
 * @return true表示继续执行，false表示退出程序
 */
bool ParseCommandLine(int argc, char* argv[], FunASREngine::Config& config, std::string& report_file) {
    report_file = "funasr_cpu_performance_report.txt"; // 默认报告文件名
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // 帮助和版本信息
        if (arg == "--help" || arg == "-h") {
            PrintHelp(argv[0]);
            return false;
        } 
        else if (arg == "--version" || arg == "-v") {
            PrintVersion();
            return false;
        }
        else if (arg == "--system-info") {
            DetectSystemInfo();
            return false;
        }
        
        // CPU和性能配置
        else if (arg == "--cpu-threads" && i + 1 < argc) {
            int threads = std::stoi(argv[++i]);
            if (threads > 0 && threads <= 256) {
                config.cpu_threads = threads;
                Logger::Info("设置CPU线程数: {}", threads);
            } else {
                Logger::Error("无效的CPU线程数: {}，应在1-256之间", threads);
                return false;
            }
        }
        else if (arg == "--concurrent" && i + 1 < argc) {
            int concurrent = std::stoi(argv[++i]);
            if (concurrent > 0 && concurrent <= 1000) {
                config.max_concurrent_sessions = concurrent;
            } else {
                Logger::Error("无效的并发数: {}，应在1-1000之间", concurrent);
                return false;
            }
        }
        else if (arg == "--enable-optimization") {
            config.enable_cpu_optimization = true;
        }
        else if (arg == "--disable-optimization") {
            config.enable_cpu_optimization = false;
        }
        
        // 音频文件配置
        else if (arg == "--audio-dir" && i + 1 < argc) {
            std::string dir = argv[++i];
            if (std::filesystem::exists(dir)) {
                config.audio_files_dir = dir;
            } else {
                Logger::Error("音频目录不存在: {}", dir);
                return false;
            }
        }
        else if (arg == "--max-files" && i + 1 < argc) {
            int max_files = std::stoi(argv[++i]);
            if (max_files > 0) {
                config.max_test_files = max_files;
            } else {
                Logger::Error("无效的最大文件数: {}", max_files);
                return false;
            }
        }
        else if (arg == "--enable-resampling") {
            config.enable_audio_resampling = true;
        }
        else if (arg == "--disable-resampling") {
            config.enable_audio_resampling = false;
        }
        
        // 测试模式配置
        else if (arg == "--test-all") {
            config.enable_offline_test = true;
            config.enable_streaming_test = true;
            config.enable_two_pass_test = true;
            config.enable_concurrent_test = true;
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
        else if (arg == "--test-concurrent-only") {
            config.enable_offline_test = false;
            config.enable_streaming_test = false;
            config.enable_two_pass_test = false;
            config.enable_concurrent_test = true;
        }
        
        // 输出控制
        else if (arg == "--report-file" && i + 1 < argc) {
            report_file = argv[++i];
        }
        else if (arg == "--log-level" && i + 1 < argc) {
            std::string level = argv[++i];
            if (level == "DEBUG") Logger::SetLevel(Logger::DEBUG);
            else if (level == "INFO") Logger::SetLevel(Logger::INFO);
            else if (level == "WARN") Logger::SetLevel(Logger::WARN);
            else if (level == "ERROR") Logger::SetLevel(Logger::ERROR);
            else {
                Logger::Error("无效的日志级别: {}", level);
                return false;
            }
        }
        else if (arg == "--quiet") {
            Logger::SetLevel(Logger::ERROR);
        }
        else if (arg == "--verbose") {
            Logger::SetLevel(Logger::DEBUG);
        }
        
        // 未知参数
        else {
            Logger::Error("未知参数: {}", arg);
            Logger::Info("使用 --help 查看帮助信息");
            return false;
        }
    }
    
    return true;
}

/**
 * 验证配置合理性
 * @param config 引擎配置
 * @return 配置是否合理
 */
bool ValidateConfig(const FunASREngine::Config& config) {
    Logger::Info("========== 配置验证 ==========");
    
    // 检查音频目录
    if (!std::filesystem::exists(config.audio_files_dir)) {
        Logger::Error("音频目录不存在: {}", config.audio_files_dir);
        return false;
    }
    
    // 检查CPU线程数合理性
    unsigned int max_threads = std::thread::hardware_concurrency();
    if (config.cpu_threads > static_cast<int>(max_threads * 2)) {
        Logger::Warn("CPU线程数({})超过推荐值({}), 可能影响性能", 
                    config.cpu_threads, max_threads * 2);
    }
    
    // 检查并发数合理性
    if (config.max_concurrent_sessions > config.cpu_threads * 4) {
        Logger::Warn("并发会话数({})过高，建议不超过CPU线程数的4倍({})", 
                    config.max_concurrent_sessions, config.cpu_threads * 4);
    }
    
    // 检查至少启用一个测试
    if (!config.enable_offline_test && !config.enable_streaming_test && 
        !config.enable_two_pass_test && !config.enable_concurrent_test) {
        Logger::Error("至少需要启用一种测试模式");
        return false;
    }
    
    Logger::Info("配置验证通过");
    Logger::Info("==============================");
    return true;
}

/**
 * 显示最终配置信息
 */
void DisplayFinalConfig(const FunASREngine::Config& config, const std::string& report_file) {
    Logger::Info("========== 最终配置 ==========");
    
    std::ostringstream config_log;
    config_log << "设备模式: " << config.device;
    Logger::Info(config_log.str());
    
    config_log.str("");
    config_log << "CPU线程数: " << config.cpu_threads << " 核";
    Logger::Info(config_log.str());
    
    config_log.str("");
    config_log << "最大并发数: " << config.max_concurrent_sessions << " 路";
    Logger::Info(config_log.str());
    
    config_log.str("");
    config_log << "音频目录: " << config.audio_files_dir;
    Logger::Info(config_log.str());
    
    config_log.str("");
    config_log << "最大测试文件: " << config.max_test_files << " 个";
    Logger::Info(config_log.str());
    
    config_log.str("");
    config_log << "CPU优化: " << (config.enable_cpu_optimization ? "启用" : "禁用");
    Logger::Info(config_log.str());
    
    config_log.str("");
    config_log << "音频重采样: " << (config.enable_audio_resampling ? "启用" : "禁用");
    Logger::Info(config_log.str());
    
    config_log.str("");
    config_log << "报告文件: " << report_file;
    Logger::Info(config_log.str());
    
    // 显示测试计划
    Logger::Info("\n📋 测试计划:");
    if (config.enable_offline_test) Logger::Info("  ✅ 离线识别性能测试");
    if (config.enable_streaming_test) Logger::Info("  ✅ 流式识别性能测试");
    if (config.enable_two_pass_test) Logger::Info("  ✅ 2Pass模式性能测试");
    if (config.enable_concurrent_test) Logger::Info("  ✅ 并发性能测试");
    
    Logger::Info("==============================");
}

/**
 * 实时进度监控 - 增强版
 * 持续监控测试进度并显示详细统计信息
 */
void MonitorProgress() {
    Logger::Info("🧪 性能测试运行中，按Ctrl+C可安全退出...");
    int progress_count = 0;
    const int max_progress_cycles = 60; // 增加到10分钟
    
    // 等待测试真正开始
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    while (progress_count < max_progress_cycles && !g_shutdown_requested) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        progress_count++;
        
        if (g_engine && !g_shutdown_requested) {
            auto metrics = g_engine->GetPerformanceMetrics();
            
            // 只有当有实际请求时才输出详细进度
            if (metrics.total_requests > 0) {
                std::ostringstream progress_log;
                progress_log << "📈 进度报告 [" << (progress_count * 10) << "秒]:\n"
                            << " 总请求: " << metrics.total_requests << "次"
                            << " | 成功: " << metrics.success_requests << "次\n"
                            << " 离线RTF: " << std::fixed << std::setprecision(4) << metrics.offline_rtf
                            << " | 处理时长: " << std::setprecision(2) << metrics.total_audio_processed_hours << "小时";
                Logger::Info(progress_log.str());
            } else {
                Logger::Info("📈 等待测试开始... [{}秒]", progress_count * 10);
            }
        }
        
        // 检查测试是否完成
        if (g_engine && !g_engine->IsTestingActive()) {
            Logger::Info("✅ 测试已完成，准备生成报告...");
            break;
        }
    }
}


/**
 * 生成并保存性能报告
 */
bool GeneratePerformanceReport(const std::string& report_file) {
    if (!g_engine) {
        Logger::Error("引擎未初始化，无法生成报告");
        return false;
    }
    
    Logger::Info("📊 正在生成最终性能报告...");
    
    auto final_metrics = g_engine->GetPerformanceMetrics();
    
    // 显示到控制台
    std::cout << "\n" << final_metrics.ToString() << std::endl;
    
    // 保存到文件
    try {
        std::ofstream report_output(report_file);
        if (report_output.is_open()) {
            // 添加文件头信息
            report_output << "FunASR CPU版性能测试报告\n";
            report_output << "生成时间: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
            report_output << "测试平台: CPU多线程模式\n";
            report_output << "========================================\n\n";
            
            // 写入详细报告
            report_output << final_metrics.ToString();
            
            // 添加额外的分析信息
            report_output << "\n📋 性能分析:\n";
            if (final_metrics.streaming_rtf < 1.0) {
                report_output << "✅ 流式识别性能良好，可满足实时处理需求\n";
            } else {
                report_output << "⚠️  流式识别性能较慢，可能无法满足实时需求\n";
            }
            
            if (final_metrics.offline_rtf < 0.5) {
                report_output << "✅ 离线识别性能优秀\n";
            } else if (final_metrics.offline_rtf < 1.0) {
                report_output << "✅ 离线识别性能良好\n";
            } else {
                report_output << "⚠️  离线识别性能需要优化\n";
            }
            
            if (final_metrics.GetSuccessRate() >= 95.0) {
                report_output << "✅ 测试成功率优秀\n";
            } else if (final_metrics.GetSuccessRate() >= 85.0) {
                report_output << "✅ 测试成功率良好\n";
            } else {
                report_output << "⚠️  测试成功率偏低，需要检查配置\n";
            }
            
            report_output.close();
            
            std::ostringstream save_log;
            save_log << "📄 性能报告已保存到: " << report_file;
            Logger::Info(save_log.str());
            return true;
        } else {
            Logger::Error("无法创建报告文件: {}", report_file);
            return false;
        }
    } catch (const std::exception& e) {
        Logger::Error("保存报告时发生异常: {}", e.what());
        return false;
    }
}

/**
 * 主函数 - FunASR CPU版本程序入口
 */
int main(int argc, char* argv[]) {
    try {
        // 显示程序Banner
        PrintBanner();
        
        // 检测系统信息
        DetectSystemInfo();
        
        // 解析命令行参数
        FunASREngine::Config config;
        std::string report_file;
        
        if (!ParseCommandLine(argc, argv, config, report_file)) {
            return 0; // 正常退出（如显示帮助信息）
        }
        
        // 设置日志级别（如果未在命令行指定，使用默认INFO）
        Logger::SetLevel(Logger::INFO);
        
        // 设置信号处理
        signal(SIGINT, SignalHandler);
        signal(SIGTERM, SignalHandler);
        
        // 验证配置
        if (!ValidateConfig(config)) {
            Logger::Error("❌ 配置验证失败，程序退出");
            return -1;
        }
        
        // 显示最终配置
        DisplayFinalConfig(config, report_file);
        
        // 创建和初始化引擎
        Logger::Info("🚀 正在启动FunASR CPU引擎...");
        g_engine = std::make_unique<FunASREngine>(config);
        
        Logger::Info("📥 正在初始化模型和加载音频文件...");
        if (!g_engine->Initialize()) {
            Logger::Error("❌ 引擎初始化失败");
            return -1;
        }
        
        Logger::Info("✅ FunASR CPU引擎初始化成功！");
        
        // 启动性能测试
        Logger::Info("🧪 启动性能测试套件...");
        if (!g_engine->RunPerformanceTests()) {
            Logger::Error("❌ 性能测试启动失败");
            return -1;
        }
        
        // 监控测试进度
        MonitorProgress();
        
        // 生成性能报告
        if (!GeneratePerformanceReport(report_file)) {
            Logger::Warn("⚠️  性能报告生成失败，但测试已完成");
        }
        
        // 程序正常结束
        Logger::Info("🎉 FunASR CPU版本测试完成，感谢使用！");
        return 0;
        
    } catch (const std::exception& e) {
        Logger::Error("程序运行异常: {}", e.what());
        return -1;
    } catch (...) {
        Logger::Error("程序遇到未知异常");
        return -1;
    }
}
