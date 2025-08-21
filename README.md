```markdown
funasr-gpu-prototype/
├── src/
│   ├── main.cpp                    # 主程序
│   ├── funasr_engine.h            # FunASR引擎
│   ├── funasr_engine.cpp
│   └── utils.h                     # 工具类
├── audio_files/                    # 音频文件目录
│   ├── test_001.wav
│   ├── test_002.wav
│   └── ... (500个WAV文件)
├── scripts/
│   ├── build.sh                   # 构建脚本
│   └── run.sh                     # 运行脚本
├── CMakeLists.txt                 # 构建文件
└── README.md                      # 使用说明
```
1. 离线识别工作流

WAV文件读取 → 音频预处理 → VAD分段(可选) → 离线ASR识别 → 标点符号恢复 → 最终结果
    ↓             ↓            ↓              ↓             ↓            ↓
16bit PCM → float32归一化 → 语音段提取 → paraformer-zh → ct-punc → 完整文本


2. 实时流式工作流

音频流输入 → 600ms分块 → 流式ASR → 实时输出 → 缓存更新 → 下一块处理
    ↓           ↓          ↓        ↓        ↓         ↓
连续音频 → chunk分割 → streaming → 即时文本 → 状态保持 → 持续循环
3. 2Pass混合工作流
                     音频块输入
                         ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
      VAD语音检测                   流式ASR识别
            ↓                           ↓
      检测语音边界                   实时文本输出
            ↓                           ↓
      语音结束信号 ────→ 触发离线精化识别 → 高精度文本输出


4. 性能测试工作流
扫描WAV文件 → 随机选择 → 并行测试 → 指标收集 → 统计分析 → 生成报告
     ↓           ↓         ↓         ↓         ↓         ↓
   500个文件 → 100个文件 → 4种模式 → RTF等指标 → 平均值等 → 详细报告
