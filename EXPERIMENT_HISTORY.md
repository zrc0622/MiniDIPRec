# Experiment history

## 2026-09-09 — MiniOneRec / Qwen3-0.6B / recent-50 / four GPUs

本次完成复现代码与本地验证，**未执行正式 GPU 训练或最终测试集评估**。没有论文指标复现结论，也不预设 RL 提升。

### 来源与范围

- fork 起点：`5f4f733`（MiniDIPRec origin: `https://github.com/zrc0622/MiniDIPRec.git`）。
- 对照本地官方仓库 `MiniOneRec`：`0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`；初始 `rl.py` / `data.py` 与 fork 相同。
- 基于官方真正启用的 `sft.py` / `sft.sh`、`rl.py` / `rl.sh`、`minionerec_trainer.py` 和数据导出实现；保持 GPR、TS-Rec 和未启用任务关闭。
- 原始 CSV、SID index、item metadata、划分及目标均未改写。原八卡入口保留供溯源；复现入口为 `scripts/reproduce.sh`。

### 已执行的数据核验

| 数据集 | train / valid / test 样本数 | 获得额外历史的 train / valid / test 样本数 | 用户数 | 完整轨迹最大交互数 |
|---|---|---|---|---|
| Office_Products | 38924 / 4866 / 4866 | 1956 / 359 / 443 | 8328 | 77 |
| Industrial_and_Scientific | 36259 / 4532 / 4533 | 1909 / 364 / 391 | 7694 | 88 |

按官方 train→valid→test 原始行序重放，全部用户从长度 1 的历史开始；每个后续窗口严格匹配已恢复序列的末尾最多 10 项。item/SID/title 不匹配数为 0。输出历史最大为 50；Office test 本身最长仅 30，不能把“上限 50”误报为每个 split 都有长度 50 的样本。

SID 碰撞保留：Office 3459 item / 3444 SID（多出 15 个共享 SID 的 item），Industrial 3686 / 3670（多出 16 个）。最终指标沿用官方 SID 命中定义，预测文件保留候选 SID 的全部 item ID，不做任意单 item 归属。

### 真实 Qwen3 tokenizer 检查

仅下载官方 `Qwen/Qwen3-0.6B` 的 config/tokenizer，没有下载 0.6B 模型权重。Hugging Face 直连被重置，使用 `hf-mirror.com` 的同名文件；tokenizer 文件副本在验证结果目录。

| 项目 | Office | Industrial |
|---|---|---|
| 新增 SID token | 600 | 560 |
| 扩展后 tokenizer 大小 | 152269 | 152229 |
| SFT SID-history 最大长度（含标签） | 330 | 330 |
| SFT item alignment 最大长度 | 145 | 141 |
| SFT Fusion 最大长度 | 390 | 384 |
| SFT 最终上限 | 512 | 384 |
| RL 全任务最长 prompt | 1737 | 1781 |
| RL prompt 上限 | 1792 | 1792 |
| SID completion（SID + newline + EOS） | 5 | 5 |

EOS/PAD 为 151645；所有 SID 原子 token、连接编码、换行及 EOS 都通过 round-trip。SFT 推荐 prompt 与 RL/评估的公共 prompt 逐 token 一致。使用真实 tokenizer、随机小 Qwen3 和完整 Office catalog 执行 beam=50，对不同长度的左 padding prompt 均生成 50 个唯一合法 SID；这验证了约束逻辑，不代表训练后准确率。

### 已执行的小规模验证

- 最终定向单测与 runtime 回归 **13/13 PASS**，没有跳过真实 tokenizer 测试。
- 标准库全量数据检查：原文件指纹、恢复窗口、目标/划分不变、错误窗口与错误 title 拒绝、结果文件篡改拒绝。
- 奖励检查：重复 prompt 不覆盖 row target，分组错位拒绝，零命中整组奖励为零，rank penalty 与官方公式一致。
- 指标检查：HR=Recall 的单目标口径、NDCG rank 折扣。
- 真实 Transformers `Trainer`：小随机 Qwen3 SFT backward 一步、标签 mask/EOS、超长拒绝。
- 真实官方 `ReReTrainer`：小随机 Qwen3 RL train/eval、按验证 reward 选 checkpoint、同步 reference 保存/加载、继续训练到下一 step。
- sampler 检查：实际 Accelerate `BatchSamplerShard`，四个 rank、micro=16/32/64、尾批次补齐，每组保持同一输入行；epoch 顺序可恢复。
- **真实四进程 CPU/Gloo 训练通过**：四 rank 各训练 2 step、梯度累积 2，G=16，non-reentrant gradient checkpointing；每 rank 检查 8 个生成组，reference 跨 rank 一致；checkpoint/reference 保存通过，训练中 KL 非零。
- 四卡 serial dry-run：指定 `2,3,6,7`，Office SFT→valid/test eval→RL→valid/test eval 完成后才进入 Industrial；两次 SFT 均指向原始 Qwen3，两次 RL 指向对应 SFT selected_model。
- Python 编译、shell 语法、git whitespace 检查通过；打包工具拒绝软链接和模型权重混入。

本地 runtime：macOS ARM64，Python 3.12，PyTorch 2.11.0 CPU，Transformers 4.57.1、TRL 0.24.0、Accelerate 1.10.1、Datasets 4.2.0。小模型 CPU 测试使用 AdamW，以验证 trainer 逻辑；**正式 RL 配置仍是官方 paged_adamw_32bit + ZeRO-2**。CPU 测试不是 Linux/PyTorch 2.6/CUDA 路径的验证。

### 改造与必要修复

完整清单见 README“历史恢复与必要修复”。主要差异：恢复最近 50 历史、Qwen3 词表与模板/终止约束统一、4 卡微批次和累积、奖励标签随行携带、重复 catalog 键保留样本、Fusion title 按 item ID 对齐、验证选模与保存间隔对齐、reference 与 sampler 恢复、本地完整产物归档。

### 尚未执行

- Qwen3-0.6B 两类别的完整 SFT/RL 训练及真实 HR/Recall/NDCG 对比。
- 四卡 CUDA/NCCL、DeepSpeed ZeRO-2、bitsandbytes paged optimizer、BF16 kernel 的实际运行。
- 4 GPU 上的显存峰值、吞吐、完整模型断点恢复与长时间稳定性。

远端应先执行 README 的 `gpu_smoke`，再运行独立正式 run。两者结果目录和 checkpoint 分离；所有最终指标由实际评估生成，不用 CPU 小模型结果填充。

### 验证产物

- `results/validation_20260909/`：生成数据与 audit、真实 tokenizer/长度扫描、CPU 单测与四进程日志、运行环境、检查总结；无正式指标。
- `results/verification_20260909/`：面向四卡 `2,3,6,7` 的完整 serial dry-run 命令与数据副本。
- 可用 `python scripts/package_results.py validation_20260909` 打包并独立复制。
