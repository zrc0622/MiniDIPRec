# MiniDIPRec：MiniOneRec 四卡复现

本分支提供 `Qwen/Qwen3-0.6B`、最近最多 50 次交互、单机指定四卡的 **Office SFT → 评估 → RL → 评估 → Industrial 同一流程**。两个类别分别从原始 Qwen3 开始；RL 仅继承本类别通过验证集选择的 SFT 模型。方法以官方实际启用的 `sft.py` / `rl.py` 为准，不包含 GPR、TS-Rec 或新增推荐方法。

当前已完成全量数据检查、真实 Qwen3 tokenizer 全任务长度扫描、小模型 SFT/RL/恢复、四进程 CPU 分布式检查。2026-09-10 远端 Office SFT 已完成；最新 micro64 RL 在完成 3 次更新后显存不足。另确认 Qwen3 默认生成参数覆盖了 RL 温度和旧评估的确定性设置，现已修复。**SFT 权重可继续用，旧 SFT 指标须重新评估后才能与修复后的 RL 比较。** 本机无 CUDA GPU，未执行本次修复后的四卡训练/评估；Industrial 尚无结果。详细记录见 [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)。

## 安装与一键运行

Linux、Python 3.11、4 张支持 BF16 的 NVIDIA GPU；建议从官方使用的 A100/H100 级设备开始。以下安装 CUDA 12.4 的 PyTorch 2.6 wheel，需匹配宿主驱动。命令均在仓库根目录执行。

首次安装环境（已有环境和 CUDA 12.4 Toolkit 时无需重复安装）：

```bash
conda create -n minidiprec python=3.11 -y
conda activate minidiprec
# PyTorch wheel 不包含 nvcc；将匹配的 Toolkit 安装到当前 conda 环境
conda install -c nvidia/label/cuda-12.4.1 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-reproduction.txt
```

**每次新开终端，或切换到其他终端运行训练，都先执行以下设置和检查。** `export` 只影响当前终端，没有写入持久化配置，也不会同步到其他已打开的终端。以下假设 Toolkit 安装在该 conda 环境；若使用系统已有的 CUDA 12.4 Toolkit，将 `CUDA_HOME` 改为其实际目录。

```bash
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

"$CUDA_HOME/bin/nvcc" -V
CUDA_VISIBLE_DEVICES=0,1,2,3 python -c 'import torch; import deepspeed; assert torch.cuda.is_available() and torch.cuda.device_count() == 4, "需要四张可见 CUDA GPU"; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "DeepSpeed:", deepspeed.__version__)'
```

`nvcc` 应显示 `release 12.4`，Python 检查应无 traceback。若使用其他四卡编号，同步修改检查命令中的 `CUDA_VISIBLE_DEVICES` 和运行命令中的 `--gpus`。检查失败先按 [debug.md](debug.md) 排查；检查通过后，在**同一个终端**首次运行：

```bash
bash scripts/reproduce.sh --run-name qwen3_h50_seed42 \
  --gpus 0,1,2,3 --model Qwen/Qwen3-0.6B \
  --checkpoint-root /path/to/large_disk/minionerec_checkpoints
```

`--model /path/to/Qwen3-0.6B` 支持含完整原始模型和 tokenizer 的本地目录。下载源配置沿用 Hugging Face 环境变量；脚本不强制第三方镜像。`--gpus 4,5,6,7` 或四个 GPU UUID 均可；所有训练、评估子进程只看到这四卡，内部 local rank 为 0–3，不额外占用 vLLM GPU。

执行顺序严格为 Office 全流程后 Industrial 全流程，所有阶段串行。SFT 使用 Transformers `Trainer` + DDP；RL 使用仓库原有 `ReReTrainer` + 官方 ZeRO-2 配置，保留 `paged_adamw_32bit`，没有换为其他 GRPO 实现。

| 参数 | 官方运行配置 | 本复现 |
|---|---|---|
| SFT batch | 8 × 16 × 8 = 1024 | 4 × 4 × 64 = 1024 |
| SFT 训练 | 10 epochs，LR 3e-4，AdamW，linear，warmup 20 | 保留；每 5% 验证/保存，patience 3 |
| RL batch（候选数） | 8 × 64 × 2 = 1024 | 4 × 16 × 16 = 1024 |
| RL 每次更新的 prompt 数 | 1024 / G16 = 64 | 64 |
| RL 训练 | 2 epochs，LR 1e-5，cosine，warmup 3%，grad norm 0.3 | 保留 |
| RL 奖励/生成 | exact + ranking，G=16，beam search，temperature 1 | 保留官方 beam sampling 配置 |
| RL reference | beta .001，sync=True，alpha .6，每 512 step 同步 | 保留并保存同步后的 reference 以支持恢复 |

SFT 全模型训练，保留 SID 历史→SID、SID↔title、SID 历史→title 任务；Fusion 的 description 分支原本未启用，保持关闭。RL 保留 SID 历史→SID、title/description→SID、随机抽取 10,000 条 title 历史→SID；抽样沿用官方 pandas `random_state=0`，整体 shuffle/训练 seed=42。保留 `add_gt=False`、`dynamic_sampling=False`、`test_during_training=False`、`dapo=False`、`gspo=False`。

可用 `--sft-micro-batch 8 --rl-micro-batch 32` 适配显存，累积自动计算，仍保持 batch 1024；RL micro 必须为 16 的倍数，低于 16 会明确拒绝。每张卡的候选生成必须持有完整的 G=16 组。默认评估 batch 为 2，可通过 `--eval-batch-size` 调整。各 epoch 的不足整批尾部沿用官方 Trainer/Accelerate 行为，可能有补齐和不足完整累积更新；批大小等价指正常完整更新。

### 显存允许时提高 RL 微批次

只需设置 `--rl-micro-batch`，梯度累积自动为 `256 / micro`：16→累积16，32→累积8，64→累积4；均为四卡、G16、每次完整更新 1024 个候选。当前先建议 32。增大微批次可能减少循环开销，但 prompt padding 增多也可能抵消收益，未实测 GPU 加速倍数。

最新 micro64 日志确认在生成候选时 OOM：GPU 2 总显存 44.39 GiB，同卡另一进程占用 9.14 GiB，当前训练占用 33.28 GiB，仅剩 58.12 MiB 时申请 1.54 GiB 失败。前几个短输入能运行不代表长输入也能运行。先回到 micro32；若仍 OOM，再用 micro16。检查方法见 [debug.md](debug.md)。

**从已有 SFT 重新开始 micro32 RL**：先停止自己占用同一组 GPU 的旧训练，将 `scripts/start_rl_from_sft.py`、`scripts/migrate_rl_config_fix.py`、`minionerec_trainer.py` 和 `reproduction/evaluate.py` 的最新版本同步到服务器仓库。在仓库根目录执行：

```bash
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

python scripts/start_rl_from_sft.py \
  --source-run qwen3_h50_seed42 \
  --run-name qwen3_h50_seed42_rl32_fixed \
  --dataset Office_Products --rl-micro-batch 32
```

此命令创建新 run，继承源 run 的 GPU、原始模型名、评估设置及 checkpoint 根目录；复制已完成的 Office SFT 产物、50 条历史数据及长度扫描结果，并将验证集选中的 SFT `selected_model` 实体复制到新 checkpoint 目录。SFT 训练不重跑，旧 RL 的权重/优化器/步数不继承，新 RL 从 step 0 开始。如果源 run 使用旧版评估器，脚本自动把旧评估移入新 run 的 `sft_import/superseded_sft_evaluation/`，先重评 SFT valid/test，再启动 RL，最后评估并汇总。旧 run 不修改。

若原始 `qwen3_h50_seed42` 已删除，可将 `--source-run` 换成仍保留完整 SFT 产物和模型副本的 `qwen3_h50_seed42_rl32` 或实际的 rl64 run 名称。脚本校验该副本的模型哈希和选中 step，无需最早的 checkpoint 仍存在。`sft_import/record.json` 记录来源及复制文件哈希；大模型权重仍在结果目录外。该入口只运行所选类别；后续可用常规入口在新 run 中执行 Industrial 全流程。

`--prepare-only` 可只导入而不启动训练。新 run 已创建后需要继续运行时，保持 CUDA 环境设置并执行 `bash results/qwen3_h50_seed42_rl32_fixed/resume.sh`；无需重复导入。首次运行从 SFT 开始，之后中断才恢复新 run 自己的 RL checkpoint。打包命令为 `python scripts/package_results.py qwen3_h50_seed42_rl32_fixed`。

**从旧 RL checkpoint 接着训练**时，才使用下面的 batch 迁移流程：

此流程要求源码已一致，或只有旧 dtype 字典复制修复的差异。**它不迁移本次生成参数修复；旧温度 0.6 的 RL 应使用上面的 SFT 导入命令重新开始。** 不要删除源码哈希文件绕过检查。

**已有 run 不能只改启动参数。** Transformers 恢复时会从 `trainer_state.json` 取旧微批次，可能覆盖新配置。先将 `scripts/migrate_rl_batch.py`、`scripts/migrate_rl_config_fix.py` 和已修复的 `minionerec_trainer.py` 同步到服务器，等待完整 checkpoint 保存后停止当前训练；以下操作在服务器仓库根目录执行：

```bash
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

# 可选预览：增加 --dry-run；正式迁移不要带该参数
python scripts/migrate_rl_batch.py --run-name qwen3_h50_seed42 --rl-micro-batch 32

# 迁移成功后继续；其他模型/路径/batch 参数沿用原实验
bash scripts/reproduce.sh --run-name qwen3_h50_seed42 \
  --gpus 0,1,2,3 --model Qwen/Qwen3-0.6B \
  --rl-micro-batch 32 --resume
```

迁移会在结果目录 `config_migrations/` 备份并记录旧配置和 checkpoint JSON，再更新运行配置与未完成 RL 阶段的 checkpoint batch 字段。已有 SFT、数据、模型权重、优化器、scheduler、reference、RNG 状态及 global_step 不变；需要时会先执行上一项 dtype 修复的源码迁移。没有 checkpoint 时该 RL 阶段从头开始；已有 checkpoint 则恢复最近完整 checkpoint，未保存的进度需重算。此适配改变并行分组、padding 和随机生成调用，不保证逐位一致；epoch 尾部补齐也可能不同。RL 训练期间的每卡验证微批次跟随该参数变为 32，最终 beam=50 评估设置不变。迁移后每次续跑都带 `--rl-micro-batch 32`。

## 单阶段与断点恢复

所有阶段均通过相同入口，沿用同一 run-name、模型、四卡和 batch 参数：

```bash
# 无 GPU 也可以执行：恢复并检查两份数据，不覆盖官方文件
bash scripts/reproduce.sh --run-name qwen3_h50_seed42 --stage prepare

# 无 GPU 也可以执行：真实 tokenizer 扫描所有启用任务
bash scripts/reproduce.sh --run-name qwen3_h50_seed42 --stage preflight

# 以下 GPU 阶段先完成上面的环境检查；新终端需重新激活并设置 CUDA 路径
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

# 仅 Office SFT；stage 还支持 eval-sft、rl、eval-rl、summary
bash scripts/reproduce.sh --run-name qwen3_h50_seed42 \
  --gpus 0,1,2,3 --dataset Office_Products --stage sft

# 自动跳过已完成阶段；未完成的训练恢复最新完整 checkpoint
bash scripts/reproduce.sh --run-name qwen3_h50_seed42 --gpus 0,1,2,3 --resume

# 只生成完整顺序和命令，执行数据检查，不加载模型或启动训练
bash scripts/reproduce.sh --run-name inspect_commands --gpus 0,1,2,3 --dry-run

# 四卡 GPU 冒烟：独立目录，每个训练阶段只跑 2 次更新
bash scripts/reproduce.sh --run-name gpu_smoke --gpus 0,1,2,3 --max-steps 2
```

上述单阶段示例使用默认 checkpoint-root；若一键命令使用了自定义 `--checkpoint-root` 或本地 `--model`，**每次继续运行时也传相同参数**。配置、数据指纹或实现变化会拒绝复用旧 run；正式实验不能沿用 `--max-steps` 冒烟 run。恢复包含模型、优化器、scheduler、随机状态、SFT early-stop 状态和 RL reference；RL 缺失 reference 的旧官方 checkpoint 会被拒绝作为续训状态。`selected_model` 是选出的模型导出，只供评估和下一阶段初始化；继续训练使用 `checkpoint-<step>`。

同一 run-name 下，已生成的 50 条历史 CSV 会在哈希校验通过后复用；已有且匹配的 `lengths.json` 也会复用。训练提示构造和 tokenization 目前没有磁盘缓存，各训练进程重启后仍会执行。数据复用不依赖 `--resume`；继续已有实验应加 `--resume`，以跳过已完成阶段并恢复训练 checkpoint。

### 已有实验遇到 RL 参数 JSON 错误

若 `Office_Products/rl/train.log` 报 `TypeError: Object of type dtype is not JSON serializable`，这是首次 RL 更新前记录参数时的错误。trainer 原来原地修改 `model_init_kwargs`，将字符串改成 `torch.dtype`；现改为复制字典后供模型加载使用，模型精度、训练参数、奖励和任务不变。训练结束后的参数记录也因此得到修复。

当前代码还包含生成参数修复，应使用上面的 `start_rl_from_sft.py` 新 run 流程保留 SFT、重评并重新开始 RL。历史工具 `migrate_rl_config_fix.py` 仅接受 dtype 这一行差异，无法迁移本次生成参数修复，拒绝其他源码差异属于预期行为。

## 历史恢复与必要修复

官方处理程序按目标时间排序交互并切成 train/valid/test，每条 history 最多 10 条。新程序严格按三个原始文件的原始行序逐行重放：首次出现的用户必须有 1 条历史；每个后继 history 必须等于已恢复过去交互的末尾 `min(10, 已知长度)` 条。检查通过后，先输出目标之前最近最多 50 条，随后才把当前目标加入已知序列。断链、缺开头、乱序、缺 metadata、item/SID/title 不匹配立即报错，不尝试猜测或按 item ID 排序。保留行数、行序、目标、划分、SID；新增稳定 `sample_id`、`target_position` 用于审计。生成文件单独写到结果目录。

这依赖官方 CSV 的时间排序导出契约；CSV 本身没有 timestamp，无法独立验证原始时间戳。在该契约下，较早的 valid/test 交互可成为更晚样本的已知历史，和官方连续窗口一致；当前目标和更晚交互不会进入自身历史，也不从 test 标签选模型。

必要错误修复单独列出：

1. RL 标签随样本行携带，奖励直接读取 `target`；重复 prompt/history 不再覆盖标签，不合并验证标签到训练字典。rank 内生成前检查 prompt、target、sample_id 的 G=16 分组，跨 rank reward gather 后保持相同切片顺序。
2. 官方 item/title/description 字典会丢弃重复键。本流程按 item 生成对齐样本，保留相同文本对应不同 SID 的样本；SID 碰撞时 Fusion 的 title 根据原 item ID 获取。任务种类和奖励公式不变，修复会使对齐任务样本数略有增加。
3. 官方 SFT 带 instruction 前缀，RL 缺少前缀，旧 eval 的推荐问法也不同。本流程以官方 SFT 模板统一推荐输入，item identification 沿用其官方模板；保留纯文本 `### User Input` / `### Response`，不套 Qwen3 chat/thinking 模板。
4. 新增 SID token 按官方排序扩词表，同步 resize input embedding 与 output head。检查每个 SID token 的原子性、连接编码和完整 round-trip。统一输出为 `SID + 换行 + tokenizer EOS`；Qwen3 EOS/PAD 为 151645。约束树按生成部分计算 prefix，不硬编码模型名称、提示 token 数或 token ID。非法 prefix、缺 EOS、超长输入立即报错。
5. 官方 RL 入口额外加载 `device_map=auto` 的模型；新入口只由 trainer 加载本 rank 的策略模型与 reference，使用 BF16。TRL generation context 恢复 gradient checkpointing 时会丢失 non-reentrant 设置，已在 backward 前恢复，四进程小模型验证通过。
6. 为支持用户要求的验证选模，SFT 保留最低 validation loss；RL 以最高 validation ranking reward（exact + rank penalty 之和）选模。RL eval 间隔保留 .0999，save 从 .1 对齐到 .0999；修复 trainer 只把 eval_reward 写日志而不返回的问题，供最佳 checkpoint 选择使用。test 仅在已选择模型上最终评价。
7. RL checkpoint 增加同步 reference 权重，修复 sampler 在恢复后无法复现对应 epoch 顺序的问题。模型保存仅由主 rank 写，训练日志和状态持续归档。W&B 改为本地记录。
8. Transformers 4.57.1 会用模型默认值覆盖 `GenerationConfig` 中等于全局默认值的字段。Qwen3 因此把 RL temperature=1 改成 0.6，也把旧评估 do_sample=False 改成 True。生成调用显式传 `use_model_defaults=False`，保留官方 RL beam sampling/temperature=1，统一最终评估为 deterministic beam50。此修复不改变 SFT 训练及基于验证 loss 的选模；旧 SFT 需要重评，旧 RL 需要从 SFT 重启。

SID 映射本身存在碰撞：Office 3459 个 item / 3444 个 SID，Industrial 3686 / 3670。保留官方 SID 命中口径，预测同时写每个 SID 对应的完整 item ID 列表；不宣称模型能区分共享 SID 的 item。

## 统一评估与产物

SFT 和 SFT+RL 都用 full-catalog、deterministic beam=50、temperature=1、length_penalty=0，不剔除历史 item，不抽负例；输出 50 个唯一合法 SID。报告 HR/Recall@5、@10 和 NDCG@5、@10；每行一个目标，因此 HR=Recall。四个评估 rank 用原始行号步进切分，合并时检查每个样本恰好出现一次。

token 上限来自全量任务扫描并向上取整到 128；本次 Office/Industrial 的 SFT 上限分别 512/384，RL prompt 都为 1792，SID completion 为 5。不会通过截断悄悄丢掉更早历史。运行仍会针对实际提供的 tokenizer 重新扫描，并记录配置。

```text
results/<run_name>/
  run_config.json, invocations.jsonl, commands.sh, commands.jsonl, environment.log
  source/, source_sha256.json
  summary.csv, summary.json, summary.md
  Office_Products/                        # Industrial 同结构
    data/{train,valid,test}.csv, *.index.json, *.item.json, info.txt, audit.json
    lengths.json, tokenizer/, preflight.log
    sft/                                 # rl 同结构
      train.log, metrics.jsonl, training_args.json, trainer_state.json, training.json
      valid.predictions.jsonl, test.predictions.jsonl
      valid.metrics.json, test.metrics.json, eval-valid.log, eval-test.log
      *.complete.json
<checkpoint-root>/<run_name>/<category>/{sft,rl}/
  checkpoint-<step>/, selected_model/
```

`training.json` 记录父模型、选中 checkpoint 路径/step、训练终止 step 和验证选择依据。结果目录保存实际数据与源码副本，不使用外部软链接；复制结果目录即可分析。模型权重在 checkpoint-root 单独存储，只有重新推理或续训才需要拷贝它们。没有训练结果时 summary 明确为空，不生成虚构指标。

一条打包命令（校验没有软链接或混入模型权重）：

```bash
python scripts/package_results.py qwen3_h50_seed42
```

输出 `results/qwen3_h50_seed42.tar.gz`。

## 本地检查

```bash
python -m unittest tests.test_reproduction tests.test_reproduction_runtime tests.test_rl_config_migration tests.test_rl_batch_migration tests.test_start_rl_from_sft -v
# 可选：使用真实 Qwen3 tokenizer 跑 beam=50 小模型解码检查
QWEN3_TOKENIZER=/path/to/Qwen3-0.6B python -m unittest tests.test_reproduction_runtime -v
# CPU 四进程检查（使用小随机 Qwen3；不是正式 GPU 性能验证）
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 \
  --master_port=29719 --module tests.reproduction_distributed_smoke
```

以下保留原仓库 README，原 `sft.sh` / `rl.sh` / `evaluate.sh` 是官方八卡示例；本复现请使用上面的 `scripts/reproduce.sh`。

---

<div align="center">


<img src="./assets/logo.png" width="500em" ></img> 

**An Open-Source Framework for
Scaling Generative Recommendation**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 Technical Report</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>
</div>

**MiniOneRec** is the first fully open-source **generative recommendation** framework, which provides an end-to-end workflow spanning **SID construction**, **supervised fine-tuning (SFT)**, and recommendation-oriented **reinforcement learning (RL)**. 

---

## 📢 Announcement

- 2026-05-13 — We have introduced the new TS-Rec codebase, following the method proposed in [Fine-grained Semantics Integration for Large Language Model-based Recommendation](https://arxiv.org/pdf/2602.22632). We sincerely thank the contributors for their valuable efforts and support in making this update available.

- 2026-01-04 — Regarding the potential discrepancies between the reproduced results based on the Instruct model and our reported metrics, please check whether the CC metric in the evaluation log is non-zero (refer to calc.py). If it is non-zero, it indicates that the model is still generating a large number of invalid items, and constrained decoding has not been successful. We suspect this issue may be related to the versions of dependencies such as the transformer library, and we are still investigating the cause to provide a universal solution. In the meantime, you may switch the Instruct model to a base model, such as Qwen2.5-base, to avoid this problem.

- 2025-12-04 — We update new scripts to support processing the Amazon23 dataset.

- 2025-12-01 — We fix a bug in data.py that could cause the SID–item alignment task to see the answers in advance. This was because we had previously attempted to use partial trajectories to guide the full SID–item generation and does not affect the model performance.

- 2025-11-20 — The SID construction method in **RQ-Kmeans+** has been updated (first proposed in **GPR** and this is the first open-source reproduction).

- 2025-11-19 — We implemented a multi-GPU parallel text-to-embedding method based on Accelerate, which is significantly more efficient than the original version: rq/text2emb/amazon_text2emb.py

- 2025-11-19 — The SID construction method in **constrained-RQ-Kmeans** has been updated.

- 2025-11-07 — Thank you for submitting issues! Based on your feedback, we have released a new implementation. If you encounter any problems while running the code, please update to and consult the **latest version** first.
  
- 2025-11-07 — You can now choose to freeze the LLM parameters during the SFT stage and train only the embeddings for the newly added SID vocabulary.

- 2025-10-31 — You can now directly download the implementation **checkpoints** of our MiniOnRec model.

- 2025-10-31 — The SID construction method in **RQ-Kmeans** has been updated.

---

## 🛠️ Key Techniques 
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID Construction: MiniOneRec begins by transforming every product into a compact, semantically meaningful token.** It concatenates an item’s title and description, feeds this sentence through a frozen text encoder, and then quantises the resulting embedding with a three-level RQ-VAE.

- **SFT: With all items rewritten as SIDs, the model is first trained in a supervised fashion.** It views the chronologically ordered user history as a token sequence and learns, via next-token prediction, to generate the SID of the next product the user is likely to consume. Crucially, this stage is co-trained with a set of language-alignment objectives that map back and forth between natural language and SID space, allowing the recommender to inherit the world knowledge embedded in large language models while grounding that knowledge in discrete item codes.

- **Recommendation-Oriented RL: After SFT, MiniOneRec is further polished with a recommendation-oriented RL phase based on GRPO.** Multiple candidate recommendations are generated for each prompt, their rewards are normalised within the group to stabilise gradients, and a KL penalty keeps the updated policy close to its reference. Because the action space is a closed list of item SIDs, the system switches to constrained beam search, which guarantees that every beam is unique and valid, greatly improving sampling efficiency and diversity. The reward signal itself blends a binary correctness term with a rank-aware component that penalises high-probability yet incorrect items more heavily, and can be augmented with collaborative-filtering scores. Together, this pipeline enables MiniOneRec to couple dense linguistic knowledge, achieving a high-performance, lightweight generative recommendation system.

---

## 📊 Evaluation

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img> 
</div>

---

## 🗂️ Repository Overview

| File / Directory          | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `sft.sh`                  | Shell script to start the Supervised Fine-Tuning (SFT) stage                                           |
| `sft.py`                  | Python implementation of the SFT training loop                                                            |
| `sft_gpr.py`              | GPR-inspired SFT with Value-Aware Fine-Tuning (VAFT): implements weighted loss based on simulated item value                            |
| `rl.sh`                   | Shell script to start the Reinforcement Learning (RL) stage                             |
| `rl.py`                   | Python implementation of the RL training loop                                              |
| `rl_gpr.py`               | GPR-inspired RL with Hierarchy Enhanced Policy Optimization (HEPO)                                                 |
| `minionerec_trainer.py`   | MiniOneRec trainer — GRPO-based trainer specialized for generative recommendation                              |
| `configs/`                | YAML configuration files                                            |
| `evaluate.sh`     | One-click offline Top-K evaluation script                                                        |
| `evaluate.py`     | Evaluation utilities for computing HR@K and NDCG@K.                                                           |
| `LogitProcessor.py`                | Logit processor for constrained decoding (Python implementation)                                         |
| `data.py`                | Data pipeline for SFT and RL training                          |
| `convert_dataset.py`                | Converts an RQ-trained dataset to the SFT-then-RL format                                            |
| `convert_dataset_gpr.py`           | GPR-inspired dataset converter: injects simulated heterogeneous tokens (U/E/I/O) to emulate unified input representation                                         |
| `data/amazon18_data_process.sh`                |    Shell script to filter and preprocess Amazon18 data into an RQ-ready format                                      |
| `data/amazon18_data_process.py`                |   Python implementation of the Amazon18 data preprocessing pipeline                                        |
| `data/amazon18_data_process_gpr.py`            |   GPR-inspired Amazon18 preprocessing: extracts heterogeneous features for unified input representation                         |
| `data/amazon23_data_process.sh`                |    Shell script to filter and preprocess Amazon23 data into an RQ-ready format                                      |
| `data/amazon23_data_process.py`                |   Python implementation of the Amazon23 data preprocessing pipeline                                        |
| `rq/text2emb/amazon_text2emb.sh`                |   Shell script to generate item embeddings (title + description) via emb_model for the Amazon dataset                                   |
| `rq/text2emb/amazon_text2emb.py`                |   Python implementation of the above embedding generation                                         |
| `rq/text2emb/amazon_text2emb_gpr.py`           |   GPR-inspired text-to-embedding                                 |
| `rq/generate_indices.py`                |   Generates the SID file after training an RQ-VAE model                                       |
| `rq/rqvae.sh`                |   Shell script to train RQ-VAE on Amazon item embeddings                        |
| `rq/rqvae.py`                |   Python implementation of RQ-VAE training                                            |
| `rq/rqkmeans_faiss.py`                |   Python implementation of RQ-Kmeans training based on faiss                                          |
| `rq/rqkmeans_constrained.py`                |   Python implementation of Constrained RQ-Kmeans                         |
| `rq/rqkmeans_constrained.sh`                |   Shell script to train constrained RQ-Kmeans constrained on Amazon item embeddings                        |
| `rq/rqkmeans_plus.py`                |   Python implementation of RQ-Kmeans+                        |
| `rq/rqkmeans_plus.sh`                |   Shell script to train RQ-Kmeans+ constrained on Amazon item embeddings                        |
| `rq/generate_indices_plus.py`                |   Generates the SID file after training an RQ-Kmeans+ model                                       |
| `rq/generate_indices_plus.sh`                |   Shell script to generate the SID file after training an RQ-Kmeans+ model                                       |
| `requirements.txt`        | List of Python dependencies                                                                                |

---

## 🚀 Quickstart

Use the pre-trained Industrial/Office SIDs we provide for a quick start!
Reproduction can be achieved with just 4–8 A100/H100 GPUs.

### 1. Create an isolated Python environment

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. SFT

```bash
bash sft.sh
```

### 4. Recommendation-Oriented RL

```bash
bash rl.sh
```

### 5. Run the evaluation bash

```bash
bash evaluate.sh
```

---

## 📜 Full Pipeline Walk-through

### 0. Prerequisites
- GPUs: <e.g., 4–8 × A100/H100 80 GB or comparable>
- Python: 3.11

### 1. Environment Setup
- **1.1 Clone the repo**
```
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```
- **1.2 Create and activate a conda env**
```
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```
- **1.3 Install dependencies**
```
pip install -r requirements.txt
```

### 2. Data Preparation

- **2.1 Download the raw dataset (Optional)**  
  Get it from the official page:
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), 
  [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), 
  [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
  Note: The Industrial and Office datasets are included in Amazon 2018; the Amazon 2014 and 2023 versions require slight modifications to our data/amazon18_data_process.py.
- **2.2 Filter and preprocess**
```
bash data/amazon18_data_process.sh \
     --dataset  your_dataset_type \ # e.g. Industrial
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```
- **2.3 Encode item text to embeddings**
```
bash rq/amazon_text2emb.sh \
     --dataset your_dataset_type \ # e.g., Industrial 
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID Construction

Choose either 3.1.1, 3.1.2, 3.1.3 or 3.1.4.

- **3.1.1 Train RQ-VAE on the embeddings**
```
bash rq/rqvae.sh \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

- **3.1.2 Train RQ-Kmeans on the embeddings**

```
conda install faiss-gpu
python rqkmeans_faiss.py --dataset Industrial_and_Scientific # The RQ-Kmeans method based on semantic embeddings has a relatively high collision rate.
```

- **3.1.3 Train constrained RQ-Kmeans on the embeddings**
For conflicting items, we add an extra layer to perform deduplication; meanwhile, we use a balanced constraint to ensure that the SIDs are evenly distributed.
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
```

- **3.1.4 Train RQ-Kmeans+ on the embeddings**
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
bash rqkmeans_plus.sh
```

- **3.2 Generate indices(only RQ-VAE & RQ-Kmeans+ needed)**
```
python rq/generate_indices.py
# or
bash rq/generate_indices_plus.sh
```

- **3.3 Convert dataset format**
```
python convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/ourput_dir \

```

### 4. SFT

```
bash sft.sh \
     --base_model your_model_path \
     --output_dir your_ourput_dir \
     --sid_index_path your_.index.json_path \
     --item_meta_path your_.item.json_path
```

### 5. Recommendation-Oriented RL
> (Optional) For production-scale datasets, considering the cost of reinforcement learning and diminishing marginal returns, you can perform the RL stage using only a relatively small subset on the order of tens of thousands of samples.
```
bash rl.sh \
     --model_path your_model_path \
     --output_dir output_dir \
```

### 6. Offline Evaluation

```
bash evaluate.sh \
     --exp_name your_model_path 
```

---

## 🤖 Supported LLM Providers

MiniOneRec supports multiple LLM providers for text enrichment tasks (e.g., user preference and item characteristic extraction). Configure the provider in your `api_info` dictionary:

| Provider | `provider` value | Default Base URL | Example Models |
|----------|-----------------|------------------|----------------|
| OpenAI | `"openai"` | — | `text-davinci-003` |
| DeepSeek | `"deepseek"` | `https://api.deepseek.com` | `deepseek-chat` |
| [MiniMax](https://www.minimaxi.com) | `"minimax"` | `https://api.minimax.io/v1` | `MiniMax-M2.7`, `MiniMax-M2.5` |

**Example — using MiniMax:**

```python
api_info = {
    "provider": "minimax",
    "api_key_list": ["your-minimax-api-key"],
    "base_url": "https://api.minimax.io/v1",  # optional, this is the default
}
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=api_info)
```

---

## 📝 Upcoming Features

We are actively extending MiniOneRec’s capabilities. The following enhancements are already on our roadmap:
* ⏱️ **More SID Construction Algorithms**: forthcoming support for R-VQ, RQ-Kmeans, RQ-OPQ, and RQ-VAE-v2 (PLUM).
* ⚙️ **MiniOneRec-Think**: a module that seamlessly integrates dialogue, reasoning, and personalized recommendation, providing an all-in-one solution for complex interactive scenarios.
* 🔍 **Broader Dataset Support**: additional popular public datasets, including Yelp, to further validate the generality of our algorithms.

---

## 🏫 Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)
 
---

## 🧩 Contributing

We welcome and appreciate all contributions! If you have ideas to improve MiniOneRec, please feel free to submit a pull request (PR).

---
## 🙏 Acknowledgements

This repository reuses or adapts portions of code from the following open-source projects. We gratefully acknowledge their authors and contributors:

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 🔖 Citation <!-- omit in toc -->

If you find our code/paper/model helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}

```

---

<div align="center">
We welcome contributions from the community! 🤝
</div>
