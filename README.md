# MiniDIPRec：MiniOneRec 四卡复现

本分支提供 `Qwen/Qwen3-0.6B`、最近最多 50 次交互、单机指定四卡的 **Office SFT → 评估 → RL → 评估 → Industrial 同一流程**。两个类别分别从原始 Qwen3 开始；RL 仅继承本类别通过验证集选择的 SFT 模型。方法以官方实际启用的 `sft.py` / `rl.py` 为准，不包含 GPR、TS-Rec 或新增推荐方法。

当前已完成全量数据检查、真实 Qwen3 tokenizer 全任务长度扫描、小模型 SFT/RL/恢复、四进程 CPU 分布式检查。2026-09-11 收到远端 Office micro16 fixed 完整结果：RL完成1746步，按验证奖励选中1400步；在相同确定性beam50评估下，测试HR@10由0.165023降至0.150226、NDCG@10由0.124228降至0.115154，验证集也下降。已核验源码/数据哈希和19464条预测，训练前期存在KL估计值与梯度尖峰，尚不能确定单一原因。此前Qwen3生成参数覆盖已修复，本次SFT已重评。**本机无CUDA，未独立执行GPU推理或checkpoint权重复核；Industrial尚无结果，不宣称RL必然提升。** 详细记录见 [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)。完整分析在本地 `results/qwen3_h50_seed42_rl16_fixed/analysis_20260911/report.md`。

补充 checkpoint 验证：收到服务器175/350/1575步评估后，五组共24330条验证预测核验通过。175步Recall@10已从SFT的0.232635降至0.211673，1400步仅恢复到0.213933；1400仍是本次四个RL checkpoint中NDCG@10最高者。现有证据更支持早期退化后未充分恢复，未证明具体原因。详细报告在 `results/qwen3_h50_seed42_rl16_fixed/Office_Products/diagnostics/checkpoint_analysis_20260911/report.md`；此次未改变训练方法或原模型选择。

## 最新：直接调用官方实现，Office → Industrial，各自 SFT + RL700

这轮使用固定官方源码的实际 `sft.py`、`rl.py`、`evaluate.py`、`calc.py` 入口。
两组都从 **Qwen2.5-0.5B Base** 独立重新 SFT，再训练 RL 到700次优化器更新。
原始 CSV 历史、三任务自然配比、奖励字典、采样器、约束解码和损失均保留。
兼容、日志与保存方面的改动见 [源码说明](reproduction/official_direct/README.md)，每次运行另存准确的 `runtime.patch`。

在服务器仓库根目录、原训练环境中执行（首次补齐官方入口依赖）：

```bash
python -m pip install -r requirements-official-direct.txt
python scripts/run_official_two.py \
  --run-name official_qwen25_two700 \
  --model Qwen/Qwen2.5-0.5B \
  --gpus 0,1,2,3
```

该命令串行完成 **Office SFT → SFT valid → RL700 → 七个 RL valid → Industrial 同一流程**。
训练用四张卡，官方评估入口每次使用指定列表中的第一张卡，全部评估串行。
无需手动 export；脚本设置 CUDA 可见卡、禁用 WandB 和 tokenizer 并行日志。
本地基础模型可替换 `--model /path/to/Qwen2.5-0.5B`；已缓存时可加 `--offline`。
先检查参数可加 `--dry-run`，只准备源代码/数据可加 `--stage prepare`。

SFT 有效 batch1024，四卡 micro4 / accumulation64，最多10epochs，按官方验证loss和早停选模型。
RL 四卡 micro16 / accumulation16，候选 batch1024、G16、每更新64组，LR1e-5、beta.001、
beam sampling、paged_adamw_32bit。保留官方两轮 cosine/warmup3% 学习率计划，**没有把 max_steps 改成700**。
保存并完整验证50/100/175/350/500/550/700，额外保留官方原有定期checkpoint；500/550跨越首次reference同步512。
实际 scheduler、dtype、generation config 和上游来源写入每组记录。
这轮只评 valid，共16份预测；每组与自己的新 SFT 比较，700步仍是官方完整训练的部分复现。

结果在 `results2/official_qwen25_two700/`，权重在 `checkpoints/official_qwen25_two700/`。
可用 `--checkpoint-root /path/to/large_disk` 指定权重父目录，禁止放进results/results2。
每个进程命令仅保留256KiB控制台尾部和64KiB告警尾部；标量指标完整保留，预测保存为gzip，
不复制重复的最终RL权重，不写WandB日志。checkpoint仍包含恢复所需的优化器和reference，权重目录会显著大于结果目录。

中断后重复原命令，或执行结果目录中的 `resume.sh`。已完成阶段跳过；RL恢复时同时加载reference，
避免跨512同步后换回初始reference。仅支持这两组700步以内、第一轮内恢复，配置和源文件发生变化会拒绝继续。
损坏或未完成的checkpoint会隔离保留；完整checkpoint校验失败直接报错。任一步失败即停止，不继续下一组。
训练停止后打包小结果目录：

```bash
python scripts/package_official_two.py official_qwen25_two700
```

本地已进行真实tokenizer全量数据预检和CPU测试；本机无CUDA，未启动这两组正式训练。

## 前一轮五组实验：写入 results2，RL 均为 350 步

在已运行旧六组的服务器上，同步本次新脚本和 `reproduction/upstream_minionerec/`，使用原训练环境，在仓库根目录执行：

```bash
python scripts/run_rl_five.py \
  --run-name minionerec_five350 \
  --source-suite results/qwen3_six350 \
  --gpus 0,1,2,3
```

| 组 | 模型 / 数据 | SFT 来源 | RL 差异 |
|---|---|---|---|
| 01_history10_restart | Qwen3-0.6B / Office | 导入旧第六组 SFT10 step378 | 原第六组重跑，beta.04、确定性beam |
| 02_history10_adamw | Qwen3-0.6B / Office | 同一导入 SFT10 | 仅将优化器改为 adamw_torch |
| 03_official_qwen3_office | Qwen3-0.6B / Office | 从原始 Qwen3 重新 SFT | 官方自然任务配比、原始 RL prompt、beta.001、beam sampling |
| 04_official_qwen25_office | Qwen2.5-0.5B Base / Office | 从原始 Qwen2.5 重新 SFT | 同第03组官方配方 |
| 05_official_qwen25_industrial | Qwen2.5-0.5B Base / Industrial | 独立重新 SFT | 同第04组官方配方、更换数据集 |

前三项旧方案的 FP32 概率计算、单任务 RL、CE 混合损失均未加入。五组都从自己的 SFT 初始化，不继承旧中断 RL 或上一组 RL；之后重复同命令才恢复本轮自身 checkpoint。SFT 采用原最多10epochs/验证loss最优/早停规则，**不限制350步**。

结果固定在 `results2/minionerec_five350/`，权重默认在 `checkpoints/minionerec_five350/`。用 `--checkpoint-root /path/to/large_disk` 可更换权重父目录，禁止放进results/results2。旧 SFT10 权重必须仍在服务器；如果已经移动，可用 `--source-sft /new/path/selected_model`，脚本核验旧完成记录中的全部哈希。本地收到的压缩包不含权重，单有压缩包不能启动训练。旧六组目录只读，完成初始化后本轮可独立恢复。

默认允许自动下载/使用缓存中的原始模型。可通过 `--qwen3-model /path/to/Qwen3-0.6B --qwen25-model /path/to/Qwen2.5-0.5B` 使用本地模型；已缓存且不希望联网时加 `--offline`。预检验证结构及SID词表扩展后的参数量严格小于1B，扫描所选实验的全部train/valid任务，提前拒绝截断和tokenizer不匹配。四卡编号可替换，SFT micro默认4、评估batch默认2。

每组RL候选batch1024、G16、每更新64输入组，350步共22400组。保存并用完整valid评估50/100/175/350，SFT基线共4份，合计24份压缩预测。LR1e-5，完整两轮cosine/warmup3%不压缩成350步；reference间隔512。官方数据自然去重使总任务数不同：Office55290（38924/6366/10000），Industrial52775（36259/6516/10000）；各自完整scheduler为1728/1650、warmup52/50。旧history10配方仍55842、scheduler1746/warmup53。

后三组是**官方配方经必要兼容适配后的350步短程复现**，不是未经修改的官方脚本或完整两轮结论。固定官方源码commit、原始数据构造和全部差异见 [upstream说明](reproduction/upstream_minionerec/README.md)。特别是官方RL保留未指定加载dtype的路径，并记录实际精度；官方去重/Fusion SID查表/原始RL提示词保留。奖励使用随行target避免官方train/valid字典覆盖，统一评估使用SFT推荐prompt和确定性beam50。第03与第01是配方整体对照，不作单因素归因。

中断恢复、仅查看计划和打包：

```bash
bash results2/minionerec_five350/resume.sh
python scripts/run_rl_five.py --dry-run
python scripts/package_results2.py minionerec_five350
```

`--stage prepare/train/eval` 分别用于仅准备、仅训练、仅评估，默认all；`--experiments 1 2` 可只执行指定组，始终按数字顺序。任何失败停止后续训练；已完成阶段不重复执行，350步已完整保存但收尾中断时只补记录。checkpoint完整标记在各rank保存完成后写入，缺标记的半成品保留为隐藏 `.incomplete-*` 目录，恢复使用最近完整checkpoint。每次恢复清理回放区间的重复标量日志。

`summary.md/.csv/.json` 汇总相对各自SFT的HR/NDCG变化，`status.json` 区分训练完成和评估完成。每个子进程只保留**256KiB控制台尾部＋64KiB警告尾部**；逐步标量完整保存，预测无损gzip后删除本轮未压缩副本，数据/源码共享。默认不开wandb、不写TensorBoard、不转储logits。打包入口使用同一文件锁，训练正在运行时拒绝打包，停止后的部分结果可打包并显示完成组数。输出为 `results2/minionerec_five350.tar.gz`，不会混入旧results或模型权重。

本地28项定向测试通过（含单独补跑的真实Qwen3 tokenizer测试），最终五组入口8项回归通过；真实Qwen2.5 Base tokenizer在两类数据上的SFT标签/EOS/beam50验证通过。全量官方train/valid预检确认三组词表扩展后参数量约0.596B/0.494B/0.494B，均小于1B。验证记录在 `results2/five_validation_20260913/`；本机无CUDA和服务器SFT权重，尚未启动真实四卡实验。

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

### 串行运行 A/B/C：每组先验证 175 步

针对 Enhanced 的两个配置差异，新增三个独立的短跑实验。三组均从同一个选中的 SFT checkpoint 开始，学习率保持 `1e-5`；这是参数适配实验，原官方复现入口的默认参数不变。

| 实验 | beta | RL do_sample | 改动 |
|---|---:|---|---|
| 已有 baseline（不重跑） | 0.001 | True | 当前原配方 |
| A | 0.04 | True | 加强 KL 约束 |
| B | 0.001 | False | 确定性 beam 候选 |
| C | 0.04 | False | 两项一起改 |

关闭 `do_sample` 仍使用16 beams生成16个候选，继续计算组内奖励并更新模型；并非单候选训练。该设置作用于训练和原trainer的G16中间验证奖励，最终排名评估始终为确定性beam50。**beta增大或关闭采样不保证提升，也不保证消除KL尖峰。**

同步代码后，在服务器仓库根目录执行：

```bash
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/run_rl_abc.py \
  --source-run qwen3_h50_seed42_rl16_fixed \
  --run-name qwen3_h50_abc175 \
  --dataset Office_Products --gpus 0,1,2,3 \
  --rl-micro-batch 16 --stop-after-steps 175
```

脚本先实体复制三组SFT模型、已处理数据及长度报告并校验三者一致；只校验复用历史50数据，不重建、不重训SFT。随后严格串行执行 **A训练→A验证→B训练→B验证→C训练→C验证**，复用指定四卡。三组都从同一SFT重新初始化优化器、scheduler和reference，不继承前一组RL。每组micro16×累积16×4卡，有效候选batch1024；奖励、任务配比、优化器、reference同步及attention实现保持原设置。

默认每组175个优化器更新，保存50/100/175；保留完整2epoch的cosine调度和warmup（当前Office共1746步、warmup53步）。175覆盖已有baseline的113～116步尖峰与第一个已评估的下降点，适合初筛；**不代表完整训练结论**。新增快照只保存、不插入额外G16验证，原约10%的中间验证仍照常。每组训练结束后评估SFT和三个快照，全部使用完整验证集，输出HR/Recall@5/@10、NDCG@5/@10；不运行测试集，不自动替换selected_model。

若更急，可换新名字并指定 `--stop-after-steps 100`，默认只保存50/100；但100尚未覆盖此前113～116步的尖峰。只想看停止点可加 `--snapshot-steps 175` 减少保存和评估开销，代价是缺少早期曲线。不要用 `--max-steps` 替代提前停止预算。

中断后重跑**完全相同的命令**，或：

```bash
bash results/qwen3_h50_abc175/resume.sh
```

会跳过完成的训练，校验并跳过完成的评估；未完成组从自身最新完整RL checkpoint恢复，出错时停止后续组。`--stage prepare/train/eval` 分别用于仅准备、仅训练、仅验证；默认 `all`。改步数/GPU/微批次等设置需新run-name。单组可用 `run_rl_short.py --beta 0.04 --no-do-sample ...`，未指定时仍为beta .001和采样开启。本次修改了短跑脚本指纹，旧版本创建的短跑应使用原代码恢复；常规fixed源run的14个核心文件未改，不需要迁移其SFT。

运行中的三个独立目录为 `results/qwen3_h50_abc175_A`、`_B`、`_C`；权重位于源checkpoint根目录的同级新run目录，始终在results外。套件每完成一组就将日志/配置/源码/预测等实体复制至下列目录，并刷新跨组汇总；套件可独立复制分析：

```text
results/qwen3_h50_abc175/
  abc_config.json, shared_sft.json, commands.jsonl, resume.sh
  summary.md, summary.csv, summary.json
  A.log, B.log, C.log
  Office_Products/A/                 # B/、C/结构相同；普通文件副本
    run_config.json, sft_import/, source/, short_source/
    Office_Products/rl/              # train.log、candidate_config、scheduler、checkpoint路径/step
    Office_Products/diagnostics/short_validation/  # 逐样本验证预测及指标
```

分析副本不替代原 `_A/_B/_C` 工作目录用于断点恢复；确认不再续训后才考虑清理工作目录。一条命令打包三组（不包含权重）：

```bash
python scripts/package_results.py qwen3_h50_abc175
```

本地已验证三组真实tiny Qwen3/ReReTrainer的候选模式、G16分组及反向传播，并通过调度/恢复和串行脚本回归；**本机无CUDA，未执行这三组的真实四卡训练/评估，尚无新推荐指标。**

### 六组串行实验：所有 RL 固定 350 步

在服务器仓库根目录、原四卡训练环境中执行：

```bash
python scripts/run_rl_six.py \
  --run-name qwen3_six350 \
  --source-run qwen3_h50_seed42_rl16_fixed \
  --gpus 0,1,2,3
```

前五组从同一 history50 SFT step378 独立初始化。共同配置以已有 C 为基准：LR `1e-5`、beta `.04`、确定性 beam、G16、micro16 × 累积16 × 4卡，候选 batch1024。每组仅作表中变化：

| 顺序 / 目录 | 变化 | RL 更新步 | 每步输入组 / 候选数 |
|---|---|---:|---:|
| `01_lr5e6` | LR 降至 `5e-6` | 350 | 64 / 1024 |
| `02_beta01` | beta 增至 `.1` | 350 | 64 / 1024 |
| `03_batch128` | 累积改为2，候选 batch128 | 350 | 8 / 128 |
| `04_equal_tasks` | 三类 RL 任务等量采样 | 350 | 64 / 1024 |
| `05_g32` | G32，micro32，累积16 | 350 | 64 / 2048 |
| `06_history10` | 原始 Qwen3 → history10 SFT → RL | 350 | 64 / 1024 |

严格按1→6串行：先共享 SFT50 验证，然后逐组 RL→快照验证；到第六组才重训 SFT10、验证 SFT10，再 RL→快照验证。六组 RL 合计2100次优化器更新，第六组 **SFT不受350步限制**，沿用10 epochs、LR3e-4、全局batch1024、warmup20、每5%验证保存、patience3，选验证loss最小的模型。三列 item/SID/title 输入历史同步截为最近10条，样本、标签和数据划分保持一致；先用真实 tokenizer 扫描所有启用的 train/valid 任务长度。需要源run的 SFT 权重，以及已缓存的原始 `Qwen/Qwen3-0.6B`；本地原始基座路径可用 `--base-model /path/to/Qwen3-0.6B` 指定。

第4组保持训练行数55842，以seed42在各任务内打乱、循环抽取，各18614条；因此改变任务权重而不同时缩短数据集。它不是 Enhanced 三任务各10000条的完整复刻。第5组保持每步64个输入组，候选计算量翻倍；micro32的显存需求也高于micro16，脚本不会失败后自行换配方。

**350指优化器更新次数，不是 `max_steps=350` 压缩调度。** 保留之前方案按输入样本量对齐的完整 cosine/reference 节奏：Office普通组调度1746步、warmup53、reference间隔512；小batch组调度13968步、warmup424、reference间隔4096。因此第3组只见2800个输入组，其他组见22400个，且第3组350步全部在warmup内。这轮是固定更新预算的初筛，不能据此得出等样本预算或完整 Enhanced 训练结论。

每组保存并在完整验证集评估50/100/175/350，统一确定性beam50。前五组共享一次SFT基线评估，第六组比较自己的SFT10，共26次验证；不自动评估test。每组仍记录原间隔的内部G16/G32验证奖励，新增快照只保存，不插入额外训练期验证。汇总在 `results/qwen3_six350/summary.md`（另有CSV/JSON），包含相对各自SFT的HR10、NDCG10变化和实际样本预算。

同一命令重新运行即可恢复；也可使用生成的命令：

```bash
bash results/qwen3_six350/resume.sh
python scripts/package_results.py qwen3_six350
```

完成的训练/评估经校验后跳过，未完成RL从本组最新含reference的checkpoint恢复；350已保存时只补记录，不增加更新。失败停止后续组。`--dry-run` 只显示六组配置，不复制文件、不访问模型权重；`--stage prepare/train/eval` 支持分阶段；`--experiments 1 2` 可执行子集，仍按数字顺序。配置、数据、代码指纹锁定，改变配方需新run-name。`--checkpoint-root` 指定新suite权重的父目录，默认与源run同级。

新的results采用紧凑布局：源码/原始来源只存一次，两套history数据各存一份，六组结果直接写在suite内；权重、优化器、reference均放在results外。保留每步结构化标量、关键配置/哈希、完整验证指标及预测；预测保存为无损 `.jsonl.gz`，核验解压后SHA256再移除新生成的未压缩副本。每个子进程普通控制台仅保留末尾256KiB，警告/错误末尾64KiB。真实Office预测抽查从14.1MB压至3.8MB（约减少73%）。旧results约1.39GB，其中普通log仅10.9MB，主要占用是重复预测和数据；此次不改写旧结果及其清单。

CPU验证覆盖真实tiny Qwen3的G32生成/奖励/反向传播、调度停止及恢复权重一致、四rank采样布局、串行路由、压缩核验和打包。实际四卡CUDA、ZeRO2/paged AdamW运行仍需在服务器执行。

### 从已有 SFT 做单组 350 步 RL 快速验证

原 Office 实验在175步时已出现下降，可以先检查前350步，无需每次跑满1746步。这个实验用于观察早期退化是否减轻，不能证明完整训练有效或无效。下面只把RL学习率从原 `1e-5` 改为 `5e-6`，其他训练方法保持一致；这是参数适配实验，原官方配置入口不变，issue #5 本身没有给出已验证的修复参数。

同步最新代码后，在服务器仓库根目录运行（源run需保留SFT模型副本）：

```bash
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/run_rl_short.py \
  --source-run qwen3_h50_seed42_rl16_fixed \
  --run-name qwen3_h50_seed42_rl_lr5e6_short350 \
  --dataset Office_Products --gpus 0,1,2,3 \
  --rl-micro-batch 16 --learning-rate 5e-6 \
  --stop-after-steps 350 --snapshot-steps 50 100 175 350
```

脚本实体复制选中的SFT模型、已处理的历史50数据及长度报告；校验已有数据后复用，不重新重建历史或重训SFT。RL优化器、scheduler、reference从SFT重新初始化，不加载源run的RL。权重放在源checkpoint根目录下的新run同级目录；旧run不改。

`--stop-after-steps` 计数的是**优化器更新步**。仍按完整2epochs计算scheduler（当前Office为1746步，warmup53步），到350步提前停止；**不要替换成 `--max-steps 350`**，后者会压缩学习率曲线。G16、四卡micro16×累积16、有效候选batch1024、beta .001、任务/奖励/优化器及reference同步规则均保持。原每约10%的采样验证/保存照常；50/100等新增点只保存checkpoint，不插入采样验证，不改变训练随机数序列。为保留新增快照，checkpoint保留数量上限相应增加。

训练停止后，四张卡串行评估SFT（RL step0）及50/100/175/350快照，统一使用原确定性beam50、**完整验证集**，输出HR/Recall@5、@10、NDCG@5、@10及相对SFT差值。本次也重新评估SFT，避免沿用导入结果中的旧模型路径；不运行测试集评价、不修改原run的模型选择或测试汇总。当前服务器每组验证约6分钟，五组额外约30分钟；350步训练粗估3～4小时，受共享GPU和保存开销影响。

中断后沿用保存配置继续，不重复导入：

```bash
bash results/qwen3_h50_seed42_rl_lr5e6_short350/resume.sh
```

也可执行 `python scripts/run_rl_short.py --run-name qwen3_h50_seed42_rl_lr5e6_short350 --resume`。完成的训练和已校验的评估会跳过；未完成训练恢复**新run自身**的最新完整RL checkpoint（包含reference），最后一步已保存但进程退出时只补齐记录，不再多训练一步。更改LR/步数/微批次应换新run，不能修改已有配置后续跑。该实验使用独立入口，不要用 `reproduce.sh` 恢复短跑。

单阶段：首次命令加 `--stage prepare` 仅复制/校验；随后用 `--run-name ... --resume --stage train` 仅训练，`--stage eval` 仅评估。`--snapshot-steps` 可选；默认保存不超过停止预算的50/100/175步以及最后一步。不传学习率时保持官方默认 `1e-5`，可用于短跑对照。

产物在 `results/<run_name>/`：顶层 `summary.md/csv/json` 是验证集对比；`Office_Products/rl/` 保存训练日志、逐步指标、实际scheduler预算及checkpoint路径/step；`Office_Products/diagnostics/short_validation/` 保存每个模型的验证预测、指标、命令及评估日志。配置和源码快照随run保留；导入的旧SFT测试结果仅归档至 `sft_import/`，不进入新汇总。打包给分析时执行：

```bash
python scripts/package_results.py qwen3_h50_seed42_rl_lr5e6_short350
```

本地已用真实小型Qwen3/ReReTrainer进行CPU学习率前缀、停止和断点恢复验证；本机没有CUDA，**此350步实验的真实四卡训练/指标尚未执行**。

### 显存允许时提高 RL 微批次

只需设置 `--rl-micro-batch`，梯度累积自动为 `256 / micro`：16→累积16，32→累积8，64→累积4；均为四卡、G16、每次完整更新 1024 个候选。最新日志显示当前共享GPU环境中32也发生OOM，建议先回到16。增大微批次可能减少循环开销，但 prompt padding 增多也可能抵消收益，未实测 GPU 加速倍数。

最新 micro64 日志确认在生成候选时 OOM：GPU 2 总显存 44.39 GiB，同卡另一进程占用 9.14 GiB，当前训练占用 33.28 GiB，仅剩 58.12 MiB 时申请 1.54 GiB 失败。前几个短输入能运行不代表长输入也能运行。先回到 micro32；若仍 OOM，再用 micro16。检查方法见 [debug.md](debug.md)。

micro32随后在21次更新后也发生OOM：反向传播申请4.78 GiB，GPU 2剩余2.76 GiB，外部进程仍占9.14 GiB。下面命令改用micro16；不保证所有长输入都能在当前共享显存下稳定运行。

**从已有 SFT 重新开始 micro16 RL**：先停止自己占用同一组 GPU 的旧训练，将 `scripts/start_rl_from_sft.py`、`scripts/migrate_rl_config_fix.py`、`minionerec_trainer.py` 和 `reproduction/evaluate.py` 的最新版本同步到服务器仓库。在仓库根目录执行：

```bash
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

python scripts/start_rl_from_sft.py \
  --source-run qwen3_h50_seed42 \
  --run-name qwen3_h50_seed42_rl16_fixed \
  --dataset Office_Products --rl-micro-batch 16
```

此命令创建新 run，继承源 run 的 GPU、原始模型名、评估设置及 checkpoint 根目录；复制已完成的 Office SFT 产物、50 条历史数据及长度扫描结果，并将验证集选中的 SFT `selected_model` 实体复制到新 checkpoint 目录。SFT 训练不重跑，旧 RL 的权重/优化器/步数不继承，新 RL 从 step 0 开始。如果源 run 使用旧版评估器，脚本自动把旧评估移入新 run 的 `sft_import/superseded_sft_evaluation/`，先重评 SFT valid/test，再启动 RL，最后评估并汇总。旧 run 不修改。

若原始 `qwen3_h50_seed42` 已删除，可将 `--source-run` 换成仍保留完整 SFT 产物和模型副本的 `qwen3_h50_seed42_rl32` 或实际的 rl64 run 名称。脚本校验该副本的模型哈希和选中 step，无需最早的 checkpoint 仍存在。`sft_import/record.json` 记录来源及复制文件哈希；大模型权重仍在结果目录外。该入口只运行所选类别；后续可用常规入口在新 run 中执行 Industrial 全流程。

`--prepare-only` 可只导入而不启动训练。新 run 已创建后需要继续运行时，保持 CUDA 环境设置并执行 `bash results/qwen3_h50_seed42_rl16_fixed/resume.sh`；无需重复导入。首次运行从 SFT 开始，之后中断才恢复新 run 自己的 RL checkpoint。打包命令为 `python scripts/package_results.py qwen3_h50_seed42_rl16_fixed`。

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

## 用已有 checkpoint 诊断 RL 的验证集表现

在服务器上串行评估已保存的 RL checkpoint，不重新训练、不重新处理历史数据。默认比较 SFT 基线和 RL 的 175、350、1400、1575 步；SFT 和训练选中步数（本次为 1400）的已有验证结果通过核验后直接复制复用，因此本次实际只需重新推理三个 checkpoint。统一使用原 `reproduction.evaluate` 的确定性 beam50，输出 HR/Recall@5、@10、NDCG@5、@10，以及相对 SFT 的差值和对应训练期间的 `eval_reward`。

同步新增的 `scripts/evaluate_checkpoints.py` 和 `scripts/evaluate_checkpoints.sh` 后，在服务器执行：

```bash
conda activate minidiprec
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

cd /data/zuorongchang/project/MiniDIPRec
bash scripts/evaluate_checkpoints.sh \
  --run-name qwen3_h50_seed42_rl16_fixed \
  --dataset Office_Products --gpus 0,1,2,3
```

该脚本只调用 `--split valid`，不读取测试集样本或指标，不改写原 `selected_model`、训练配置、完成标记和顶层测试汇总；新增文件也不会改变原 runner 的训练源码指纹。验证结果用于诊断“是否较早下降”和“采样奖励与确定性排名指标是否一致”，不会自动替换原实验选中的模型。复用的 RL 1400 结果来自 `training.json` 标明该步导出的 `selected_model`，没有重新比较两个目录的权重字节。

参数和恢复方式：

- `--dry-run`：只核验现有结果、打印计划和所需 checkpoint 路径；不写文件、不加载模型。缺失权重会明确显示，正式执行会在 GPU 推理前报错。
- `--steps 175 350 525 700 875 1050 1225 1400 1575`：检查更多已保存步数。不要填写已删除的 checkpoint；SFT 基线始终包含。
- `--checkpoint-root /path/to/large_disk/minionerec_checkpoints`：权重移动后指定根目录，其下应包含 `<run_name>/<dataset>/rl/checkpoint-<step>/`；默认使用原 `run_config.json` 记录的路径。
- `--batch-size 1`：评估显存不足时降低每卡评估批次；默认沿用原值 2，和 RL 训练 micro batch 无关。
- 中断后直接重跑**同一条命令**：校验并跳过已完成项；未完成的 checkpoint 从验证集开头重算，不续算其部分预测。
- 修改步数、batch、GPU 或其他诊断配置后，增加 `--diagnostic-name checkpoint_validation_v2` 写入另一目录。
- `--reevaluate-selected --diagnostic-name checkpoint_validation_fresh`：也重新评估 SFT `selected_model` 和请求列表中的 RL 选中 checkpoint，需对应权重仍存在。

脚本核验当前评估源码与实验快照一致，校验准备好的 valid CSV、SID 映射及长度报告，再校验复用预测的覆盖率、行标签、50 个合法候选和指标重算。新推理记录权重 SHA256，检查完整 checkpoint/tokenizer 和 step，强制只看到指定四卡，禁用 Hugging Face 在线下载。全部产物是普通文件；模型权重仍留在 checkpoint 目录。

输出位于 `results/<run_name>/<dataset>/diagnostics/checkpoint_validation/`：

```text
config.json, commands.jsonl, commands.sh, environment.log
source/, inputs/                           # 评估源码及 valid 输入/配置快照
summary.md, summary.csv, summary.json       # 仅验证集诊断汇总
sft/, checkpoint-175/, checkpoint-350/, ...
  valid.predictions.jsonl, valid.metrics.json
  provenance.json, complete.json
  eval-valid.log                           # 实际重新推理的 checkpoint 才有
```

训练内 `eval_reward` 使用 G16 采样候选，不能直接当作 beam50 的 Recall/NDCG。汇总明确区分 SFT 训练步数（378）和 RL 进度（SFT 基线记为 0）。打包包含诊断结果的整个 run：

```bash
python scripts/package_results.py qwen3_h50_seed42_rl16_fixed
```

本地已完成 CPU 编排/恢复/完整性检查及真实 Office 产物的 dry-run；**本机没有 CUDA 和服务器 checkpoint，尚未执行新增 checkpoint 的真实四卡推理**。

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
