# Experiment history

## 2026-09-11 — 保留完整调度的短 RL 验证入口

用户希望缩短训练步数快速检查效果。新增 `scripts/run_rl_short.py` / `train_rl_short.py`，默认提前停止350个优化器更新步，保存50/100/175/350；使用原ReReTrainer、三类RL任务、exact+ranking奖励、G16、temperature1、beta .001、paged AdamW和reference同步。完整2epoch/cosine/warmup3%调度保持，未用max_steps压缩调度；原周期采样验证照常，新快照只保存。为避免删除新增诊断checkpoint，保留上限增加。没有新增attention/约束解码加速。

学习率作为显式参数，默认仍1e-5，文档给出5e-6的新run单参数适配实验。此为待验证建议，不是issue #5提供的已证实解法。源SFT/历史50数据/长度报告实体复制并核验；源RL不继承，源run不改。现有 `reproduction/*.py`、trainer、四卡官方配置指纹不变；原SFT导入入口增加可选短跑元数据/归档分支，常规入口行为不变。

训练后只用完整验证集，以同一beam50设置重新评估SFT及所有请求快照，记录相对SFT的HR/Recall/NDCG差值；不进行测试集评价或替换原选模。顶层汇总仅验证结果，旧SFT测试结果归档。支持prepare/train/eval分阶段和同run恢复；在最终checkpoint已经保存的情况下从其状态补齐记录，不多训练一步。配置变更拒绝续跑，独立记录新增脚本指纹和来源，权重仍在结果目录外。

本地定向CPU测试 **26/26 PASS**，原fixed run的14个核心源码哈希完全一致，`git diff --check`通过。真实tiny Qwen3 + ReReTrainer完整/短跑初始LR序列相等，停止在指定global_step且scheduler总步数不变；模拟保存后中断，恢复后模型及reference权重与连续短跑逐项一致；预算已到时拒绝再更新。其余回归覆盖原配置字段对齐、独立SFT副本/源run不变、参数校验、仅验证集路由、恢复幂等及源码变更拒绝。**无CUDA、无服务器权重，未执行350步真实GPU训练和新的推荐评估指标。**

## 2026-09-11 — 结合上游 issue 排查 RL 下降原因

阅读用户提供的issues.md（六段有内容、末尾第七段为空，编号是用户整理编号；未联网验证讨论/PR状态），区分Owner回复与其他用户观察。第六段Owner明确解释exact+负rank penalty、全错组奖励归零，与当前官方奖励实现一致。当前1746次训练日志均候选多样性1、completion长度5，reward/grad_norm非零；与issue中的空EOS/重复输出/全零梯度不同。关键依赖版本也匹配官方requirements，已修复generation默认值覆盖，不建议无证据重装或把RL改成确定性生成。

按G16唯一候选估算，训练每更新有奖励命中组的比例中位约34.4%，验证约22.1%～22.8%；两者口径不同，训练混合三类任务。全错组虽无推荐奖励优势，仍可贡献KL项，不能说完全不影响训练。推导并核验一组仅第j名命中时的平均reward为w_j/(16Σw)，正负抵消使原始reward较小，不代表命中率0.2%或无梯度。43条舍入loss0也不能视为故障。

结合175步已经退化、113～116步KL/梯度尖峰与未见item目标损失，当前优先假说为前期优化漂移、KL/数值尾部以及稀疏奖励下的任务信号差异，未确定因果。issue中“主要改beta，指标到0.0996”未提供beta值/方向，0.0996不能当作beta建议。1400在本次四个RL checkpoint中仍NDCG10最高，选模错误降为次要怀疑。独立Preference Summarization未启用、RL description任务存在是代码事实，是否解释论文差距仅为用户推测；本轮不增删任务。

额外核验：实际自定义ReReTrainer的compute_loss并不使用序列化GRPOConfig的loss_type/epsilon分支，入口dapo=False/gspo=False；不能把它当作现代TRL GRPOTrainer解读。当前有意保留官方trainer。完整对照、建议的逐token/分任务判别检查、原issue副本和可复查统计在 `results/qwen3_h50_seed42_rl16_fixed/Office_Products/diagnostics/issues_analysis_20260911/`。未更改训练代码/超参数、未新增GPU实验或测试集选模。

## 2026-09-11 — 收到 checkpoint 验证结果：前175步已退化

收到 `diagnostics/checkpoint_validation/`，46个文件实体复制至原run类别下的diagnostics目录并校验SHA一致，原传入目录保留并加入gitignore。6份评估/脚本源码、9份输入快照与原run匹配；五组共24330条验证预测全量核验通过，generation config完全一致。175/350/1575均有四卡valid-only命令、日志和完成标志；每次约6分钟。SFT/1400与原验证预测字节一致。未执行新的训练或本地GPU推理。

| 模型 | valid HR/Recall@10 | valid NDCG@10 | Top10命中数 |
|---|---:|---:|---:|
| SFT | 0.232635 | 0.188749 | 1132 |
| RL175 | 0.211673 | 0.172099 | 1030 |
| RL350 | 0.211262 | 0.170102 | 1028 |
| RL1400（原选中） | 0.213933 | 0.174388 | 1041 |
| RL1575 | 0.214139 | 0.173798 | 1042 |

175步约为总更新的10%，此时HR@10净少102次命中；175→1400仅净回升11次。早期丢失的178条Top10命中中，128条到1400仍未恢复。1400仍为这四个RL模型中HR@5/NDCG@5/NDCG@10最高者，1575的HR@10只多1次。因此此前“选模指标不一致”不能作为本次下降的主要已证实解释；观察更支持前期退化后未充分恢复，未证明单一因果关系。

前113～116步KL/grad_norm尖峰处于0～175的未评估区间，需优先调查但不能据此认定BF16/裁剪错误。175也早于首次512步reference同步，第一次同步不能解释已存在的175退化。未评估其余五个RL checkpoint，不能宣称整个训练所有checkpoint均不如SFT。

配对用户bootstrap（1795用户、2000次、seed42）四个RL对SFT的四项指标逐项区间均负；175→1400的HR10/NDCG10区间跨0。仅为当前模型对和验证用户的不确定性，不代表跨训练seed结论。分组显示1400净损失91中85来自历史≤10；重复目标净+8、未见目标净-99。未据此调参或改原测试汇总。

分析代码、报告、JSON/逐样本CSV与PNG/PDF保存在 `results/qwen3_h50_seed42_rl16_fixed/Office_Products/diagnostics/checkpoint_analysis_20260911/`。模型哈希记录一致性已核验，本地无权重，无法独立重算服务器模型哈希或比对selected_model与1400权重。

## 2026-09-11 — 已保存 RL checkpoint 的验证集诊断入口

新增 `scripts/evaluate_checkpoints.py` / `.sh`，默认串行比较 Office SFT 基线与 RL 175、350、1400、1575 步。复用校验通过的 SFT 和选中步数的现有验证预测，其余 checkpoint 使用原 `reproduction.evaluate`、四个指定 GPU、BF16、确定性 beam50；仅验证集，不启动训练、不读取测试集指标、不改原选模与汇总。新增代码位于 scripts，训练/评估源码及原 run 指纹不变。

先检查原评估源码快照、valid 数据/catalog/长度指纹，再逐行核验复用预测的覆盖率、目标、历史、候选合法性及指标重算。正式推理前检查 checkpoint 的模型分片、tokenizer 和 global_step；记录权重 SHA256。诊断配置、源码/输入快照、命令/环境日志、逐样本预测及汇总保存在该 run 类别下的 `diagnostics/checkpoint_validation/`，不复制权重、不创建软链接。复用 selected_model 结果的步数关联来自原 training.json，明确标注未字节比对导出与 checkpoint 权重。

同命令重跑校验并跳过已完成项，中断项从验证集开头重评；文件锁避免同目录并发写入，变更配置需独立 diagnostic-name。支持指定步数、迁移后的 checkpoint 根目录、评估 batch 和强制重评 selected 模型。汇总附原训练的 eval_reward 和相对 SFT 指标差值，SFT 基线记为 RL step0，另记原 SFT step378；不会根据此次诊断自动替换原模型。

本地定向 CPU 回归 **13/13 PASS**（8 项新增诊断测试、5 项已有 SFT 导入回归），覆盖串行 valid-only/four-GPU 路由、复用、恢复、数据/源码/预测篡改拒绝、缺模型/错误 step 拒绝、完整分片检查、并发锁及权重目录迁移。真实 Office 产物 dry-run 校验已有两组验证预测通过，确认只需新增175/350/1575推理。**本机没有 CUDA 或服务器权重，未执行新增 checkpoint 的真实 GPU 评估；没有新增推荐指标或选模结论。**

## 2026-09-11 — Office micro16 fixed 完成与负结果审计

用户提供 `qwen3_h50_seed42_rl16_fixed/` 完整产物，已实体复制到 `results/qwen3_h50_seed42_rl16_fixed/`；原目录不改，加入gitignore避免上传数据。4×L40、Torch2.6+cu124、DeepSpeed0.18，micro16/累积16/G16，有效batch1024。RL完成1746step/2epochs，训练含中间验证耗时17.80小时，按最高eval_reward选中1400step。SFT保留原378step权重并使用fixed重评。

| split | 阶段 | HR/Recall@5 | HR/Recall@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| valid | SFT | 0.206535 | 0.232635 | 0.180333 | 0.188749 |
| valid | SFT+RL | 0.194616 | 0.213933 | 0.168188 | 0.174388 |
| test | SFT | 0.139334 | 0.165023 | 0.115878 | 0.124228 |
| test | SFT+RL | 0.132141 | 0.150226 | 0.109252 | 0.115154 |

14份源码快照、6份生成文件及官方源数据哈希核验通过。valid/test的SFT和RL共19464条预测，样本覆盖、history/target、SID/item映射、50个唯一合法候选、排序和指标重算通过。完整评估参数一致，RL训练日志无0.6覆盖或OOM，结尾析构异常发生在训练完成后。配置与命令确认从SFT新启RL，四卡限制及batch设置一致；没有服务器模型权重，无法独立比较selected_model与checkpoint1400的字节内容。

测试HR@10净少72次（803→731，相对-8.97%）：丢失170条，新增98条；丢失中83条落到11～50名、87条不在Top50。历史≤10样本贡献净损失71条，历史>10贡献1条。目标在历史出现的样本净+9，未出现的样本净-81。预测Top10全局SID覆盖2138→2455，不能简单归因于输出少数商品。fixed重评相对旧SFT采样评估的HR@10仅少1次，无法解释RL净少72次。

最需进一步排查的是早期优化稳定性：step114日志KL估计值993168.81、loss993.9177，step116日志grad_norm198695.48。这个KL是官方指数概率比的采样估计，不是精确全目录KL；梯度范数也不能直接视为裁剪后的更新范数。九次验证reward近乎持平，1400step相对175step只高1.08%，exact奖励相同。当前选择的是采样G16综合奖励最佳，不保证确定性beam50 Recall/NDCG最佳，且初始SFT不参与RL checkpoint选择。

本次只分析已存在的产物，没有改训练/奖励/超参数，没有根据test调整模型。用户分组配对bootstrap显示该模型对在当前测试用户样本上HR@10差值95%区间约[-2.136,-0.824]个百分点；不代表跨训练seed结论。建议下一步仅在valid评估已有checkpoint，判断退化发生时间及选模指标是否一致；GPU检查和逐token KL追踪均尚未执行。

完整报告、复查脚本、配对排名CSV、验证曲线CSV、PNG/PDF图位于 `results/qwen3_h50_seed42_rl16_fixed/analysis_20260911/`。原101份文件已逐一哈希确认复制一致。仅收到Office结果，Industrial尚未完成。

## 2026-09-10 — micro32 反向传播 OOM

用户更新的日志在14:37:42退出：DeepSpeed采用梯度累积8，完成21次更新后 `backward()` 申请4.78 GiB失败。GPU 2容量44.39 GiB，空闲2.76 GiB，外部进程3492676仍占9.14 GiB，本训练进程占30.85 GiB。与前次micro64在生成时OOM不同，本次栈仅定位到反向传播，无法确定具体算子或断言只有碎片问题。

该日志仍有 `generation_config ... temperature: 0.6`，因此不是修复后的RL运行。建议同步生成修复后从SFT启动micro16/累积16，有效batch1024及G16不变；可尝试 `expandable_segments:True` 缓解碎片。此次未改trainer、奖励或生成算法。README/debug已更新，远端原日志归档在 `results/oom_rl32_20260910/checks/remote_train.log`。本机无CUDA，未执行micro16四卡容量验证，也没有新的完整RL指标。

## 2026-09-10 — micro64 OOM 与生成默认值覆盖修复

用户最新 `train.log:803` 确认 micro64/累积4 完成3次更新后，候选生成的 Qwen3 attention 发生 CUDA OOM。GPU 2 总44.39 GiB，另一进程占9.14 GiB，当前训练占33.28 GiB，剩余58.12 MiB时申请1.54 GiB失败。建议回到micro32/累积8，保持有效候选batch1024和G16；长输入和外部任务占用决定能否稳定运行，不根据前几步显存承诺micro64可用。

独立发现并复现 Transformers 4.57.1 的 generation config 合并行为：显式 `GenerationConfig` 中等于全局默认值的字段，仍会被保存的模型默认值覆盖。最新RL日志温度变为0.6；原 `qwen3_h50_seed42/Office_Products/sft/eval-{valid,test}.log` 同时记录 `do_sample=True, temperature=0.6`。**更正此前评估结论：下文保存的 SFT 指标是采样 beam 的历史结果，不是声明的 deterministic beam50 基线，不能与修复后的 RL 直接比较。** 样本、标签、指标计算的既有核验仍成立；SFT 训练及验证 loss 选中的 step378 权重不受影响，无需重训 SFT。

在实际启用的 RL beam 生成调用及统一评估调用显式设置 `use_model_defaults=False`，恢复官方 RL beam sampling/temperature1，最终评估保持 deterministic beam50。此为 Qwen3/Transformers 兼容修复，不是调参。旧RL从SFT重新开始，旧SFT valid/test重评；不根据测试指标选择设置。

更新 `start_rl_from_sft.py`：严格只接受 dtype 复制和本次生成修复差异；原源码及旧评估归档至新 run 的 `sft_import/`，自动重评再训练，原 run 不改写。支持从此前 rl32/rl64 run 的独立 SFT 副本导入，校验模型哈希/step，保留多次导入来源，即使最早 run 已删除也不依赖旧路径。

本地 CPU 回归 **29/29 PASS**，包含真实 ReReTrainer 生成参数检查、真实 Qwen3 tokenizer + 完整 Office catalog beam50、旧评估归档与重评路由、旧源删除后再次导入及篡改拒绝。记录在 `results/generation_fix_20260910/checks/regression.log`。**本机没有 CUDA GPU，未执行修复后的真实 SFT 重评或四卡 RL；未报告新的推荐指标、显存上限或加速倍数。**

## 2026-09-10 — 从已有 SFT 开始新的 micro32 RL

用户明确改为从 SFT checkpoint 初始化新 RL。新增 `scripts/start_rl_from_sft.py`：在独立 run 中实体复制所选类别已经完成的 SFT 产物、已校验历史数据/长度报告和 SFT selected_model；模型权重放到新 checkpoint 目录。保留原 SFT 参数记录、验证选中 step 及源路径，在 `sft_import/` 保存导入来源和模型/产物哈希。旧 run、旧 RL 训练进度和 checkpoint 均不修改。

此版本入口继续使用原串行 runner：跳过导入的 SFT/评估，仅以对应 SFT 模型初始化所选类别 RL，micro32/accum8，从 step0 开始；新目录不含任何旧 RL checkpoint，实际 RL 训练命令不传 `--resume`。随后执行 valid/test 评估和汇总。后续发现生成默认值覆盖问题，当前版本导入旧结果时会先重评 SFT，见上方更正。新 run 中断后用生成的 `resume.sh` 恢复。默认仅 Office；如需新 run 的 Industrial 全流程，另用常规入口指定该类别并保留相同配置。

新增导入与命令路由测试：验证旧产物哈希不变、模型真实复制、无旧 RL 目录、只启动一条 RL 训练和两条评估命令、不向 RL 传续训标志；缺模型/目标目录冲突拒绝、复制失败清理。测试首次发现 macOS `/var` 与 `/private/var` 规范路径差异引起配置校验不匹配，已统一新 checkpoint 根目录的绝对路径。**本机无 GPU，也未收到服务器 SFT 权重；未执行真实模型复制或新的四卡 RL。**

最终回归 **26/26 PASS**，日志位于 `results/fresh_rl32_validation_20260910/checks/regression.log`。Python/README Bash 语法和 whitespace 检查通过。原实验的环境、命令和调用记录（若存在）也复制到 `sft_import/`，便于区别旧 SFT 与新 RL 的执行来源。

## 2026-09-10 — 显存允许时将 RL 微批次提高到 32

用户要求加速并确认显存允许。保持默认 recipe 不变，为已有实验提供显式运行适配：`--rl-micro-batch 32`，梯度累积自动从 16 降到 8，四卡每次完整更新仍为 1024 个候选、64 个 G16 prompt 组。实际训练函数、奖励、模型、学习率、attention backend 和训练轮数不变。RL 内部验证微批次跟随变为 32，最终推荐评估配置不变。

新增 `scripts/migrate_rl_batch.py`，避免直接改参数后被运行配置校验拒绝，以及 Transformers 4.57.1 用 checkpoint `train_batch_size` 覆盖新微批次的问题。迁移在结果目录归档原配置、checkpoint JSON 与 step，再更新 batch 元数据；模型、优化器、scheduler、reference 和 RNG 文件不变。源码仍严格检查，可连同前述 dtype 修复完成迁移，保留已有 SFT；未对收到的远端结果执行实际迁移。

本地 CPU 实际 ReReTrainer 验证 micro16/accum4 训练保存后，以 micro32/accum2 恢复：实际 dataloader batch 为32，global_step 正确继续，下一更新的 prompt 恰为应消费样本。实际 Accelerate 四 rank sampler 核验 micro16/32/64、跨 epoch 与非整批数据：每完整更新 prompt 集合一致，G16 分组完整，更新总数一致。并行分配、随机生成调用及尾部补齐可能变化，不承诺逐位复现。**尚未执行四卡 DeepSpeed 的这次 batch 切换及吞吐验证，不宣称加速倍数。**

最终回归 **23/23 PASS**；迁移预览、JSON 备份、参数拒绝、中断重试和重复执行通过。日志：`results/rl_batch_adaptation_20260910/checks/regression.log`。Python、README Bash 语法及 whitespace 检查通过。

## 2026-09-10 — 远端 Office SFT 结果与 RL 初始化修复

检查用户提供的 `results/qwen3_h50_seed42/`。Office SFT 在 step 504（epoch 6.07248）正常早停：step 378 验证 loss 最低为 1.3855953216552734，后续 step 420/462/504 连续三次未改善。评估与 RL 初始化使用该验证集选出的模型，checkpoint 路径记录在 `Office_Products/sft/training.json`，服务器默认目录为 `checkpoints/qwen3_h50_seed42/Office_Products/sft/`。

以下为旧采样评估的历史数值，待 deterministic beam50 重评后替换正式对比基线：

| 数据集 / 阶段 | split | 样本数 | HR/Recall@5 | HR/Recall@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| Office / SFT | valid | 4866 | 0.206535 | 0.232840 | 0.180333 | 0.188823 |
| Office / SFT | test | 4866 | 0.139334 | 0.165228 | 0.115878 | 0.124300 |

独立读取两个逐样本预测文件，核验每行 sample_id、目标 SID/item 与对应 CSV 一致，各行有 50 个唯一合法候选 SID，候选 item 列表与官方映射一致；重算全部指标与保存值一致。此结果来自用户服务器，未在本地重新训练或推理；测试指标只记录，不用于调参或选模型。

RL 在首次更新前写入 `training_args.json` 时失败，日志第 613 行为 `TypeError: Object of type dtype is not JSON serializable`。根因是官方 trainer 将 `args.model_init_kwargs` 字典直接作为模型加载参数使用，原地将 `'bfloat16'` 改成 `torch.bfloat16`，使后续参数记录无法 JSON 序列化。本次仅将该字典复制后用于模型加载，保留实际 BF16、use_cache、trainer、奖励及优化器设置；同时修复训练结束时 `training.json` 的同类隐患。

新增 `scripts/migrate_rl_config_fix.py`，严格验证旧 run 只有这一行修复差异后，归档原始源码哈希与修复前后 trainer，再更新当前源码快照。已完成的 Office SFT/评估、所有数据和 checkpoint 不变；迁移后原命令加 `--resume` 即可继续，不必重新 SFT。迁移用法见 README。

本地已用真实 ReReTrainer + 小随机 Qwen3 重现相同报错；修复后 BF16 模型/参考模型加载、配置无副作用和训练前 JSON 写入通过。CPU RL train/eval、训练后参数记录、reference checkpoint 恢复通过。**未在本地执行修复后的四卡 CUDA/ZeRO-2/bitsandbytes RL；当前没有 RL 指标，Industrial 尚未开始，无法判断 RL 是否提升。**

本次最终回归 **17/17 PASS**，包含真实 Qwen3 tokenizer 检查及迁移归档、重复执行、中断恢复、额外源码修改/快照篡改拒绝。对收到的原实验仅执行迁移 dry-run，未改写其产物。验证日志和预测复核报告位于 `results/rl_startup_fix_20260910/checks/`。Python 解析、README Bash 语法及 git whitespace 检查通过。

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
