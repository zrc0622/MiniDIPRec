# Experiment history

## 2026-09-13 — 新五组串行 RL350：results2 与官方小模型配方

按用户最终确认实现五组：01导入旧第六组已完成的SFT10 step378，重新运行原RL；02同一SFT10且仅改为adamw_torch；03原始Qwen3-0.6B/Office重新官方配方SFT+RL；04原始Qwen2.5-0.5B Base/Office重新SFT+RL；05同模型Industrial独立SFT+RL。所有RL均350次优化器更新，SFT仍为最多10epochs/验证loss选模/早停。未加入已否决的FP32概率计算、单任务或CE混合组。

新增run_rl_five/train_rl_five/five_utils及package_results2入口；旧六组依赖21份源码哈希不变。固定官方commit 0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed，保留原始history10 CSV、任务去重、Fusion查表、原始RL prompt及shuffle。官方三组采用beta.001、beam sampling、paged AdamW、未指定模型加载dtype；记录实际dtype，使用现有兼容ReReTrainer和随行奖励target避免train/valid字典覆盖。统一完整valid确定性beam50评估。此为必要适配后的350步短程复现，不是未经修改的官方执行，也不声称完整复现论文结果。

候选batch1024/G16，每组22400输入组；50/100/175/350四快照，4个SFT基线共24份压缩预测。旧配方55842任务、完整调度1746/warmup53；官方Office55290任务（38924/6366/10000）、调度1728/warmup52；Industrial52775任务（36259/6516/10000）、调度1650/warmup50。均保留完整两轮cosine而在350停止，reference间隔512。

结果仅写results2/minionerec_five350，权重在独立checkpoints目录。每个进程256KiB控制台尾部+64KiB警告尾部，保留完整逐步标量、gzip预测、共享数据与源码。串行失败停止；同命令恢复，已完成350checkpoint不多更新；完整保存标记与各rank barrier避免恢复半成品，半成品仅改名保留。打包与训练共用锁，只打包本轮results2，无旧results和权重。保留用户原summary.md和results.tar.gz。

28项定向CPU测试通过（27项批量通过，原需tokenizer的1项单独补跑通过）；最后改动后五组入口8项再回归通过。真实tiny Qwen2采样RL及中断恢复后policy/reference/scheduler与连续训练一致；模拟GPU验证串行路由、独立SFT、失败停止、重复运行、预测压缩及打包。真实Qwen3和Qwen2.5 tokenizer beam50合法性通过；Qwen2.5两类别SFT屏蔽/EOS通过。三组全量官方train/valid长度扫描通过，词表扩展后参数596390912/494327552/494291712，全部小于1B。记录在results2/five_validation_20260913；本机无CUDA/服务器权重，未执行真实四卡训练、ZeRO2/bitsandbytes恢复或新推荐指标评估。

## 2026-09-13 — 收到六组实验压缩包：五组 RL 未转正，第六组结果不完整

将 `results.tar.gz` 的995个普通文件独立解压到 `results/received_20260913/`，未覆盖旧结果。新套件22次验证、107052条预测及历史C四个快照通过指标重算、标签/顺序/候选/哈希检查；37项共享产物、21个运行源码与当前代码相符。最终222项分析输入和原始压缩包哈希不变。

SFT50 HR10/NDCG10为.232635/.188749。五组350步按LR5e-6、beta.1、batch128、等量任务、G32顺序为.225236/.184864、.225031/.182161、.228113/.185391、.220715/.178791、.224620/.183612，净Top10命中分别−36/−37/−22/−58/−39。全部20个新RL快照NDCG10低于SFT。按1795名用户做3000次配对bootstrap，五组350步NDCG10差值区间均低于0；batch128 HR10区间包含0，其余为负。区间未校正多重比较，不包含训练seed方差。相对历史C多数组缓解退化，但尚未超过SFT。

history10数据逐行验证为三列历史同步取末10项，标签不变；SFT504步早停、选378，HR10.234073/NDCG10.189155，净+7命中且两项配对区间均包含0。其RL只有169条完整逐步记录，末尾400字节NUL，控制台含step170；没有RL完成记录和验证结果。31个命令启动、30个退出且均为0，最后是第六组RL。此包不能证明服务器当前状态，不将第六组判为失败，也不推断未保存checkpoint。

LR/beta/G32同样本顺序、每更新64输入组，三者最大grad_norm均在179步，94～95步也有共同KL尖峰；值得追踪样本和逐token policy/reference概率，但未确定根因。G32提高推算有命中奖励组比例至41.5%，仍未转化为排名增益。小batch累计仅2800组、warmup424尚未结束，不能视为等样本预算的纯batch结论。history10已有KL559/grad232尖峰，但缺少排名结果。建议先补齐第六组，再做固定checkpoint的样本/数值诊断；本次未修改训练代码或启动实验。

压缩包展开985.3MB，旧ABC/fixed/seed42合计840.5MB，占85.3%；新六组144.8MB，其中压缩预测84.93MB、共享数据/来源57.37MB、tail日志1.41MB。没有模型权重混入，日志限额生效；后续可仅打包 `qwen3_six350`。完整中文报告、可复查JSON/CSV、分析脚本、核验日志及已检查PNG/PDF在 `results/six_analysis_20260913/`。用户原始 `summary.md` 与压缩包保持不变。

## 2026-09-12 — 六组串行 RL350 与紧凑 results

用户授权实现此前六组规划，并将所有RL预算统一改为350次优化器更新。新增独立 `scripts/run_rl_six.py`、`train_rl_six.py` 和 `six_utils.py`：LR5e-6、beta.1、batch128、三任务等量、G32、history10依次运行；其他设置以C（LR1e-5/beta.04/确定性beam/G16/batch1024）为基准。前五组共用SFT50，第六组从原始Qwen3重训SFT10，SFT沿用10epochs/验证loss选模/早停，随后RL350。每组保存50/100/175/350，用完整valid统一beam50评估；共用SFT50基线一次，SFT10独立基线一次，共26次评估。失败停止后续组；可重复命令恢复，最终350checkpoint存在时不额外更新，完成SFT后中断导出可从best checkpoint补齐。

保留按输入样本量映射的完整cosine/reference节奏：普通组1746/warmup53/ref512，小batch13968/warmup424/ref4096；固定350预算使小batch仅见2800组、全程warmup，其余22400组，不能视为等样本预算对比。G32保持64输入组/更新，候选数增至2048，micro32。等量任务组总数仍55842，各18614，区别于Enhanced各10000。第六组三列历史同步截断，标签/样本划分不变；真实Office train/valid与Qwen3 tokenizer预检通过：SFT上限256、RL上限1792、补全5token，未读取test。

results盘点1,388,882,402bytes：JSONL946.5MB、CSV264.6MB、JSON143.7MB，普通log仅10.9MB。新suite共享两套history数据、源码/来源仅一份，避免ABC整组副本重复；预测无损gzip并核验解压SHA256后移除新生成原文副本，真实14,062,700bytes预测压至3,796,677bytes（减少73%）。子进程控制台尾部256KiB、警告/错误尾部64KiB；所有逐步标量、配置、哈希、完整预测与指标保留。权重及恢复状态仍存results之外；旧结果及其哈希清单未改写。

定向回归原35项全部通过，后续新增SFT导出恢复及四rank G32/G16采样测试，最终新suite12项全部通过（合计37个不同测试）。真实tiny Qwen3验证G32生成、rank21命中奖励、有限非零梯度；原调度前缀、提前停止及断点恢复后模型/reference/scheduler一致；模拟GPU验证六组串行、失败续跑、配置篡改拒绝、26次gzip评估和实际tar打包。首次G32随机fixture恰好全错、policy=reference导致零梯度，调整为实际beam中的rank21目标后通过，生产奖励未改。四个新Python文件解析、README23段Bash语法、dry-run零写入及whitespace通过；原fixed源run14个核心文件哈希不变，旧short/ABC脚本与Enhanced仓库未改。检查记录在 `results/six_suite_validation_20260912/`。本机无CUDA及完整模型权重，未启动真实四卡训练。

## 2026-09-12 — 结合 ABC350 重新审计 Enhanced 的 RL 正收益

只读重审 Enhanced `2fd7042`。普通 Ranking 已贡献 HR50 +0.7487pp，First-Diff 另加 +0.4454pp。按作者最终3750分桶的舍入值推算：8494/16163=52.55% 冷启动样本贡献总体 HR50 +1.3821pp，高于总增益 +1.1941pp，已见商品合计贡献约 −0.1880pp；这是作者统计的代数推算，原始预测未提供。当前 Office valid 同定义冷启动36/4866=.740%。整体商品等权收益并未显著，不能解释为所有商品普遍改善。

仍未被 ABC 控制的关键差异：Enhanced 三任务各1万（1:1:1），Office 38924/6918/10000；Enhanced 候选batch128=8输入组/更新，Office1024=64组；优化器及实际模型加载精度路径也不同。Enhanced350/750/3750消费2800/6000/30000组，Office350消费22400组，不能按更新步数直接判断训练长短。Enhanced 显式BF16加载的对象没有传给训练器，后者按路径重新加载且未指定dtype；本机默认加载实验得到FP32，但不能据此认定其远端完整训练精度或因果收益。

实际奖励函数17种命中／全错情况与当前标量奖励一致；First-Diff 检查确认只有组内受监督得分有差异时才提供额外信号，同位置全部同错仍归零。已有Office valid预测重算HR50：SFT .314838、A350 .303535、B350 .278668、C350 .297370，均退化，不能归因于Top10/Top50口径。修改下一步建议优先级：先核验数值路径，再分别控制batch或任务权重；LR5e-6保留为候选诊断，未执行任何新实验。完整报告、计算脚本、PASS日志、19份Enhanced源码快照及SHA256在 `results/qwen3_h50_abc350/analysis_20260912/enhanced_reaudit/`。训练代码未改。

## 2026-09-12 — A/B/C 350步结果：beta缓解退化，尚未超过SFT

收到 `results/qwen3_h50_abc350` 完整三组结果。最终预算为每组350步，保存/验证50、100、175、350及SFT；同一Office SFT step378，LR1e-5、micro16×累积16×4卡、候选batch1024、完整1746步调度/warmup53。A beta=.04/sample=True，B beta=.001/sample=False，C beta=.04/sample=False。三组均完成。

| 配置 | 175步HR@10 | 350步HR@10 | 350步NDCG@10 | 350步相对SFT净Top10命中 |
|---|---:|---:|---:|---:|
| SFT | 0.232635 | 0.232635 | 0.188749 | 0 |
| 原baseline（历史对照） | 0.211673 | 0.211262 | 0.170102 | -104 |
| A | 0.226469 | 0.216605 | 0.178366 | -78 |
| B | 0.210234 | 0.208590 | 0.168389 | -117 |
| C | 0.225442 | 0.217633 | 0.177346 | -73 |

18组验证、共87588条预测重算/标签/候选核验通过，492项非test副本hash及74项源码核验通过，四份SFT预测逐字节相同。按1795名用户做2000次配对bootstrap：A/B/C350相对SFT的HR10、NDCG10区间均为负；C相对B显著改善（本次未校正区间），C相对A差异区间包含0。A/C从175到350继续下降。所有12个已测RL快照的四项排名指标均低于SFT；三组最好的已测RL点均为50步，不能声称未测步数的全局最优。

前350步KL估计中位数A/C约.21、B/原baseline约2.6–2.8；最大值A8832/B57346/C111/原baseline993169。C仍有日志grad_norm121.74尖峰，A为16.75，不能称C全面稳定；日志KL不是全词表精确KL。reference同步间隔512，本次早期退化发生于第一次同步前。DeepSpeed析构错误发生于训练结束，后续完整验证产物通过。结果仅支持beta缓解当前退化，未显示关闭采样的排名收益或可靠组合增益；历史对照、单seed及多重比较限制见报告。

建议保留SFT，暂停B及不改配置的长跑；若继续单因素验证，可从A仅将LR降至5e-6并观察早期快照，仍属待验证假设。本次只分析既有验证结果，不改训练代码、不启动实验、不使用test预测。完整报告、可复现脚本、配对区间、切片和PNG/PDF曲线位于 `results/qwen3_h50_abc350/analysis_20260912/`。

## 2026-09-11 — A/B/C 串行175步初筛入口

用户授权三组短跑：A beta=.04/sample=True，B beta=.001/sample=False，C beta=.04/sample=False；LR均1e-5。新增 `scripts/run_rl_abc.py`，默认每组175更新、保存50/100/175，原完整2epoch cosine/warmup调度不变。选择175以覆盖已观察到的113～116步尖峰和baseline首个下降评估点，短跑只作早期筛查，不代表最终效果。

在独立短跑入口增加beta/候选采样显式选项，默认仍 .001/True；通过trainer实例的GenerationConfig设置候选模式，不修改共享ReReTrainer或原 `reproduction/*.py`。记录实际beta/采样/beam数/temperature；HF `use_model_defaults=False`仍有效。关闭采样时仍为G16完整beam候选，适用于训练及原G16验证奖励；后续排名评估统一确定性beam50。任务/奖励/优化器/reference同步/有效batch1024/attention均保持。旧短跑脚本指纹不自动迁移。

三组先导入并校验相同SFT权重、step、数据和长度，再A训练→验证→B训练→验证→C训练→验证，使用同四卡。支持单阶段、同命令重跑及checkpoint恢复，错误停止后续组。完成的结果逐组实体复制入套件目录，含日志/配置/源码/预测/指标/权重路径，自动跨组汇总，一条package_results命令即可打包，无软链接和模型权重。数据校验后复用，不重训SFT，不运行测试集或更改原选模。

本地定向CPU回归 **32/32 PASS**：包括三种实际tiny Qwen3/ReReTrainer的采样开关未被Qwen默认覆盖、两个输入各16个唯一合法SID候选/标签分组、有限且非零梯度、原调度前缀/停止/恢复，以及串行顺序、源run不变、失败后不启动C、数据/SFT不一致拒绝、普通文件副本、重复运行与参数变更拒绝。额外端到端脚本测试模拟GPU子进程，实际执行准备/训练记录收尾/逐样本指标核验/三组汇总及tar打包；完成后重跑无GPU调用，打包内含三组预测、无软链接和权重。Python解析、README的21段Bash语法、git whitespace检查通过；核心文件相对HEAD未修改。验证日志在 `results/abc_script_validation_20260911/checks/tests.log`。真实GPU训练、ZeRO2/paged AdamW与推荐指标尚未执行；本机无CUDA。

## 2026-09-11 — 对照 MiniOneRec-Enhanced 的 RL 增益记录

只读审阅本地Enhanced commit2fd7042：脚本确实使用Qwen3-0.6B，Amazon23 Industrial，自建Balanced RQ-KMeans SID3/4层、原始历史10。作者聚合表记录SFT→Ranking的测试HR10 .060447→.066262、NDCG10 .048873→.052984；First-Diff另有增益。当前有效配方不同：beta .04、确定性beam16、候选batch128、三任务各10000、1epoch3750、adamw_torch。MONITORING_LOG记载早期beta.001也出现KL尖峰，之后改beta和生成方式；缺少原始日志/权重/run源码快照，无法独立复核历史配置或归因单一修改。

Enhanced报告测试冷启动8494/16163=52.55%，收益偏冷启动、部分中高频损失。按训练history+target集合统一定义直接统计本run Office：valid36/4866=.740%、test58/4866=1.192%；只描述既有数据分布，没有新增测试评估/选模。两套官方SID均3层，Enhanced的第4层结束冲突不适用。源码还保留训练/验证history2target合并覆盖风险，未获其数据不能量化；文档的surrogate ratio爆炸因果说明也与活跃损失不符。

完整差异、证据边界和单变量验证建议在 `results/qwen3_h50_seed42_rl16_fixed/Office_Products/diagnostics/enhanced_comparison_20260911/`，含14份源码副本、哈希、聚合指标与可重复统计脚本。本轮没有更改训练实现/参数、没有启动GPU实验。

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
