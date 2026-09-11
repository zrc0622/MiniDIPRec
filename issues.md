# 1
@fukioston
fukioston
opened on Jan 26
Currently, the reward during reinforcement learning fluctuates significantly, but the evaluation performance shows improvement compared to SFT. Is this behavior normal?

Image
Activity
SuperTom99 commented on Jan 30
@SuperTom99
SuperTom99
on Jan 30
hi，do you observe lots of invalid tokens during RL？what base model do you use？

fukioston commented on Jan 30
@fukioston
fukioston
on Jan 30
Author
hi，do you observe lots of invalid tokens during RL？what base model do you use？
I don’t observe this situation,sorry

fukioston commented on Jan 30
@fukioston
fukioston
on Jan 30
Author
hi，do you observe lots of invalid tokens during RL？what base model do you use？

I use qwen3-1.7b-base n qwen3-0.6b-base

cocoshe commented on Jan 31
@cocoshe
cocoshe
on Jan 31
Contributor
@AkaliKong Can you share the RL training loss wandb log?

# 2
@rijinjjjj
rijinjjjj
opened on Jan 3
您好，本人使用Qwen2.5-3B-Instruct进行复现，使用单块A800进行sft，最终结果和论文内有较大差距（未进行RL，理由是之前有一次复现，进行RL后指标更低了）：
Calculating metrics...
2589it [00:00, 228169.71it/s]
50
[1, 3, 5, 10, 20, 50]
NDCG: [0.05446118 0.06694503 0.07276139 0.08094718 0.08602484 0.09111838]
HR [0.05446118 0.07609115 0.09038239 0.11587486 0.13595983 0.1614523 ]
30119
Completed processing for category: Industrial_and_Scientific
我想知道性能达不到论文报告可能是什么原因导致的？我的数据处理使用了amazon18_data_process.sh，配置如下：
python amazon18_data_process.py
--dataset Industrial_and_Scientific
--user_k 5
--item_k 5
--st_year 1996
--st_month 10
--ed_year 2018
--ed_month 11
--metadata_file ./metadata/meta_Industrial_and_Scientific.json
--reviews_file ./reviews/Industrial_and_Scientific.json
--output_path ./Amazon18
amazon_text2emb.sh使用Qwen3-Embedding-4B进行编码，--num_processes为1，rqvae训练使用rqvae.sh，在单卡上进行训练（这里是否需要使用rqkmeans等才能达到论文效果，还是您论文内的结果就仅使用了rqvae.sh？）
安装requirements.txt时没有安装nvidia-cudnn-cu12等，会不会对实验结果有影响？

Activity
AkaliKong commented on Jan 4
@AkaliKong
AkaliKong
on Jan 4
Owner
Hello, thank you for your interest in our paper.

In your evaluation results, the 30119 indicates that there are still a large number of invalid items being generated, and constrained decoding has failed. We suspect this may be due to version issues with certain related dependencies in the Transformer library, and we are still investigating the specific cause. Until we provide a universal solution, you can temporarily switch from the instruct model to the base model to bypass this issue.

rijinjjjj commented on Jan 7
@rijinjjjj
rijinjjjj
on Jan 7 via email
Author
您好，很抱歉我没有保存之前的result.json，但是之前翻阅过该文件，对比你您提供在huggingface上的模型，印象比较深的是较多"predict"的SID并未达到50个（有的甚至不足25个）。对于非SID的预测结果，使用Qwen-2.5-instruct系列模型并未出现。




rijin
***@***.***
Luo-Jiaming commented on Jan 8
@Luo-Jiaming
Luo-Jiaming
on Jan 8
We encountered the same issue with Qwen2.5-1.5B-Instruct: constrained beam search was not working as expected. After some investigation, we found that the parameters do_sample, temperature, top_k, and top_p were not explicitly set in the GenerationConfig. As a result, the model loaded its default configuration from file Qwen2.5-1.5B-Instruct/generation_config.json, where do_sample=True by default. This setting takes precedence over beam search and effectively disables it.

The issue was resolved by explicitly setting these parameters (i.e., do_sample=False, top_p=None, top_k=None, temperature=1.0) when initializing the GenerationConfig. By the way, this is clearly a problem with the transformers library, so downgrading might help.

Reacted by Jiyuan Yang, Jiawei Feng, Strace, OHHHHHH, yydxlv and Fang Yu
Reacted by Jiyuan Yang, Jiawei Feng, Strace and OHHHHHH
lenbow commented on Jan 12
@lenbow
lenbow
on Jan 12
We encountered the same issue with Qwen2.5-1.5B-Instruct: constrained beam search was not working as expected. After some investigation, we found that the parameters do_sample, temperature, top_k, and top_p were not explicitly set in the GenerationConfig. As a result, the model loaded its default configuration from file Qwen2.5-1.5B-Instruct/generation_config.json, where do_sample=True by default. This setting takes precedence over beam search and effectively disables it.

The issue was resolved by explicitly setting these parameters (i.e., do_sample=False, top_p=None, top_k=None, temperature=1.0) when initializing the GenerationConfig. By the way, this is clearly a problem with the transformers library, so downgrading might help.

Thank you for your sharing. We tried your method, and it indeed improved, but it hasn't reached the expected effect yet. We applied your method during the verification stage. Do we need to modify the configuration from the initial stage? Or are there other configurations needed?

Luo-Jiaming commented on Jan 13
@Luo-Jiaming
Luo-Jiaming
on Jan 13
Hi @lenbow,

We’ve added explicit parameter settings to the GenerationConfig upon declaration in both evaluate.py (used for evaluation) and minionerec_trainer.py (used for RL). This change helps prevent the model from generating too many empty SIDs, which previously hindered the effectiveness of RL. As a result, the RL component now yields positive improvements.

While our reproduced results are still slightly below those reported in the paper, the gap has been significantly reduced and now falls within an acceptable range.

Reacted by CharlieGuo, Strace, Jiawei Feng, OHHHHHH and user683
SuperTom99 commented on Jan 30
@SuperTom99
SuperTom99
on Jan 30
@Luo-Jiaming RL is not allow to set do_sample=False
if you do so，the model will stop exploring

SuperTom99 commented on Jan 30
@SuperTom99
SuperTom99
on Jan 30
@lenbow @rijinjjjj 我也遇到了相同的问题，请问解决了吗


cocoshe
mentioned this on Jan 31
Fix: invalid tokens #57

jonathan-politzki
mentioned this 1w ago
Fix evaluate.sh passing arguments evaluate.py does not accept #81

jonathan-politzki
added a commit that references this issue 1w ago
Add Stage A: full-scale SFT reproduction + do_sample trap analysis

db3e4ed

jonathan-politzki
mentioned this 1w ago
Fix constrained decoding silently falling back to sampling (#49, #28, #44) #82
jonathan-politzki commented last week
@jonathan-politzki
jonathan-politzki
1w ago
Contributor
Independent reproduction data point, plus a correction to the do_sample workaround discussed above.

SFT reproduces on a single A100
Ran the repo's own sft.py → evaluate.py → calc.py on Industrial_and_Scientific with paper hyperparameters (Qwen2.5-1.5B base, batch 1024 via micro-batch 8 × grad-accum 128 on one GPU, lr 3e-4, early stopping — stopped at epoch 6.5):

K:     [1, 3, 5, 10, 20, 50]
NDCG:  [0.0660, 0.0810, 0.0859, 0.0952, 0.1037, 0.1154]
HR:    [0.0660, 0.0918, 0.1039, 0.1328, 0.1666, 0.2255]
invalid items (CC): 0
HR@10 0.1328 / NDCG@10 0.0952, within ~3% of the reproduction in #35 (0.1370 / 0.0969) and in line with the paper's SFT stage. Crucially CC = 0 — versus the 30119 reported at the top of this thread. So the method and the constraint machinery are fine; the failures here are environmental.

Correction: setting do_sample=False on the GenerationConfig does NOT fix this on transformers >= 4.50
The diagnosis in this thread is right about the cause but the suggested fix is insufficient on current transformers. _prepare_generation_config overwrites any value in a passed GenerationConfig that equals the global default with the model's own default:

if (custom_gen_config_value == global_default_value
        and model_gen_config_value != global_default_value):
    setattr(generation_config, key, model_gen_config_value)
do_sample=False is the global default, so it gets overwritten too. Measured on transformers 4.57.1 against a model whose generation_config sets do_sample: true:

configuration	resulting do_sample
evaluate.py today (unset)	True — bug
GenerationConfig(do_sample=False)	True — still broken
generate(..., do_sample=False) (kwarg)	False — fixed
generate(..., use_model_defaults=False)	False — 4.50+ only
Only generate() kwargs take priority over model defaults. Submitted as #82.

This also explains why base models work (they don't set do_sample in their generation_config, so nothing is overridden) and why downgrading transformers helped (the merge behaviour didn't exist before 4.50).

Other environment blockers hit on a clean checkout
evaluate.sh passes --temperature / --guidance_scale, which evaluate.py doesn't accept → fire aborts before evaluating anything. The documented one-click eval path fails as shipped (Fix evaluate.sh passing arguments evaluate.py does not accept #81).
POT is imported by rq/rqkmeans_faiss.py but missing from requirements.txt (Add missing POT dependency to requirements.txt #80).
transformers 5.x cannot load Qwen2 in this setup; 4.57.x (the repo pin) works.
On stock cloud images, pip install 'numpy<2' can resolve to 1.21.5, which predates ndarray.__class_getitem__ and breaks the transformers data collator — pin numpy==1.26.4.
Full run log, config and metrics: https://github.com/jonathan-politzki/MiniOneRec/blob/experiment/item-text-enrichment/lab-notes/LOG.md

Hope this is useful for others trying to reproduce.


AkaliKong
added 2 commits that reference this issue 1d ago
Fix constrained decoding silently falling back to sampling

9fded8c
Merge pull request #82 from jonathan-politzki/fix/constrained-decodin…

0518dae

# 3
@ashleys0
ashleys0
opened on Dec 9, 2025
Hi thank you to the authors for your wonderful work!

I'm wondering if anyone has similar issues -- I'm using default configs, except a few things like batch size because I'm using 1.5B with 4 GPUs.
The SFT step seems to be working fine as expected, but I'm not sure what's happening in the RL step.

Test ndcg goes down instead of improving.
val ndcg improves, but only a little bit.
SFT: Test ndcg@10: 0.0961 | Val ndcg@10: 0.0941
RL: Test ndcg@10: 0.0896 | Val ndcg@10: 0.1001

I would love to hear any advice or thoughts.

Thanks,
Ashley

Activity
jiaweif3ng commented on Dec 12, 2025
@jiaweif3ng
jiaweif3ng
on Dec 12, 2025
Contributor
Hi thank you to the authors for your wonderful work!

I'm wondering if anyone has similar issues -- I'm using default configs, except a few things like batch size because I'm using 1.5B with 4 GPUs. The SFT step seems to be working fine as expected, but I'm not sure what's happening in the RL step.

Test ndcg goes down instead of improving.
val ndcg improves, but only a little bit.
SFT: Test ndcg@10: 0.0961 | Val ndcg@10: 0.0941 RL: Test ndcg@10: 0.0896 | Val ndcg@10: 0.1001

I would love to hear any advice or thoughts.

Thanks, Ashley

This issue might be related to dependency version mismatches. In some cases, Qwen2.5-Instruct may produce invalid SIDs during inference.
I recommend trying the Qwen2.5 model to check whether the problem still occurs.

Reacted by Ashley Shin
newwindhxx commented on Dec 18, 2025
@newwindhxx
newwindhxx
on Dec 18, 2025
Hi thank you to the authors for your wonderful work!

I'm wondering if anyone has similar issues -- I'm using default configs, except a few things like batch size because I'm using 1.5B with 4 GPUs. The SFT step seems to be working fine as expected, but I'm not sure what's happening in the RL step.

Test ndcg goes down instead of improving.
val ndcg improves, but only a little bit.
SFT: Test ndcg@10: 0.0961 | Val ndcg@10: 0.0941 RL: Test ndcg@10: 0.0896 | Val ndcg@10: 0.1001

I would love to hear any advice or thoughts.

Thanks, Ashley

I meet the same issues, have you found the way to solve it?

ashleys0 commented on Dec 20, 2025
@ashleys0
ashleys0
on Dec 20, 2025
Author
@newwindhxx I changed some hyperparameters, mostly beta and now it's 0.0996, which isn't as good as it should be but it is an improvement vs before.


jonathan-politzki
mentioned this 1w ago
Fix constrained decoding silently falling back to sampling (#49, #28, #44) #82

AkaliKong
added a commit that references this issue 1d ago
Merge pull request #82 from jonathan-politzki/fix/constrained-decodin…

0518dae

# 4
@maobenz
maobenz
opened on Dec 2, 2025
Firstly, thank you for the great work!

We have reproduced the experiments using both SFT and RL, following the exact hyperparameters provided in the paper.
The experiments were conducted on a single node with 8× H100 GPUs.

After SFT, our results are:

Top-K: [1, 3, 5, 10, 20, 50]
NDCG: [0.0671, 0.0804, 0.0865, 0.0969, 0.1045, 0.1142]
HR: [0.0671, 0.0902, 0.1050, 0.1370, 0.1672, 0.2160]

After RL, our results are:

Top-K: [1, 3, 5, 10, 20, 50]
NDCG: [0.0743, 0.0883, 0.0935, 0.1001, 0.1052, 0.1100]
HR: [0.0743, 0.0984, 0.1107, 0.1313, 0.1516, 0.1756]

However, these results are consistently lower than those reported in the paper.
Since no hyperparameters were changed on our side, we are wondering whether:

There are any additional implementation details that are not explicitly mentioned in the paper?

Is there any recommended training trick or configuration that may significantly affect performance?

Are the reported results averaged over multiple runs or using a different random seed?

Any advice on improving reproducibility?

Any insights or suggestions would be greatly appreciated!

Activity
AkaliKong commented on Dec 2, 2025
@AkaliKong
AkaliKong
on Dec 2, 2025
Owner
Hi, thanks for your interest in our work!

MiniOneRec is fully reproducible in our environment.
Regarding the issue you mentioned, we found that differences in dependency versions can sometimes lead to noticeable variations in performance. For example, earlier versions of trl caused a significant drop in model accuracy.

Since my compute resources are currently limited, could you please first make sure your setup aligns with the latest version of our code?
I will re-verify the full reproducibility on my side within the next few days.

If you have any further observations or updates, feel free to reach out at any time.

Thanks again for your patience and support!

maobenz commented on Dec 2, 2025
@maobenz
maobenz
on Dec 2, 2025
Author
We initially suspected that RL might be the cause of the issue. Could you confirm whether our SFT results are within the expected range? We compared our results with the figure in your paper, although it only reports HR@10. Based on this, we believe our results are comparable.

AkaliKong commented on Dec 2, 2025
@AkaliKong
AkaliKong
on Dec 2, 2025
Owner
One version of our SFT-only model achieves the following metrics:
Top-K: [1, 3, 5, 10, 20, 50]
NDCG: [0.0655, 0.0828, 0.0907, 0.0999, 0.1095, 0.1230]
HR: [0.0655, 0.0953, 0.1145, 0.1430, 0.1811, 0.2491]

We then observed a clear performance improvement after applying reinforcement learning.
Hope this information is helpful to you!

Reacted by Ashley Shin and mind2026
maobenz commented on Dec 2, 2025
@maobenz
maobenz
on Dec 2, 2025
Author
Thanks. But I have one more question: the result from the Industrial_and_Scientific dataset? In Figure 1 of the paper, the HR@10 after SFT is below 0.14, about 0.135, but your result shows 0.143. Is there a mistake somewhere?

AkaliKong commented on Dec 2, 2025
@AkaliKong
AkaliKong
on Dec 2, 2025
Owner
Hello, after the paper was released, we made several updates to the code, including new RQ methods and fixes for some minor bugs, which led to a certain degree of performance improvement.

AkaliKong commented on Dec 2, 2025
@AkaliKong
AkaliKong
on Dec 2, 2025
Owner
My collaborators and I verify that we can successfully reproduce MiniOneRec in our respective environments. We believe the issues you are currently facing may be related to differences in dependency versions.

Wish you smooth experiments, and we look forward to your further feedback!

ashleys0 commented on Dec 9, 2025
@ashleys0
ashleys0
on Dec 9, 2025
@maobenz hi were you able to figure this out?

I'm using 1.5B due to smaller compute, and after SFT I get test ndcg@10 of 0.0961, which seems to be in line with the original 7B results. But after 1 epoch of RL, it drops to 0.0896 (test ndcg drops but val ndcg does improve by 0.006, which is still not that much).

newwindhxx commented on Dec 18, 2025
@newwindhxx
newwindhxx
on Dec 18, 2025
One version of our SFT-only model achieves the following metrics: Top-K: [1, 3, 5, 10, 20, 50] NDCG: [0.0655, 0.0828, 0.0907, 0.0999, 0.1095, 0.1230] HR: [0.0655, 0.0953, 0.1145, 0.1430, 0.1811, 0.2491]

We then observed a clear performance improvement after applying reinforcement learning. Hope this information is helpful to you!

Is this result base Qwen-2.5-7B?


jonathan-politzki
mentioned this 1w ago
Fix evaluate.sh passing arguments evaluate.py does not accept #81

jonathan-politzki
added a commit that references this issue 1w ago
Add Stage A: full-scale SFT reproduction + do_sample trap analysis

db3e4ed

jonathan-politzki
mentioned this 1w ago
Fix constrained decoding silently falling back to sampling (#49, #28, #44) #82
关于文章结果复现的问题 #49
# 5
@cywuuuu
cywuuuu
opened on Feb 1
您好，首先非常感谢作者开源了这份工作，对我很有启发和帮助。我目前在使用 SFT（监督微调）完成基线模型训练后，进一步进行强化学习（RL）训练，但发现 RL 后的模型性能相比基线模型有所下降。

问题：
RL 训练后模型性能下降的可能原因是什么？以及如何优化训练配置以提升 RL 模型的性能。如有需要，我们可以提供更多详细信息，例如相关数据和SFT后的权重。还希望您能够帮忙，谢谢！

数据集信息
数据集：Amazon23/Industrial_and_Scientific

训练配置信息

SFT 训练配置
基础模型：Qwen3-1.7B
训练参数：
batch_size: 128
micro_batch_size: 8
num_epochs: 10
learning_rate: 3e-4
cutoff_len: 512
temperature: 0.07
seed: 42
使用 flash attention: True
RL 训练配置
训练脚本： scripts/rl.sh
训练参数：
model_path: [SFT 输出目录]
train_batch_size: 32
eval_batch_size: 128
num_train_epochs: 2
gradient_accumulation_steps: 2
learning_rate: 1e-5
reward_type: ranking
num_generations: 2
beam_search: True
temperature: 1.0
beta: 1e-3
评估配置
评估脚本： scripts/evaluate_baseline.sh
评估参数：
batch_size: 4
num_beams: 20
max_new_tokens: 256
temperature: 1.0
guidance_scale: 1.0
length_penalty: 0.0
性能对比数据
NDCG 指标
模型	@1	@3	@5	@10	@20	@50
SFT 基线	0.0255	0.0313	0.0330	0.0355	0.0387	0.0427
RL checkpoint-423	0.0257	0.0318	0.0335	0.0359	0.0390	0.0428
RL checkpoint-705	0.0250	0.0309	0.0325	0.0351	0.0379	0.0417
RL checkpoint-846	0.0248	0.0305	0.0323	0.0348	0.0377	0.0415
HR 指标
模型	@1	@3	@5	@10	@20	@50
SFT 基线	0.0255	0.0355	0.0395	0.0474	0.0602	0.0806
RL checkpoint-423	0.0257	0.0361	0.0403	0.0478	0.0600	0.0792
RL checkpoint-705	0.0250	0.0351	0.0390	0.0473	0.0583	0.0777
RL checkpoint-846	0.0248	0.0344	0.0389	0.0467	0.0580	0.0771
Activity
cywuuuu commented on Feb 1
@cywuuuu
cywuuuu
on Feb 1
Author
此外，还请问能否利用现有的 sasrec.py直接复现得到论文中的sasrec等baseline的结果？具体参数应该如何设置。希望能够有一个较快复现其他baseline的方式以便后续进一步的研究，还请作者赐教。

xym0601 commented on Feb 10
@xym0601
xym0601
on Feb 10
num_generations: 2，太小了

huoyushi commented on Feb 12
@huoyushi
huoyushi
on Feb 12
你好 有遇到过，rl过程中 loss 全是 0 的情况吗？
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.2727272727272729e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0}
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.525252525252525e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0}

LNH2002 commented on Feb 22
@LNH2002
LNH2002
on Feb 22
你好 有遇到过，rl过程中 loss 全是 0 的情况吗？ {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.2727272727272729e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0} {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.525252525252525e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0}

我也遇到了同样的情况，但是我的grad norm并不为0，模型不收敛，使用的qwen2.5-1.5B-instruct模型，但是hr@N和NDCG@N在训练过程中指标均有上升（使用跑了0.2个epoch的checkpoint去验证），目前还未解决。在复现实验的时候，使用了amazon23的数据，在sft阶段参数设置有如下修改

torchrun --nproc_per_node 4

batch_size 1024

micro_batch_size 32
rl阶段将num_generations从16改为了8，其他保持不变

Image
已解决，在该模型中reward值以及loss值无法代表模型训练程度，loss和reward发生波动但是HR@N和NDCG@N增加属正常现象。并且模型启动阶段item命中率较低，而内部的loss算法决定了命中率较低时loss值趋近于0，可以通过增加num generations的数值来增大命中率缓解该现象，另外已通过实验证明使用add_gt的方式无法提高模型指标。

leonli3238 commented on Feb 26
@leonli3238
leonli3238
on Feb 26
你好 有遇到过，rl过程中 loss 全是 0 的情况吗？ {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.2727272727272729e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0} {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.525252525252525e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0}

我也遇到了同样的情况，但是我的grad norm并不为0，模型不收敛，使用的qwen2.5-1.5B-instruct模型，但是hr@N和NDCG@N在训练过程中指标均有上升（使用跑了0.2个epoch的checkpoint去验证），目前还未解决。在复现实验的时候，使用了amazon23的数据，在sft阶段参数设置有如下修改

torchrun --nproc_per_node 4
batch_size 1024
micro_batch_size 32
rl阶段将num_generations从16改为了8，其他保持不变
Image 已解决，在该模型中reward值以及loss值无法代表模型训练程度，loss和reward发生波动但是HR@N和NDCG@N增加属正常现象。并且模型启动阶段item命中率较低，而内部的loss算法决定了命中率较低时loss值趋近于0，可以通过增加num generations的数值来增大命中率缓解该现象，另外已通过实验证明使用add_gt的方式无法提高模型指标。
请问下你rl微调后相比于sft微调后的效果提升得多吗？

LNH2002 commented on Feb 26
@LNH2002
LNH2002
on Feb 26
你好 有遇到过，rl过程中 loss 全是 0 的情况吗？ {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.2727272727272729e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0} {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.525252525252525e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0}

我也遇到了同样的情况，但是我的grad norm并不为0，模型不收敛，使用的qwen2.5-1.5B-instruct模型，但是hr@N和NDCG@N在训练过程中指标均有上升（使用跑了0.4个epoch的checkpoint去验证），目前还未解决。在复现实验的时候，使用了amazon23的数据，在sft阶段参数设置有如下修改

torchrun --nproc_per_node 4
batch_size 1024
micro_batch_size 32
rl阶段将num_generations从16改为了8，其他保持不变
Image 已解决，在该模型中reward值以及loss值无法代表模型训练程度，loss和reward发生波动但是HR@N和NDCG@N增加属正常现象。并且模型启动阶段item命中率较低，而内部的loss算法决定了命中率较低时loss值趋近于0，可以通过增加num generations的数值来增大命中率缓解该现象，另外已通过实验证明使用add_gt的方式无法提高模型指标。
请问下你rl微调后相比于sft微调后的效果提升得多吗？

sft阶段的loss为
nbeam: 50
N: [1, 3, 5, 10, 20, 50]
NDCG@N: [0.02815071 0.03382418 0.03551997 0.03880099 0.04205148 0.047096 ]
HR@N [0.02815071 0.03792613 0.0420714 0.05234177 0.06527254 0.09076285]
rl阶段（checkpoint-493，0.4epoch，之前数据记录有误）
50
[1, 3, 5, 10, 20, 50]
NDCG: [0.03501825 0.04100547 0.04299932 0.04555554 0.04858764 0.0531999 ]
HR [0.03501825 0.04522675 0.05011446 0.05809565 0.07016024 0.09367073]
0

leonli3238 commented on Feb 26
@leonli3238
leonli3238
on Feb 26
你好 有遇到过，rl过程中 loss 全是 0 的情况吗？ {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.2727272727272729e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0} {'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 2.525252525252525e-07, 'rewards/rule_reward': 0.0, 'rewards/ndcg_rule_reward': 0.0, 'reward': 0.0, 'reward_std': 0.0, 'categorical_diversity': 0.0625, 'token_diversity': 0.0, 'completion_length': 1.0, 'kl': 0.0, 'epoch': 0.0}

我也遇到了同样的情况，但是我的grad norm并不为0，模型不收敛，使用的qwen2.5-1.5B-instruct模型，但是hr@N和NDCG@N在训练过程中指标均有上升（使用跑了0.4个epoch的checkpoint去验证），目前还未解决。在复现实验的时候，使用了amazon23的数据，在sft阶段参数设置有如下修改

torchrun --nproc_per_node 4
batch_size 1024
micro_batch_size 32
rl阶段将num_generations从16改为了8，其他保持不变
Image 已解决，在该模型中reward值以及loss值无法代表模型训练程度，loss和reward发生波动但是HR@N和NDCG@N增加属正常现象。并且模型启动阶段item命中率较低，而内部的loss算法决定了命中率较低时loss值趋近于0，可以通过增加num generations的数值来增大命中率缓解该现象，另外已通过实验证明使用add_gt的方式无法提高模型指标。
请问下你rl微调后相比于sft微调后的效果提升得多吗？

sft阶段的loss为 nbeam: 50 N: [1, 3, 5, 10, 20, 50] NDCG@N: [0.02815071 0.03382418 0.03551997 0.03880099 0.04205148 0.047096 ] HR@N [0.02815071 0.03792613 0.0420714 0.05234177 0.06527254 0.09076285] rl阶段（checkpoint-493，0.4epoch，之前数据记录有误） 50 [1, 3, 5, 10, 20, 50] NDCG: [0.03501825 0.04100547 0.04299932 0.04555554 0.04858764 0.0531999 ] HR [0.03501825 0.04522675 0.05011446 0.05809565 0.07016024 0.09367073] 0

感谢，你用的是哪个数据集啊

LNH2002 commented on Feb 26
@LNH2002
LNH2002
on Feb 26 via email
amazon review 23，并没有用原始的amazon review 18

Lan Li ***@***.***> 于 2026年2月26日周四 21:38写道：
…
Reacted by leonli3238
leonli3238 commented on Mar 2
@leonli3238
leonli3238
on Mar 2
amazon review 23，并没有用原始的amazon review 18

Lan Li @.***> 于 2026年2月26日周四 21:38写道：
…

bro，想问下你有用过rqkmeans生成的SID来进行实验吗，我用rqvae生成的SID进行实验能达到一定效果，但是与作者的结果还是有一定的差距，之后使用rqkmeans生成的SID运行后效果很差，不知道是哪里出了问题

LNH2002 commented on Mar 2
@LNH2002
LNH2002
on Mar 2
amazon review 23，并没有用原始的amazon review 18
Lan Li @.***> 于 2026年2月26日周四 21:38写道：
…

bro，想问下你有用过rqkmeans生成的SID来进行实验吗，我用rqvae生成的SID进行实验能达到一定效果，但是与作者的结果还是有一定的差距，之后使用rqkmeans生成的SID运行后效果很差，不知道是哪里出了问题

并未使用rqkmeans进行实验。我推测使用rqvae复现时候的实验差距是因为对齐问题导致的，在论文附录部分明确指出了sft对齐阶段会使用User Preference Summarization.进行对齐，但是直接从库中clone下来的代码并没有进行这样的对齐下，如果你后续进行了验证，不知道能否将结果分享一下

leonli3238 commented on Mar 3
@leonli3238
leonli3238
on Mar 3
amazon review 23，并没有用原始的amazon review 18
Lan Li @.***> 于 2026年2月26日周四 21:38写道：
…

bro，想问下你有用过rqkmeans生成的SID来进行实验吗，我用rqvae生成的SID进行实验能达到一定效果，但是与作者的结果还是有一定的差距，之后使用rqkmeans生成的SID运行后效果很差，不知道是哪里出了问题

并未使用rqkmeans进行实验。我推测使用rqvae复现时候的实验差距是因为对齐问题导致的，在论文附录部分明确指出了sft对齐阶段会使用User Preference Summarization.进行对齐，但是直接从库中clone下来的代码并没有进行这样的对齐下，如果你后续进行了验证，不知道能否将结果分享一下

okkk

LNH2002 commented on Mar 3
@LNH2002
LNH2002
on Mar 3
amazon review 23，并没有用原始的amazon review 18
Lan Li @.***> 于 2026年2月26日周四 21:38写道：
…

bro，想问下你有用过rqkmeans生成的SID来进行实验吗，我用rqvae生成的SID进行实验能达到一定效果，但是与作者的结果还是有一定的差距，之后使用rqkmeans生成的SID运行后效果很差，不知道是哪里出了问题

并未使用rqkmeans进行实验。我推测使用rqvae复现时候的实验差距是因为对齐问题导致的，在论文附录部分明确指出了sft对齐阶段会使用User Preference Summarization.进行对齐，但是直接从库中clone下来的代码并没有进行这样的对齐下，如果你后续进行了验证，不知道能否将结果分享一下

okkk

我在检查的时候发现除了User Preference Summarization外还有挺多不同的，在论文附录写着description2sid的对齐任务只在sft做，不在rl做，但是clone下来的代码是相反的，具体还有什么你可以论文对照代码分析以下。项目里对论文中的大部分东西应该都写对应代码了，但是有的被注释或者替换了，可以尝试修改这部分内容，我最近由于时间成本和经济成本暂时无法进行验证


cywuuuu
closed this as completedon Mar 24
# 6
@SFengU
SFengU
opened on Dec 3, 2025
Hi, thanks for the great work and code release!

I'm confused by the NDCG reward logic in ndcg_rule_reward:

if completion.strip("\n\"") == targets[i].strip("\n\""):
    flag = True
    lis.append(0.0)  # Correct → 0
else:
    lis.append(ndcg_rewards[i%num_generations])  # Wrong → negative

if (i+1)%num_generations == 0:
    if flag:
        rewards.extend(lis)  # Has correct: 0 + negatives
    else:
        rewards.extend([0.0] * repeat)  # All wrong: all zeros
Questions:
﻿
Reward direction: Why does correct=0.0, wrong=negative? And all wrong gets 0.0?
Looking forward to your clarification! Thanks a lot.

Activity
AkaliKong commented on Dec 3, 2025
@AkaliKong
AkaliKong
on Dec 3, 2025
Owner
Hi, thanks for your interest in our work.

Regarding your question about the NDCG reward: in our implementation, the HR and NDCG reward are provided together, but the NDCG reward serves more as a penalty term for negative samples.

Here is how it works:

If the model’s roll-out result hits the ground_truth:
hit reward = 1, ndcg reward = 0
(since the rule is already hit, NDCG does not apply additional ranking penalties)

If the roll-out result corresponds to a negative example:
hit reward = 0, ndcg reward < 0
(we use the negative NDCG value to penalize wrong predictions)

Because we use beam search for roll-out sampling, the generated results have a natural order.
Based on this ordering, negative samples appearing earlier receive stronger penalties, encouraging the model to push these items toward the end.

If none of the roll-out candidates hits any rule, we treat the sample as too difficult and assign zero reward to all candidates, so it does not influence training.

Feel free to reach out if you have more questions!

SFengU commented on Dec 4, 2025
@SFengU
SFengU
on Dec 4, 2025
Author
Thanks for the clear explanation! I understand now the design of using NDCG as a ranking penalty and handling difficult samples with zero rewards. Appreciate your help!


SFengU
closed this as completedon Dec 4, 2025
# 7