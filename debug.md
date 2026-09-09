# 四卡复现环境排查

在 **Linux GPU 服务器**上执行以下命令。先激活训练环境并进入仓库；路径、环境名和 GPU 编号按实际情况修改。下面默认使用 GPU `0,1,2,3`。

```bash
conda activate minidiprec
cd /data/zuorongchang/project/MiniDIPRec
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 1. 本次错误及原因

本次训练日志中的致命异常是：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/usr/local/cuda-12.2/bin/nvcc'
```

SFT 初始化 `Trainer` 时，Accelerate 导入已安装的 DeepSpeed；DeepSpeed 检查 CUDA Toolkit 版本时找不到 `nvcc`，因而退出。后面的 `ChildFailedError` 和其他进程收到 `SIGTERM` 是连带结果。这份日志没有显示显存不足或历史长度超限。

PyTorch 的 `cu124` wheel 提供 CUDA 运行库，不包含完整的 CUDA Toolkit 和 `nvcc`。换成官方 `requirements.txt` 也不能自动修正 Toolkit 路径或补齐 Toolkit。

## 2. 检查 Python、驱动和 CUDA 路径

```bash
command -v python
python --version
python -m pip --version
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_PATH=$CUDA_PATH"
command -v nvcc
nvidia-smi -i 0,1,2,3
```

`command -v nvcc` 没有输出，表示当前 `PATH` 中找不到它，不能据此认定整个服务器都没有安装 Toolkit。

`nvidia-smi` 显示的 `CUDA Version` 表示驱动支持的 CUDA 版本上限，不代表已安装的 Toolkit 版本。继续检查 PyTorch 实际检测到的路径：

```bash
python - <<'PY'
import os
import sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
print("Detected CUDA_HOME:", CUDA_HOME)
if CUDA_HOME:
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    print("Expected nvcc:", nvcc)
    print("nvcc is executable:", nvcc.is_file() and os.access(nvcc, os.X_OK))
for candidate in sorted(Path("/usr/local").glob("cuda*")):
    print("System CUDA candidate:", candidate,
          "nvcc exists:", (candidate / "bin" / "nvcc").is_file())
PY
```

本次使用的 PyTorch wheel 应显示 `PyTorch CUDA: 12.4`。指定四张卡后，应显示 `CUDA available: True` 和 `Visible GPU count: 4`。

## 3. 修正 Toolkit：以下两种情况选一种

### 已有完整 Toolkit

将变量指向实际安装目录。下面仅以 `/usr/local/cuda-12.4` 为例，先确认该路径确实有可执行的 `nvcc`：

```bash
test -x /usr/local/cuda-12.4/bin/nvcc
```

确认成功后执行；如果路径不同，请替换成实际路径：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" -V
```

### 没有完整 Toolkit

对于 PyTorch `cu124`，可在已激活的 conda 环境中安装 CUDA 12.4 Toolkit：

```bash
conda install -c nvidia/label/cuda-12.4.1 cuda-toolkit

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" -V
```

安装后应能输出 CUDA 12.4 的编译器版本。如果仍找不到 `nvcc`，先检查安装结果和实际目录，再继续。这里的 `export` 只影响当前终端；后续训练终端也需要相同设置。

## 4. 训练前验证

先验证四卡可见性、简单 CUDA 运算和 DeepSpeed 导入，避免重新构造所有训练样本后才遇到同一错误：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python - <<'PY'
import torch

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch 无法使用 CUDA"
assert torch.cuda.device_count() == 4, "应恰好可见四张 GPU"
for index in range(4):
    value = torch.ones(1, device=f"cuda:{index}")
    assert (value + 1).item() == 2
    print(f"Logical GPU {index}:", torch.cuda.get_device_name(index), "OK")

import deepspeed
print("DeepSpeed:", deepspeed.__version__, "import OK")
PY

python -m pip check
```

逻辑 GPU `0–3` 对应 `CUDA_VISIBLE_DEVICES` 中按顺序指定的四张物理 GPU。以上检查通过，只说明基本 CUDA 运算和 DeepSpeed 导入可用，尚未验证四进程 NCCL、ZeRO-2 和完整训练。

不要通过卸载 DeepSpeed 来绕过报错，后续官方 RL 流程需要它。

## 5. 修复后继续原实验

保持原来的模型、GPU、checkpoint 路径等参数，在设置好环境变量的同一终端运行：

```bash
bash scripts/reproduce.sh --run-name qwen3_h50_seed42 \
  --gpus 0,1,2,3 --model Qwen/Qwen3-0.6B \
  --resume
```

如果首次运行指定过 `--checkpoint-root` 或本地模型路径，重跑也要传入相同值。

- 同一 `run-name` 下，已有的历史数据经过原始文件和生成文件哈希校验后会复用，不会再次重建 50 条历史。
- 已有且匹配的 `lengths.json` 会复用，跳过 token 长度扫描。
- 训练提示构造和 tokenization 目前没有磁盘缓存，各进程重启后仍会执行。多条 `38924` 进度条是 Office 训练样本处理，不表示重复重建历史。
- `--resume` 跳过已完成阶段，并从未完成训练阶段的最新可用 checkpoint 恢复。本次在 Trainer 初始化时失败，尚未开始训练；若没有 checkpoint，则从头开始该训练阶段。

本文基于提供的训练日志定位错误；上述服务器 CUDA 修复和四卡检查尚未在本地执行，本地环境没有 CUDA GPU。
