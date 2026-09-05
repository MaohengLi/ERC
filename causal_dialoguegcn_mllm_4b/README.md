# Causal DialogueGCN-LLM with Multimodal Dual-System Routing

本目录是独立于 `causal_erc_repro_4b` 的改造工程，面向 IEMOCAP 六分类。它保留
Causal-ERC 的因果提示思想，用 DialogueGCN 替换 DialogueRNN，并将文本、音频、
视频和图上下文共同纳入峰终记忆与双系统路由。大语言模型固定为 4B 级
`Qwen/Qwen3-4B`，使用 NF4 QLoRA 适配 12GB 显存。

## 网络流程

```text
raw text/audio/video features
        │
        ├─ projection + 3-way cross-modal attention
        │                │
        │                └─ T/A/V auxiliary emotion logits
        │
        └─ speaker-aware past-only relational DialogueGCN ── G logits
                                     │
previous epoch / first inference pass T/A/V/G probabilities
                                     │
       multimodal Peak-End memory + confidence/disagreement
                                     │
             C1 → slow System 2 (language/history → emotion)
             C2 → fast System 1 (existing emotion → wording)
                                     │
          soft alpha routes Graph vs local multimodal state
                                     │
 audio/video/text/graph/route feature-token injection
                                     │
                 Qwen3-4B NF4 + LoRA → 6 emotions
```

默认图是过去向图：目标话语只聚合自己和过去 `graph_window` 条话语，防止未来话语
间接污染因果决策。`--bidirectional_graph` 只用于复现标准双向 DialogueGCN 或消融。

总损失为：

```text
L = L_final_CE + lambda_aux * mean(L_T, L_A, L_V, L_G) + lambda_hgr * L_Soft-HGR
```

按照原论文，`lambda_C1 < lambda_C2` 才选择 C1；C1 表示语言/历史上下文造成当前情绪，
C2 表示已有情绪状态导致当前话语表达。双系统判定只读取上一轮缓存的模型预测，不读取金标签。验证/测试采用严格两遍推理：
第一遍产生 T/A/V/G 预测，第二遍再构造峰终提示和路由。

## 文件说明

- `causal_router.py`：多模态峰终记忆、C1/C2 判定、System-1/System-2 软路由。
- `data_dual.py`：IEMOCAP 读取、提示构造和五个特征令牌定位。
- `model.py`：跨模态注意力、关系 DialogueGCN、Qwen 特征注入与联合损失。
- `runtime.py`：4-bit QLoRA、两遍评估、缓存切分、检查点保存/恢复。
- `train.py`：完整训练、上一轮预测缓存、早停与实时日志。
- `evaluate.py`：最佳检查点的 dev/test 评估。
- `tests/`：无未来泄漏、对话隔离、端到端梯度以及真实 Qwen4B 冒烟测试。

## 完整训练（PowerShell）

在 PowerShell 终端执行：

```powershell
Set-Location 'D:\C_Code\PycharmProjects\TFGCN_MLLM'
$env:PYTHONUNBUFFERED = '1'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = "causal_dialoguegcn_mllm_4b\outputs\full_train_$stamp.log"
python -u -m causal_dialoguegcn_mllm_4b.train `
  --data_path 'D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN\MultiModal_DialogGCN\save\data.pkl' `
  --model_name 'Qwen/Qwen3-4B' `
  --output_dir 'causal_dialoguegcn_mllm_4b\outputs\qwen3_4b_full' `
  --epochs 9 `
  --dialogue_batch_size 1 `
  --llm_micro_batch 2 `
  --grad_accum 4 `
  --max_length 256 `
  --hidden_dim 200 `
  --num_heads 4 `
  --graph_layers 2 `
  --graph_window 10 `
  --base_history_window 4 `
  --max_history_window 8 `
  --lambda_aux 0.2 `
  --lambda_hgr 0.1 `
  --learning_rate 2e-4 `
  --patience 5 `
  2>&1 | Tee-Object -FilePath $log
```

另开 PowerShell 实时查看 GPU：

```powershell
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 2
```

如果训练已在后台启动，可用日志文件实时监控：

```powershell
Get-Content 'D:\C_Code\PycharmProjects\TFGCN_MLLM\causal_dialoguegcn_mllm_4b\outputs\full_train_实际时间.log' -Wait
```

单轮 baseline 只需把 `--epochs 9` 改为 `--epochs 1`。第一轮是标准提示基线；从第二轮
起才启用上一轮预测驱动的因果双系统提示。因此要观察完整的动态路由，至少训练两轮。

## 测试集评估

```powershell
python -u -m causal_dialoguegcn_mllm_4b.evaluate `
  --checkpoint_dir 'causal_dialoguegcn_mllm_4b\outputs\qwen3_4b_full\best_model' `
  --split test
```

## 代码验证

```powershell
python -m compileall -q causal_dialoguegcn_mllm_4b
python -m pytest causal_dialoguegcn_mllm_4b\tests -q
$env:TRANSFORMERS_OFFLINE = '1'
python -u causal_dialoguegcn_mllm_4b\tests\smoke_qwen4b.py
```

真实 Qwen 冒烟测试会加载完整 4B 量化骨干，并对一个真实 IEMOCAP 话语执行前向和反向
传播；它不属于普通 pytest，避免每次测试都重复占用 GPU。
