# Causal DialogueGCN-LLM (Causal-ERC interface aligned)

这是最终推荐的论文主线实现：

```text
Causal-ERC
→ DialogueRNN 替换为 DialogueGCN
→ 保持 Audio / Visual / Text 三Token LLM接口
→ 多模态 Peak-End 双系统路由
→ C1/C2 通过 Prompt 和特征门控实现
```

## 数据流

```text
Text/Audio/Video + Speaker
        ↓
三路输入投影 → 话语级跨模态注意力
        ↓
说话人关系 DialogueGCN（默认过去向）
        ↓
f_audio / f_visual / f_text
        ├─ T/A/V 内部路由探针（只为下一轮判定）
        └─ C1/C2 α 特征门控
        ↓
三个投影：Tᵃ、Tᵛ、Tᵗ
        ↓
Causal Prompt + 三个连续特征Token
        ↓
Qwen3-4B NF4 + LoRA → 六分类情绪预测
```

LLM输入严格保持为：

```text
Tokenizer(Causal Prompt)
+ Audio feature token
+ Visual feature token
+ Text feature token
```

没有 `Graph Token`，也没有 `Causal Route Token`。Graph context 在三类特征投影之前
通过门控融合，C1/C2 路由信息通过 Prompt 和门控共同作用。

## 训练与评估

训练第一轮使用标准 Prompt；从第二轮开始，使用上一轮按 dialogue ID 缓存的多模态
T/A/V/LLM logits 计算 Peak-End 路由。C1 使用较长历史，C2 使用 Peak/End 提示。
训练和验证都不会用真实标签生成路由。

完整训练（PowerShell）：

```powershell
Set-Location 'D:\C_Code\PycharmProjects\TFGCN_MLLM'
$env:PYTHONUNBUFFERED = '1'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = "causal_dialoguegcn_mllm_paper_aligned_4b\outputs\full_train_$stamp.log"
python -u -m causal_dialoguegcn_mllm_paper_aligned_4b.train `
  --data_path 'D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN\MultiModal_DialogGCN\save\data.pkl' `
  --model_name 'Qwen/Qwen3-4B' `
  --output_dir 'causal_dialoguegcn_mllm_paper_aligned_4b\outputs\qwen3_4b_full' `
  --epochs 9 --dialogue_batch_size 1 --llm_micro_batch 2 --grad_accum 4 `
  --max_length 256 --hidden_dim 200 --num_heads 4 --graph_layers 2 `
  --graph_window 10 --base_history_window 4 --max_history_window 8 `
  --lambda_hgr 1.0 --lambda_aux 0.1 --learning_rate 2e-4 --patience 5 `
  2>&1 | Tee-Object -FilePath $log
```

单轮 baseline：将 `--epochs 9` 改为 `--epochs 1`。

实时GPU监控：

```powershell
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 2
```

测试集评估：

```powershell
python -u -m causal_dialoguegcn_mllm_paper_aligned_4b.evaluate `
  --checkpoint_dir 'causal_dialoguegcn_mllm_paper_aligned_4b\outputs\qwen3_4b_full\best_model' `
  --split test
```

代码检查：

```powershell
python -m compileall -q causal_dialoguegcn_mllm_paper_aligned_4b
python -m pytest causal_dialoguegcn_mllm_paper_aligned_4b\tests -q
```

