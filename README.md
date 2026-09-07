# ERC Research Models

This repository contains controlled implementations for multimodal conversational emotion recognition on IEMOCAP.

## Current experiment line

| Project | Context encoder | Purpose |
|---|---|---|
| `CausalERC_Qwen3_4B_Reproduction` | DialogueRNN | Causal-ERC Qwen3-4B reproduction baseline |
| `CausalERC_DialogueGCN_Qwen3_4B` | Bidirectional relational DialogueGCN | Controlled graph-context extension of the baseline |

The remaining lowercase directories are earlier research prototypes retained for experiment history. New controlled experiments should use the two projects above.

Datasets, model weights, checkpoints, logs and experiment outputs are intentionally excluded from Git. Place IEMOCAP at `data/iemocap/data.pkl` inside the selected project and provide Qwen3-4B locally or through Hugging Face.

The DialogueGCN project supports one-command training followed by evaluation:

```bash
cd CausalERC_DialogueGCN_Qwen3_4B
python -u run.py --config config.yaml
```
