# CausalERC DialogueGCN Qwen3-4B

Controlled extension of `CausalERC_Qwen3_4B_Reproduction` in which the three
DialogueRNN encoders are replaced by independent relational DialogueGCNs. All
other interfaces—data, A/V/T continuous tokens, Peak-End causal prompting,
Soft-HGR and Qwen3-4B LoRA—remain aligned with the baseline. The cloud path
uses BF16 and is compatible with RTX 5090.

```text
T/A/V -> modality projections -> causal dialogue graph
      -> three independent DialogueGCNs -> cross-modal attention
      -> Stage-1 Qwen -> Peak-End C1/C2 -> Stage-2 Qwen
      -> CE + lambda*Soft-HGR
```

Run tests with `python -m pytest tests -q`. All experiment settings are read
from `config.yaml`. The default five-epoch schedule still saves the checkpoint
with the best development Weighted-F1.

One command runs training and then evaluates the best checkpoint:

```bash
python -u run.py --config config.yaml
```

The individual entry points remain available for debugging:

```bash
python -u train.py --config config.yaml
python -u evaluate.py --config config.yaml
```

On an offline server, set `model.backbone` to the local model directory, for
example `/models/Qwen3-4B`, and enable `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.
