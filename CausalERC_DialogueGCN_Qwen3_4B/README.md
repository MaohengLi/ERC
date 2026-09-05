# CausalERC DialogueGCN Qwen3-4B

Controlled extension of `CausalERC_Qwen3_4B_Reproduction` in which the three
DialogueRNN encoders are replaced by independent relational DialogueGCNs. All
other interfaces—data, A/V/T continuous tokens, Peak-End causal prompting,
Soft-HGR and Qwen3-4B QLoRA—remain aligned with the baseline.

```text
T/A/V -> modality projections -> causal dialogue graph
      -> three independent DialogueGCNs -> cross-modal attention
      -> Stage-1 Qwen -> Peak-End C1/C2 -> Stage-2 Qwen
      -> CE + lambda*Soft-HGR
```

Run tests with `python -m pytest tests -q`. Training and evaluation read all
experiment settings from `config.yaml`:

```powershell
python -u train.py --config config.yaml
python -u evaluate.py --config config.yaml
```
