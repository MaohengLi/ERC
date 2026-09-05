# Causal-ERC-Qwen3-4B reproduction baseline

Independent strict baseline for the AAAI 2026 Causal-ERC method on six-class
IEMOCAP. Qwen/Qwen3-4B is used as the only backbone substitution required by
the local GPU budget.

```text
Text/Audio/Visual -> Linear+LayerNorm -> three DialogueRNN states
 -> cross-modal attention -> Soft-HGR -> Causal Prompt
 -> <|audio_feat|> <|visual_feat|> <|text_feat|> continuous embeddings
 -> Qwen3-4B NF4 + QLoRA -> six-class classifier
```

Labels: `angry, happy, sad, neutral, excited, frustrated`. C1/C2 is selected
from the current dialogue history and utterance in a single pass; no
epoch-level prediction cache is used. This baseline contains exactly A/V/T
feature tokens—no Graph Token or Route Token.

Install and test:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Train and evaluate:

```powershell
python -u train.py --model_name Qwen/Qwen3-4B --epochs 10 --batch_size 1 --grad_accumulation 4 --max_length 768
python -u evaluate.py --checkpoint_dir outputs/qwen3_4b/best_model
```
