# Causal-ERC 4B Reproduction

This directory is an isolated, paper-aligned reproduction of **Causal-ERC
(AAAI 2026)** for the six-class IEMOCAP task. The original Llama-3.1-8B is
replaced only because of the local GPU limit:

- base LLM: `Qwen/Qwen3-4B`
- quantisation: 4-bit NF4 + double quantisation
- adaptation: QLoRA, `r=16`, `alpha=32`, dropout `0.05`
- multimodal backbone: three speaker-aware DialogueRNN encoders
- fusion: three cross-attention branches
- consistency: Soft-HGR (paper Equation 6)
- causal prompting: previous-epoch Peak-End C1/C2 decisions (Algorithm 1)

## Architecture

```text
Text(100)  -> DialogueRNN(200) --\
Audio(100) -> DialogueRNN(200) ----> 3-way Cross-Attention
Video(512) -> DialogueRNN(200) --/       |       |       |
                                      f_text  f_audio  f_video
                                         |       |       |
                                     Linear projections to 2560
                                         |       |       |
Prompt + <|text_feat|> <|audio_feat|> <|visual_feat|>
                              |
                     Qwen3-4B 4-bit QLoRA
                              |
                 final-token hidden -> Linear(2560, 6)
```

From epoch 2 onward, cached logits are stored by dialogue ID. For each current
utterance, only preceding utterances in the same dialogue are used to compute:

```text
lambda_C1 = ||o_current - mean(o_history)||_2
lambda_C2 = ||o_current - (o_peak + o_end)/2||_2
```

`C1` uses a dynamically enlarged history; `C2` explicitly identifies the peak
utterance. Test evaluation is two-pass: standard prediction followed by causal
re-prompting. No future utterance or gold label enters a causal decision.

## Important correctness differences from the legacy baseline

1. DialogueRNN receives padded `[dialogue_batch, utterance, feature]` tensors;
   independent dialogues are never concatenated into one GRU sequence.
2. Previous-epoch logits and causal decisions are keyed by `vid`; shuffling the
   training dialogues cannot misalign causal types.
3. Missing multimodal special tokens cause a hard error rather than silently
   overwriting token positions 0/1 after truncation.
4. The first utterance uses a standard prompt because Peak-End history is empty.

## Step 1 - environment and unit tests

Run from the repository root:

```powershell
python -m unittest discover -s causal_erc_repro_4b/tests -v
```

Install missing training dependencies if necessary:

```powershell
python -m pip install -r causal_erc_repro_4b/requirements.txt
```

## Step 2 - train

The defaults point to the existing IEMOCAP `data.pkl` and are conservative for
the local 12 GB GPU. `llm_micro_batch` controls how many utterance prompts enter the 4B
LLM simultaneously.

Before the full experiment, run a one-epoch pipeline check on one train/dev
dialogue (the score from this command is not a valid result):

```powershell
cd D:\C_Code\PycharmProjects\TFGCN_MLLM\causal_erc_repro_4b
python -u train.py `
  --epochs 1 `
  --max_train_dialogues 1 `
  --max_dev_dialogues 1 `
  --llm_micro_batch 1 `
  --output_dir outputs/smoke_run
```

Then run the full split:

```powershell
cd D:\C_Code\PycharmProjects\TFGCN_MLLM\causal_erc_repro_4b
python -u train.py `
  --model_name Qwen/Qwen3-4B `
  --dialogue_batch_size 1 `
  --llm_micro_batch 2 `
  --grad_accum 4 `
  --epochs 9 `
  --max_length 256 `
  --base_history_window 4 `
  --max_history_window 8
```

If memory is still insufficient, set `--llm_micro_batch 1`; this changes speed,
not the mathematical batch or causal decisions.

## Step 3 - held-out test evaluation

```powershell
python -u evaluate.py `
  --checkpoint_dir outputs/qwen3_4b/best_model `
  --llm_micro_batch 1
```

Outputs:

```text
outputs/qwen3_4b/
  best_model/
    lora/
    tokenizer/
    non_llm.pt
    metadata.json
  training_history.json
  test_results.json
```

This reproduction intentionally does not include the later CVH/teacher
distillation experiments in `baseline/`; they are not part of the paper's
Algorithm 1 and would prevent a clean reproduction comparison.

## Verified locally

The initial implementation was verified on the local RTX 4070 Ti (12 GB):

- IEMOCAP splits: 108/12/31 dialogues and 5262/548/1593 utterances
- feature dimensions: text 100, audio 100, visual 512
- all 64 utterances in a real dialogue retained exactly one token per modality
- Qwen3-4B loaded successfully with NF4 QLoRA
- trainable parameters: 36,619,062
- two-utterance forward peak allocation: about 4.8 GiB
- one-utterance backward peak allocation: about 4.1 GiB

These are allocated-memory observations before AdamW has created all optimizer
states, so the conservative default remains `dialogue_batch_size=1` and
`llm_micro_batch=2`.
