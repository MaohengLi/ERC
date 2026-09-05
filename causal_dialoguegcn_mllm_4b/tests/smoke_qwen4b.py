"""Optional real-GPU smoke test; not collected by pytest."""

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal_dialoguegcn_mllm_4b.data_dual import build_tokenizer, collate_dual, load_iemocap
from causal_dialoguegcn_mllm_4b.runtime import (
    build_train_model, move_tensors, route_features_for_dialogues,
)


DATA = (r"D:\C_Code\PycharmProjects\MMAAE_MDiaGCN\MMAAE_MDiaGCN"
        r"\MultiModal_DialogGCN\save\data.pkl")


def main() -> None:
    args = SimpleNamespace(
        model_name="Qwen/Qwen3-4B", hidden_dim=200, num_heads=4,
        graph_layers=2, graph_window=10, dropout=0.0, lambda_hgr=0.1,
        lambda_aux=0.2, llm_micro_batch=1, bidirectional_graph=False,
    )
    original = load_iemocap(DATA)["train"][0]
    sample = dict(original)
    sample["utt_count"] = 1
    for key in ("text_feats", "audio_feats", "visual_feats", "speakers", "labels", "texts"):
        sample[key] = original[key][:1]
    tokenizer, special_ids = build_tokenizer(args.model_name)
    model = build_train_model(args, tokenizer, special_ids, sample)
    cpu = collate_dual([sample], tokenizer, special_ids, max_length=192)
    batch = move_tensors(cpu, torch.device("cuda"))
    batch["route_features"] = route_features_for_dialogues(
        [sample], None, batch["labels"].device)
    output = model(batch, labels=batch["labels"])
    output["loss"].backward()
    memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    print({"logits": tuple(output["logits"].shape),
           "aux_logits": tuple(output["aux_logits"].shape),
           "loss": float(output["loss"]), "gpu_peak_gb": round(memory, 3)})


if __name__ == "__main__":
    main()
