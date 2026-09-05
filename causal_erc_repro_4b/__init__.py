"""Faithful, GPU-conscious Causal-ERC reproduction for a 4B Qwen model."""

from .causal_prompting import CausalDecision, classify_dialogue, build_erc_prompt
from .model import CausalERC4B

__all__ = ["CausalDecision", "classify_dialogue", "build_erc_prompt", "CausalERC4B"]
