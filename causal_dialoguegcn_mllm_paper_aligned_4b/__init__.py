"""Paper-interface-aligned Causal-ERC -> DialogueGCN 4B implementation."""

from .causal_router import DualSystemDecision, classify_multimodal_dialogue
from .model import CausalDialogueGCNLLM

__all__ = ["DualSystemDecision", "classify_multimodal_dialogue", "CausalDialogueGCNLLM"]

