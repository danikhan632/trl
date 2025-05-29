import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Dict, Tuple


def get_token_mask(
    full_text: str,
    mask_text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize texts and return input ids, attention mask and mask for ``mask_text`` tokens."""
    tokens_full = tokenizer(full_text, return_tensors="pt")
    tokens_mask = tokenizer(mask_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    input_ids = tokens_full["input_ids"]
    attention_mask = tokens_full["attention_mask"]
    mask = torch.zeros_like(input_ids)

    full_ids = input_ids[0].tolist()
    mask_ids = tokens_mask.tolist()

    for i in range(len(full_ids) - len(mask_ids) + 1):
        if full_ids[i : i + len(mask_ids)] == mask_ids:
            mask[0, i : i + len(mask_ids)] = 1
            break
    else:
        raise ValueError("mask_text tokens not found in full_text")

    return input_ids, attention_mask, mask


def compute_masked_gradients(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    full_text: str,
    mask_text: str,
) -> Dict[str, float]:
    """Compute gradients where only tokens from ``mask_text`` contribute."""
    input_ids, attention_mask, mask = get_token_mask(full_text, mask_text, tokenizer)

    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    shifted_input_ids = input_ids[:, 1:]
    shifted_log_probs = log_probs[:, :-1, :]
    shifted_mask = mask[:, 1:]

    selected = shifted_log_probs.gather(2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = selected[shifted_mask.bool()]
    if masked_log_probs.numel() == 0:
        raise ValueError("mask_text does not align with tokens in full_text")

    loss = -masked_log_probs.mean()
    loss.backward()

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    return grad_norms
