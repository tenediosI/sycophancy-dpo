"""Load any Hugging Face chat model from the model config and generate replies."""
import logging

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def load_model(model_cfg: DictConfig):
    """Return (model, tokenizer) for the configured model, ready for batched generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name, dtype=getattr(torch, model_cfg.dtype), device_map=model_cfg.device_map
    )
    model.eval()
    log.info("Loaded %s on %s", model_cfg.name, model.device)
    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    conversations: list[list[dict]],
    max_new_tokens: int,
    temperature: float = 0.0,
    batch_size: int = 8,
    desc: str = "generate",
    continue_final_message: bool = False,
) -> list[str]:
    """Generate one assistant reply per conversation. temperature=0 means greedy decoding.

    With continue_final_message=True, the last (assistant) message is continued
    instead of starting a new reply.
    """
    sampling = (
        {"do_sample": True, "temperature": temperature}
        if temperature > 0
        else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    )
    replies = []
    for start in tqdm(range(0, len(conversations), batch_size), desc=desc):
        inputs = tokenizer.apply_chat_template(
            conversations[start : start + batch_size],
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **sampling
        )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        replies.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return replies
