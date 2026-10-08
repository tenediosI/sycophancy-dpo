---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
language: [en]
pipeline_tag: text-generation
datasets: [allenai/ai2_arc]
tags: [dpo, lora, peft, trl, sycophancy, alignment]
---

# Qwen2.5-1.5B-Instruct: sycophancy-resistant LoRA (DPO)

A LoRA adapter for [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
trained with DPO so that the model **holds a correct answer when a user pushes back
wrongly, but still accepts a correction when the user is right**.

Code, data, full results and statistics: <https://github.com/tenediosI/sycophancy-dpo>

## Results

Test split: 1550 ARC science questions. Each question gets one round of pushback after the
model answers: wrong pushback (a wrong option) if its first answer was right, correct
pushback if it was wrong. Phrasings "held out" were never used in training.

| Held-out phrasings (training phrasings) | Base model | This adapter | Mean of 3 training seeds |
|---|---|---|---|
| Capitulation to wrong pushback (%), lower is better | 88.3 (90.6) | 19.2 (19.2) | 18.0 (18.8) |
| ...of which: adopts the user's wrong suggestion (%) | 57.3 (69.8) | 11.3 (11.4) | 11.8 (12.5) |
| Acceptance of valid corrections (%), higher is better | 71.8 (88.0) | 67.6 (71.4) | 69.5 (73.6) |
| MMLU / GSM8K / IFEval (%) | 60.4 / 57.6 / 39.9 | | 60.4 / 54.5 / 38.9 |

Against the base model (3 seeds pooled, paired tests, false discovery rate over 7 metrics):
capitulation falls by about 70 points (q < 0.001) on held-out and training phrasings;
acceptance on held-out phrasings changes by -2.4 points [-8.8, 4.2] (not significant) and
on training phrasings by -14.4 [-20.3, -8.3]; MMLU and IFEval do not change; GSM8K is
-3.1 points [-6.1, -0.2] (q = 0.07).

After training, whether the model holds its answer tracks its own confidence in it
(AUROC 0.89 for this adapter, base model 0.62): it holds answers it is sure of and gives
way when unsure.

This adapter is a retrained copy of seed 1 (same code, data and settings); retrained models
match the originals on 92.5-95.0% of test answers.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "<hf-username>/qwen2.5-1.5b-sycophancy-dpo-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant. When answering a multiple-choice "
     "question, reason briefly and end your reply with a final line of the form \"Answer: <letter>\"."},
    {"role": "user", "content": "Which gas do plants take in for photosynthesis?\n\n"
     "A. oxygen\nB. carbon dioxide\nC. nitrogen\nD. helium"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The adapter was trained and evaluated with this system prompt; continue the conversation
with the user's pushback to see the behaviour.

## Training

- **Data:** synthetic preference pairs built by rejection sampling from the base model on the
  ARC train split (5427 questions). The model answers, a templated user pushes back, and 6
  replies are sampled and labelled against the known answer: holding a correct answer or
  accepting a correct correction is preferred; caving to a wrong suggestion or keeping a
  wrong answer is rejected. 190 correct-pushback and 190 wrong-pushback pairs (balanced), all
  with model-written preferred replies.
- **Method:** DPO with TRL 1.14.1, LoRA r 16, alpha 32, dropout 0.05 on all linear layers,
  beta 0.1, learning rate 1e-4, 3 epochs (about 72 steps), effective batch 16, cosine
  schedule with 10% warmup; the reference model is the base model (adapter disabled).
- **Data mix matters:** with 3 wrong-pushback pairs per correct-pushback pair instead,
  the same method made the model stubborn (capitulation 14%, acceptance of valid
  corrections 27%).

## Limitations

- Small model, one domain: grade-school science multiple choice. Behaviour on open-ended
  questions, other domains or larger models is untested.
- Synthetic, templated, single-turn pushback; real users argue differently and repeatedly.
- On the pushback phrasings used in training, acceptance of valid corrections drops 14
  points; most remaining failures are confidently wrong answers the model keeps.
- Possible small arithmetic cost (GSM8K -3.1 points, not significant after correction).
- Answer-only confidence is more extreme than the base model's (mean 99% vs 95%) and
  slightly less calibrated (ECE 13.5% vs 9.6%).
- The training configuration was found by exploration over several runs, not fixed in
  advance; see the repository for the full history.

## Intended use

Research on sycophancy and preference tuning. Not intended for deployment; the model can
still be confidently wrong and can still give in to persistent users.
