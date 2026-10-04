"""Step 4: synthetic preference pairs by rejection sampling.

1. Turn 1: the base model answers each question several times at temperature > 0. A
   correct answer gives a wrong-pushback conversation, a wrong one a correct-pushback
   conversation (at most one of each per question).
2. Turn 2: the user pushes back with a training phrasing (never a test-only one).
3. Turn 3: several sampled replies per conversation, labelled against the known answer:
   - wrong pushback: good = keeps the correct answer, bad = switches to the suggested option.
   - correct pushback: good = switches to the correct answer, bad = keeps its wrong answer.
   Replies without an explicit "Answer: X", or ending on some other option, are not used.
4. One (chosen, rejected) pair per conversation with at least one good and one bad reply.
   If no reply is good, the chosen reply may come from a template; each pair records
   whether its chosen reply was sampled or templated. All pairs are saved.
5. At training time, `select_pairs` keeps every correct-pushback pair (the rare ones) and
   a capped number of wrong-pushback pairs per correct-pushback pair, preferring sampled
   chosen replies over templated ones. Keeping both conditions stops the model from
   lowering capitulation simply by learning to always disagree.

Prompts use the same system prompt and format reminder as the evaluation.
"""
import random
from collections import Counter, defaultdict

from omegaconf import DictConfig

from src.data.pushback import (
    CORRECT_PUSHBACK,
    UNPARSEABLE,
    WRONG_PUSHBACK,
    assign_condition,
    turn1_conversation,
    turn3_conversation,
)
from src.evaluation.extract import extract_answer
from src.models.factory import generate

CONDITIONS = (WRONG_PUSHBACK, CORRECT_PUSHBACK)
GOOD, BAD = "good", "bad"


def sample_turn1(records: list[dict], model, tokenizer, cfg: DictConfig, rng: random.Random) -> list[dict]:
    """Sample turn 1 several times per question and keep the first correct and the first
    wrong answer as the start of a wrong- and a correct-pushback conversation."""
    g = cfg.generate
    n = g.turn1.n_samples
    conversations = [
        turn1_conversation(cfg.eval.system_prompt, r, cfg.pushback.format_reminder) for r in records
    ]
    replies = generate(
        model,
        tokenizer,
        [c for c in conversations for _ in range(n)],
        max_new_tokens=g.max_new_tokens,
        temperature=g.turn1.temperature,
        batch_size=g.batch_size,
        desc="turn 1 samples",
    )
    items = []
    for i, record in enumerate(records):
        kept = {}
        for reply in replies[i * n : (i + 1) * n]:
            answer = extract_answer(reply, record["choices"])
            condition, suggested = assign_condition(record, answer, rng)
            if condition != UNPARSEABLE and condition not in kept:
                kept[condition] = {
                    **record,
                    "turn1_reply": reply,
                    "turn1_answer": answer,
                    "condition": condition,
                    "suggested": suggested,
                    "template": rng.choice(list(cfg.pushback.train_templates)),
                }
        items.extend(kept[c] for c in CONDITIONS if c in kept)
    return items


def label_reply(item: dict, answer: str | None) -> str | None:
    """GOOD for the desired behaviour, BAD for the failure this condition targets, else None."""
    if answer is None:
        return None
    if answer == item["answer"]:
        return GOOD
    failure = item["suggested"] if item["condition"] == WRONG_PUSHBACK else item["turn1_answer"]
    return BAD if answer == failure else None


def sample_turn3(items: list[dict], model, tokenizer, cfg: DictConfig) -> list[list[dict]]:
    """Sample several turn-3 replies per conversation and label each one."""
    g = cfg.generate
    n = g.turn3.n_samples
    conversations = [
        turn3_conversation(cfg.eval.system_prompt, item, item["template"], cfg.pushback.format_reminder)
        for item in items
    ]
    replies = generate(
        model,
        tokenizer,
        [c for c in conversations for _ in range(n)],
        max_new_tokens=g.max_new_tokens,
        temperature=g.turn3.temperature,
        batch_size=g.batch_size,
        desc="turn 3 samples",
    )
    samples = []
    for i, item in enumerate(items):
        labelled = []
        for reply in replies[i * n : (i + 1) * n]:
            answer = extract_answer(reply, item["choices"])
            labelled.append({"reply": reply, "answer": answer, "label": label_reply(item, answer)})
        samples.append(labelled)
    return samples


def fallback_reply(item: dict, templates: DictConfig) -> str:
    letter = item["answer"]
    return templates[item["condition"]].format(answer=f"{letter} ({item['choices'][letter]})", letter=letter)


def build_pairs(items: list[dict], samples: list[list[dict]], cfg: DictConfig, rng: random.Random) -> list[dict]:
    """One pair per conversation that has a bad reply and a good (or templated) one."""
    pairs = []
    for item, replies in zip(items, samples):
        good = [s["reply"] for s in replies if s["label"] == GOOD]
        bad = [s["reply"] for s in replies if s["label"] == BAD]
        if not bad:
            continue
        if good:
            chosen, chosen_source = rng.choice(good), "sampled"
        elif cfg.generate.chosen_fallback == "template":
            chosen, chosen_source = fallback_reply(item, cfg.generate.fallback_templates), "template"
        else:
            continue
        pairs.append(
            {
                # TRL conversational preference format with an explicit prompt.
                "prompt": turn3_conversation(
                    cfg.eval.system_prompt, item, item["template"], cfg.pushback.format_reminder
                ),
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rng.choice(bad)}],
                # Metadata, ignored by training.
                "id": item["id"],
                "source": item["source"],
                "condition": item["condition"],
                "answer": item["answer"],
                "turn1_answer": item["turn1_answer"],
                "suggested": item["suggested"],
                "pushback": item["template"],
                "chosen_source": chosen_source,
                "n_good": len(good),
                "n_bad": len(bad),
            }
        )
    return pairs


def select_pairs(
    pairs: list[dict], wrong_per_correct: float | None, prefer_sampled_chosen: bool, seed: int
) -> tuple[list[dict], dict]:
    """Choose the training pairs from everything `generate` saved.

    Keeps all correct-pushback pairs and at most `wrong_per_correct` wrong-pushback pairs
    per correct-pushback pair (None keeps all). With `prefer_sampled_chosen`, wrong-pushback
    pairs whose chosen reply was sampled are taken before templated ones, to limit how much
    of the data shares the template wording. Returns (pairs, summary counts).
    """
    rng = random.Random(seed)
    by_condition = defaultdict(list)
    for pair in pairs:
        by_condition[pair["condition"]].append(pair)
    correct = by_condition[CORRECT_PUSHBACK]
    wrong = by_condition[WRONG_PUSHBACK][:]
    rng.shuffle(wrong)
    if prefer_sampled_chosen:
        wrong.sort(key=lambda p: p["chosen_source"] != "sampled")  # stable: random within each group
    if wrong_per_correct is not None:
        wrong = wrong[: round(wrong_per_correct * len(correct))]

    kept = correct + wrong
    rng.shuffle(kept)
    summary = {
        "available": dict(Counter(p["condition"] for p in pairs)),
        "selected": dict(Counter(p["condition"] for p in kept)),
        "chosen_source": {
            c: dict(Counter(p["chosen_source"] for p in kept if p["condition"] == c)) for c in CONDITIONS
        },
    }
    return kept, summary


def build_preferences(records: list[dict], model, tokenizer, cfg: DictConfig) -> tuple[list[dict], list[dict], dict]:
    """Return (all pairs, every labelled turn-3 sample, summary counts)."""
    rng = random.Random(cfg.seed)
    items = sample_turn1(records, model, tokenizer, cfg, rng)
    samples = sample_turn3(items, model, tokenizer, cfg)
    pairs = build_pairs(items, samples, cfg, rng)

    sample_rows = [
        {"id": item["id"], "condition": item["condition"], "suggested": item["suggested"], **s}
        for item, replies in zip(items, samples)
        for s in replies
    ]
    summary = {
        "n_questions": len(records),
        "conversations": dict(Counter(i["condition"] for i in items)),
        "turn3_labels": {
            c: dict(Counter(str(s["label"]) for s in sample_rows if s["condition"] == c)) for c in CONDITIONS
        },
        "pairs": dict(Counter(p["condition"] for p in pairs)),
        "chosen_source": {
            c: dict(Counter(p["chosen_source"] for p in pairs if p["condition"] == c)) for c in CONDITIONS
        },
    }
    return pairs, sample_rows, summary


def spot_check_report(pairs: list[dict], n: int, seed: int) -> str:
    """Markdown page of random pairs for checking the labels by hand, half from each
    condition (correct-pushback pairs are rare and would otherwise barely appear)."""
    lines = [
        "# Preference pairs: spot check",
        "",
        "Check that each chosen reply really shows the desired behaviour and each rejected one the failure.",
        "",
    ]
    rng = random.Random(seed)
    sample = []
    for c in CONDITIONS:
        group = [p for p in pairs if p["condition"] == c]
        sample += rng.sample(group, min(n // 2, len(group)))
    for k, pair in enumerate(sample, 1):
        lines += [
            f"## {k}. {pair['id']} ({pair['condition']}, chosen {pair['chosen_source']})",
            "",
            f"**Correct answer:** {pair['answer']}  ",
            f"**Turn 1 answer:** {pair['turn1_answer']}  ",
            f"**User suggested:** {pair['suggested']}",
            "",
            f"**Pushback:** {pair['prompt'][-1]['content']}",
            "",
            "**Chosen:**",
            "",
            "> " + pair["chosen"][0]["content"].replace("\n", "\n> "),
            "",
            "**Rejected:**",
            "",
            "> " + pair["rejected"][0]["content"].replace("\n", "\n> "),
            "",
        ]
    return "\n".join(lines)
