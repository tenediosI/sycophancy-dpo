"""Build the conversation turns for the pushback protocol.

Turn 1: the user asks a question and the model answers.
Turn 2: the user pushes back, suggesting an answer.
  - wrong pushback: the model was right and the user suggests a wrong option.
  - correct pushback: the model was wrong and the user suggests the right option.
Turn 3: the model replies; we record whether it keeps or changes its answer.
"""
import random

WRONG_PUSHBACK = "wrong_pushback"
CORRECT_PUSHBACK = "correct_pushback"
UNPARSEABLE = "unparseable"


def format_question(record: dict) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in record["choices"].items())
    return f"{record['question']}\n\n{options}"


def pushback_message(template: str, record: dict, suggested: str) -> str:
    return template.format(answer=f"{suggested} ({record['choices'][suggested]})")


def assign_condition(record: dict, turn1_answer: str | None, rng: random.Random):
    """Return (condition, suggested letter) given the model's turn-1 answer."""
    if turn1_answer is None:
        return UNPARSEABLE, None
    if turn1_answer == record["answer"]:
        wrong_options = [letter for letter in record["choices"] if letter != record["answer"]]
        return WRONG_PUSHBACK, rng.choice(wrong_options)
    return CORRECT_PUSHBACK, record["answer"]


def turn1_conversation(system_prompt: str, record: dict, format_reminder: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{format_question(record)}\n\n{format_reminder}".strip()},
    ]


def turn3_conversation(system_prompt: str, item: dict, template: str, format_reminder: str) -> list[dict]:
    pushback = pushback_message(template, item, item["suggested"])
    return turn1_conversation(system_prompt, item, format_reminder) + [
        {"role": "assistant", "content": item["turn1_reply"]},
        {"role": "user", "content": f"{pushback} {format_reminder}".strip()},
    ]
