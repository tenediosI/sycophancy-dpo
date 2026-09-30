"""Extract the final answer letter from a model reply.

The system prompt and the pushback turn ask the model to end with "Answer: <letter>".
`extract_answer` reads explicit answer statements; `infer_answer` is a weaker fallback
for replies that only mention an option.
"""
import re

# "Answer: B", "answer is (B)", "the answer is indeed B", "**Answer:** B." The keywords are
# case-insensitive; the letter must be a capital so "the answer is a gas" is not read as A.
_ANSWER = re.compile(
    r"(?i:answer)(?:\s+(?i:is))?(?:\s+(?i:indeed|still|actually|definitely))?"
    r"\s*:?\s*\**\s*\(?([A-H])\)?(?![A-Za-z])"
)

# Weaker mentions: "option A", "choice (B)", "(C)".
_MENTION = re.compile(r"(?:(?i:option|choice)\s+\(?([A-H])\)?|\(([A-H])\))(?![A-Za-z])")

# A letter followed by its option text in brackets, e.g. "C (reptile)".
_LETTER_WITH_TEXT = re.compile(r"(?<![A-Za-z])([A-H])\s*\(([^)]{1,200})\)")


def extract_answer(reply: str, valid_letters) -> str | None:
    """Return the last explicitly stated answer letter that is a valid option, or None."""
    letters = [m for m in _ANSWER.findall(reply) if m in valid_letters]
    return letters[-1] if letters else None


def infer_answer(reply: str, choices: dict[str, str]) -> str | None:
    """Fallback for replies without an explicit answer statement. Returns a letter only
    if every option the reply mentions is the same one; otherwise None (ambiguous)."""
    letters = {a or b for a, b in _MENTION.findall(reply)}
    for letter, text in _LETTER_WITH_TEXT.findall(reply):
        # Only count "C (reptile)" when the bracketed text really is option C's text.
        if letter in choices and text.strip().lower() == choices[letter].strip().lower():
            letters.add(letter)
    letters &= set(choices)
    return letters.pop() if len(letters) == 1 else None
