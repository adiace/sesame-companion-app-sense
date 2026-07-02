"""
Pre-LLM keyword matching for the Sesame kids content layer.
Returns the same {command, face, response} dict shape as the LLM, or None to fall through.
"""

import random
from typing import Optional

from kids_content import ANIMAL_SOUNDS, JOKES, QA_RESPONSES, get_time_response

_JOKE_TRIGGERS = {"joke", "funny", "laugh", "silly", "make me laugh", "tell me a joke"}
_ANIMAL_PREFIXES = {"sound", "say", "go", "noise", "what sound", "what does", "how does"}


class KidsCommandLayer:

    def match(self, text: str) -> Optional[dict]:
        t = text.lower().strip()
        return self._match_joke(t) or self._match_animal(t) or self._match_qa(t)

    # ── Jokes ─────────────────────────────────────────────────────────────────

    def _match_joke(self, t: str) -> Optional[dict]:
        if not any(kw in t for kw in _JOKE_TRIGGERS):
            return None
        setup, punchline = random.choice(JOKES)
        # Deliver as "setup ... punchline" so TTS reads them sequentially
        response = f"{setup}... {punchline}"
        return {"command": None, "face": "happy", "response": response}

    # ── Animal sounds ─────────────────────────────────────────────────────────

    def _match_animal(self, t: str) -> Optional[dict]:
        # Must have an action prefix to avoid triggering on "I have a dog"
        has_prefix = any(p in t for p in _ANIMAL_PREFIXES)
        if not has_prefix:
            return None
        for animal, (sound, face) in ANIMAL_SOUNDS.items():
            if animal in t:
                return {
                    "command": None,
                    "face": face,
                    "response": sound,
                }
        return None

    # ── Simple Q&A ────────────────────────────────────────────────────────────

    def _match_qa(self, t: str) -> Optional[dict]:
        for _key, entry in QA_RESPONSES.items():
            if any(trigger in t for trigger in entry["triggers"]):
                if entry.get("time_response"):
                    return {
                        "command": entry.get("command"),
                        "face": entry.get("face", "happy"),
                        "response": get_time_response(),
                    }
                responses = entry.get("responses") or [entry.get("response", "")]
                return {
                    "command": entry.get("command"),
                    "face": entry.get("face", "happy"),
                    "response": random.choice(responses),
                }
        return None
