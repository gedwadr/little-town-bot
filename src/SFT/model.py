from dataclasses import dataclass, field

from src.constant import SFT_PROMPT

@dataclass
class Turn:
    """All events belonging to one player's turn."""
    turn_num:        int
    round_num:       int
    player_id:       str
    decision_events: list = field(default_factory=list)
    gather_details:  list = field(default_factory=list)
    all_events:      list = field(default_factory=list)


@dataclass
class TrainingExample:
    round_num:    int
    turn_num:     int
    player_id:    str
    user_content: str
    asst_content: str

    def to_messages(self) -> dict:
        return {
            "messages": [
                {"role": "system",    "content": SFT_PROMPT},
                {"role": "user",      "content": self.user_content},
                {"role": "assistant", "content": self.asst_content},
            ]
        }