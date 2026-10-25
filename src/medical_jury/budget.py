from __future__ import annotations

import math
from uuid import uuid4

from .config import Settings
from .domain import canonical
from .store import Store, timestamp


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Pre-call upper-bound reservation. Unknown/failed requests keep their reservation."""
    def __init__(self, settings: Settings, store: Store):
        self.settings, self.store = settings, store

    def total(self) -> dict:
        with self.store.connect() as db:
            total = float(db.execute("SELECT COALESCE(SUM(accounted),0) FROM costs").fetchone()[0])
            calls = int(db.execute("SELECT COUNT(*) FROM costs").fetchone()[0])
        return {"limit_cny": self.settings.budget_cny, "accounted_cny": round(total, 6),
                "remaining_cny": round(max(0, self.settings.budget_cny - total), 6), "paid_attempts": calls,
                "note": "Conservative local accounting, not an invoice. Provider billing is authoritative."}

    def estimate(self, messages: list[dict]) -> float:
        # UTF-8 byte count + explicit framing cushion is deliberately much more conservative
        # than a chars/4 tokenizer estimate. No discounted cache/off-peak rate is assumed.
        prompt_upper = len(canonical(messages).encode("utf-8")) + 2048
        value = (prompt_upper * self.settings.input_rate + self.settings.max_output_tokens * self.settings.output_rate) / 1_000_000
        return math.ceil(value * 1_000_000) / 1_000_000

    def reserve(self, job_id: str, upper: float, run_limit: float) -> str:
        if not math.isfinite(upper) or upper <= 0 or not math.isfinite(run_limit) or run_limit <= 0:
            raise ValueError("Invalid budget reservation.")
        reservation = uuid4().hex
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM costs WHERE state='reservation_exceeded' LIMIT 1").fetchone():
                raise BudgetExceeded("A previous reservation was exceeded; operator review is required.")
            total = float(db.execute("SELECT COALESCE(SUM(accounted),0) FROM costs").fetchone()[0])
            run_total = float(db.execute("SELECT COALESCE(SUM(accounted),0) FROM costs WHERE job_id=?", (job_id,)).fetchone()[0])
            if total + upper > min(30.0, self.settings.budget_cny) or run_total + upper > run_limit:
                raise BudgetExceeded("Call blocked before sending: total or per-run CNY budget would be exceeded.")
            db.execute("INSERT INTO costs VALUES(?,?,?,?,?,?,?)", (reservation, job_id, upper, upper, "reserved", None, timestamp()))
        return reservation

    def settle(self, reservation: str, usage: dict | None, success: bool):
        # Only documented, successful usage can release unused reservation.
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM costs WHERE id=?", (reservation,)).fetchone()
            if row is None or row["state"] != "reserved":
                raise ValueError("Unknown or already settled reservation.")
            accounted = float(row["reserved"])
            state = "unknown_or_failed"
            if success and usage and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                actual = (usage["prompt_tokens"] * self.settings.input_rate + usage["completion_tokens"] * self.settings.output_rate) / 1_000_000
                accounted = actual
                state = "settled" if actual <= row["reserved"] else "reservation_exceeded"
            db.execute("UPDATE costs SET accounted=?,state=?,usage_json=? WHERE id=?", (accounted, state, canonical(usage) if usage else None, reservation))
        if state == "reservation_exceeded":
            # Halt rather than continuing on a broken price/token-bound assumption.
            raise BudgetExceeded("Actual usage exceeded reservation; review price/token accounting before continuing.")
