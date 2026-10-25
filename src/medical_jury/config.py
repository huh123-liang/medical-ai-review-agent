from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_local_env(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    file = root / ".env.local"
    if file.exists():
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values | dict(os.environ)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    api_key: str = field(default="", repr=False)
    model: str = "deepseek-flash"
    input_rate: float = 3.0
    output_rate: float = 12.0
    pricing_valid_until: str = "2026-10-31"
    budget_cny: float = 30.0
    max_output_tokens: int = 1200
    timeout_seconds: int = 90
    max_attempts: int = 1
    # Model and URL are not controlled by evaluated text or HTTP clients.
    provider_url: str = "https://api.deepseek.com/chat/completions"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    judge_profile: str = 'scoped-review-v1'
    evidence_policy: str = 'anchored-v2'

    @classmethod
    def load(cls, root: Path = ROOT) -> "Settings":
        env = load_local_env(root)
        return cls(
            root=root, api_key=env.get("DEEPSEEK_API_KEY", ""),
            model=env.get("JURY_MODEL", "deepseek-flash"),
            input_rate=float(env.get("JURY_INPUT_CNY_PER_MILLION", "3")),
            output_rate=float(env.get("JURY_OUTPUT_CNY_PER_MILLION", "12")),
            pricing_valid_until=env.get("JURY_PRICING_VALID_UNTIL", "2026-10-31"),
            budget_cny=min(30.0, max(0.0, float(env.get("JURY_BUDGET_CNY", "30")))),
            max_output_tokens=int(env.get("JURY_MAX_OUTPUT_TOKENS", "1200")),
            timeout_seconds=int(env.get("JURY_TIMEOUT_SECONDS", "90")),
            max_attempts=int(env.get("JURY_MAX_ATTEMPTS", "1")),
            judge_profile=env.get('JURY_JUDGE_PROFILE','scoped-review-v1'),
            evidence_policy=env.get('JURY_EVIDENCE_POLICY','anchored-v2'),
        )

    def prepare(self) -> None:
        for name in ("data/private", "data/public", "data/vector", ".runtime", ".cache", "reports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.root / ".cache/huggingface"))
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.root / ".cache/fastembed"))
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"

    def validate_paid(self) -> None:
        import math
        if self.judge_profile not in {'v2-strict','v3-rubric-first','v4-checklist','scoped-review-v1','complex-boundary-v1','complex-boundary-v2','task-scope-v1','evidence-id-v1','task-evidence-v2','grounded-review-v1','grounded-review-v2'} or self.evidence_policy not in {'strict-v2','anchored-v1','anchored-v2'}:
            raise ValueError('Unknown judge profile or evidence policy.')
        if not self.api_key:
            raise ValueError("No local DeepSeek credential. Run Configure-DeepSeek.cmd; never send it in chat.")
        if self.provider_url != "https://api.deepseek.com/chat/completions":
            raise ValueError("Only the explicitly approved DeepSeek HTTPS endpoint is enabled.")
        if date.today() > date.fromisoformat(self.pricing_valid_until):
            raise ValueError("Pricing snapshot expired. Recheck official prices before paid calls.")
        if not all(math.isfinite(n) and n > 0 for n in (self.input_rate, self.output_rate)):
            raise ValueError("Paid calls require positive finite pricing ceilings.")
        # Supported price bindings: never apply Flash rates to Pro silently.
        floors = {"deepseek-flash": (3.0, 12.0), "deepseek-v4-pro": (13.2, 39.6)}
        if self.model not in floors:
            raise ValueError("Model has no reviewed price binding; configure/review the adapter first.")
        minimum = floors[self.model]
        if self.input_rate < minimum[0] or self.output_rate < minimum[1]:
            raise ValueError("Configured price ceiling is below the reviewed conservative model rates.")
        if not 128 <= self.max_output_tokens <= 8192 or not 1 <= self.max_attempts <= 2:
            raise ValueError("Output/retry bounds are outside the approved small-budget policy.")
