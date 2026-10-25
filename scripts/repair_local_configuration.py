"""One-time local migration of a credential accidentally placed in a launcher.

Never prints the credential. Does not inspect the previous MVP2 platform.
"""
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / '.env.local'
    if target.exists():
        raise SystemExit('Stopped: local configuration already exists; no overwrite performed.')
    launcher = root / 'Configure-DeepSeek.cmd'
    found = re.findall(r'^cd\s+/d\s+(sk-[A-Za-z0-9_-]{12,})\s*$', launcher.read_text(encoding='utf-8-sig'), re.M | re.I)
    if len(found) != 1:
        raise SystemExit('Stopped: no unambiguous misplaced credential; use the interactive configurator.')
    content = (f'DEEPSEEK_API_KEY={found[0]}\nJURY_MODEL=deepseek-flash\n'
               'JURY_INPUT_CNY_PER_MILLION=3\nJURY_OUTPUT_CNY_PER_MILLION=12\n'
               'JURY_PRICING_VALID_UNTIL=2026-10-31\nJURY_BUDGET_CNY=30\n'
               'JURY_MAX_OUTPUT_TOKENS=1200\nJURY_MAX_ATTEMPTS=1\nJURY_TIMEOUT_SECONDS=90\n')
    with target.open('x', encoding='utf-8') as file:
        file.write(content)
    print('Local credential migrated. No credential printed. No API request sent.')


if __name__ == '__main__':
    main()
