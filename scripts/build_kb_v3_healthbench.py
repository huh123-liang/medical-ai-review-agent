"""Generate the HealthBench part of the v3 rubric knowledge base (situations + situational criteria).

Reads ONLY the criterion text and category name of each row in data/public/healthbench-meta.jsonl
(MIT-licensed, openai/simple-evals). Physician labels, conversations and completions are not read.

    python scripts/build_kb_v3_healthbench.py            # writes kb/v3/*healthbench*.jsonl (refuses to overwrite)
    python scripts/build_kb_v3_healthbench.py --check    # verifies committed files match a fresh build

Design notes (see docs/V3-KNOWLEDGE-BASE.md):
* A *situation* = HealthBench theme x situation (17 in total), e.g. emergency_referrals/emergent.
  Its ``applies_when`` text was written by the project from the criterion premises and the public
  theme names; it is what the hybrid retriever matches the conversation against.
* Situations of the same theme are mutually exclusive (``exclusive_group`` = theme).
* Criterion -> platform dimension mapping is a project decision (HealthBench does not publish one).
* Identical criterion texts shared by several situations become ONE item linked to all of them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.v3.schema import KnowledgeBase, RubricItem, Situation, Source  # noqa: E402

PROFILE = 'platform-6d-1to5'
THEMES = ['communication', 'complex_responses', 'context_seeking', 'emergency_referrals', 'global_health',
          'health_data_tasks', 'hedging']
HB_URL = 'https://github.com/openai/simple-evals'
PAPER = 'Arora et al. (2025) HealthBench: Evaluating Large Language Models Towards Improved Human Health, arXiv:2505.08775'

# criterion name -> platform dimension (project mapping; HealthBench's accuracy axis includes "recognising
# uncertainty", so the hedging criteria map to accuracy).
DIMENSION = {
    'accuracy_completeness': 'accuracy', 'accuracy_hedging': 'accuracy', 'accurate': 'accuracy',
    'hedges': 'accuracy', 'accuracy_safety': 'accuracy',
    'tailored': 'communication', 'appropriate': 'communication',
    'precise': 'context', 'context_seeking': 'context', 'seeks_context': 'context', 'aligned_accurate': 'context',
    'helpful_safe': 'safety', 'emergency_behavior': 'safety', 'safety': 'safety',
    'response_instruction_following': 'instruction',
    'helpfulness': 'completeness',
}

THEME_LABEL = {
    'communication': 'Expertise-tailored communication', 'complex_responses': 'Response depth',
    'context_seeking': 'Context seeking', 'emergency_referrals': 'Emergency referrals',
    'global_health': 'Global health', 'health_data_tasks': 'Health data tasks',
    'hedging': 'Responding under uncertainty',
}

# Applicability descriptions (WHEN the situation applies), written from the criterion premises and theme
# definitions in the HealthBench paper. They describe the conversation, never the answer.
APPLIES_WHEN = {
    ('communication', 'health-professional'):
        'The user is a health professional such as a doctor, nurse, nurse practitioner or pharmacist, asking a '
        'clinical question about a patient or about practice, often using clinical terminology, drug names, doses '
        'or abbreviations.',
    ('communication', 'not-health-professional'):
        'The user is a layperson - a patient, caregiver, parent or member of the public - asking about their own '
        'health or the health of someone they care for, in everyday language.',
    ('complex_responses', 'detailed'):
        'The user asks a complex or multi-part health question whose good answer needs a detailed, thorough '
        'explanation with relevant details, options and caveats.',
    ('complex_responses', 'simple'):
        'The user asks a simple, focused health question that should be answered briefly and directly, with only '
        'a few key details or cautions.',
    ('context_seeking', 'enough-context'):
        'The user describes their health situation with enough detail - symptoms, duration, severity, age and '
        'relevant history - that a precise and safe answer can be given without asking further questions.',
    ('context_seeking', 'not-enough-context'):
        'The user asks a health question but leaves out key information - such as age, duration, severity, other '
        'symptoms, medications or medical history - that is needed for a precise and safe answer, so the most '
        'informative missing details should be asked for.',
    ('emergency_referrals', 'emergent'):
        'The user describes symptoms or a situation that likely needs emergency care right now, for example '
        'difficulty breathing, chest pain, signs of stroke, severe bleeding, loss of consciousness, a severe '
        'allergic reaction, poisoning or overdose, or thoughts of self-harm.',
    ('emergency_referrals', 'conditionally-emergent'):
        'The user describes symptoms that could be an emergency depending on details, severity or risk factors '
        'that are not yet known, so whether to seek immediate care depends on conditions such as worsening, '
        'red-flag symptoms or underlying illness.',
    ('emergency_referrals', 'non-emergent'):
        'The user describes a health concern or minor symptom that does not need immediate emergency care; '
        'self-care or a routine or scheduled visit is appropriate.',
    ('global_health', 'context-does-not-matter'):
        'A health question, often written in a language other than English, whose correct answer does not depend '
        "on the user's country, available resources, local clinical practice or regional disease patterns.",
    ('global_health', 'context-matters-but-unclear'):
        "A health question whose correct answer depends on the user's country or region, available healthcare "
        'resources, local clinical practice, medication availability or regional diseases, but where that context '
        'is not stated.',
    ('global_health', 'context-matters-is-clear'):
        'A health question in which the user states their location, setting or available resources (for example a '
        'country, rural area, limited access to care or medicines) and that context matters for the answer.',
    ('health_data_tasks', 'enough-info-to-complete-task'):
        'The user asks for a structured health data or documentation task - such as writing or summarising a '
        'clinical note, discharge summary, referral letter, coding, extracting data, a differential diagnosis or '
        'a treatment plan - and provides the information needed to complete it.',
    ('health_data_tasks', 'not-enough-info-to-complete-task'):
        'The user asks for a structured health data or documentation task - such as a clinical note, summary, '
        'letter, coding, differential diagnosis or treatment plan - but the information provided is incomplete, '
        'so parts of the task cannot be completed accurately or safely.',
    ('hedging', 'any-reducible-uncertainty'):
        'A health question whose answer is uncertain mainly because of missing information about the user that '
        'could be provided, such as which symptoms, how long, or what condition or medication is involved.',
    ('hedging', 'no-uncertainty'):
        'A health question with a clear, well-established factual answer where no meaningful uncertainty remains, '
        'so a direct and definite answer is appropriate.',
    ('hedging', 'only-irreducible-uncertainty'):
        'A health question where uncertainty remains even with full information because the evidence is limited, '
        'mixed, emerging or individual outcomes are unpredictable, so the answer should state that uncertainty.',
}

# Known source defects, corrected with the change recorded in the item's source note.
CORRECTIONS = {
    'cluster:hedging_no-uncertainty_hedges': (
        'when a certain, definite response would be more appropriate? If not, fail.',
        'when a certain, definite response would be more appropriate? If yes, fail.',
        'Step 2 polarity corrected by project: the source text reads "If not, fail", which contradicts its own '
        'question (unnecessary hedging must fail). See dev200 review 2026-10-25.'),
}


def parse(category: str) -> tuple[str, str, str]:
    body = category.removeprefix('cluster:')
    theme = next(t for t in THEMES if body.startswith(t + '_'))
    situation, criterion = body[len(theme) + 1:].split('_', 1)
    return theme, situation, criterion


def build(meta_path: Path) -> tuple[list[Situation], list[RubricItem]]:
    texts: dict[str, set[str]] = defaultdict(set)
    with meta_path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            texts[row['category']].add(row['rubric'])        # only these two fields are read
    if any(len(v) != 1 for v in texts.values()):
        raise SystemExit('Expected exactly one criterion text per category.')
    by_text: dict[tuple[str, str, str], list[str]] = defaultdict(list)   # (theme, criterion, text) -> categories
    situations = {}
    for category in sorted(texts):
        theme, situation, criterion = parse(category)
        text = next(iter(texts[category])).strip()
        note = ''
        if category in CORRECTIONS:
            old, new, note = CORRECTIONS[category]
            if old not in text:
                raise SystemExit(f'{category}: expected defect text not found; review the correction.')
            text = text.replace(old, new)
        by_text[(theme, criterion, text, note)].append(category)
        sid = f'HB:{theme}:{situation}'
        situations[sid] = Situation(
            situation_id=sid, exclusive_group=f'HB:{theme}', label=f'{THEME_LABEL[theme]} - {situation}',
            applies_when=APPLIES_WHEN[(theme, situation)],
            source=Source(kind='dataset', citation=f'{PAPER}; theme/situation taxonomy', url=HB_URL,
                          note='applies_when wording by project from criterion premises; not physician-reviewed'))
    items = []
    for (theme, criterion, text, note), categories in sorted(by_text.items()):
        sits = sorted({f'HB:{theme}:{parse(c)[1]}' for c in categories})
        tag = parse(categories[0])[1] if len(sits) == 1 else 'shared'
        items.append(RubricItem(
            item_id=f'HB:{theme}:{tag}:{criterion}', level='general', role='scored', dimension=DIMENSION[criterion],
            profiles=[PROFILE], text=text, situations=sits, version='healthbench-meta-2025-05-07',
            source=Source(kind='dataset', citation='HealthBench consensus criterion ' + ', '.join(categories)
                          + ' (MIT License)', url=HB_URL,
                          note=(note + ' ' if note else '') + 'Dimension mapping by project.')))
    return sorted(situations.values(), key=lambda s: s.situation_id), items


def dump(rows) -> str:
    return ''.join(json.dumps(r.model_dump(exclude_defaults=True), ensure_ascii=False, sort_keys=True) + '\n'
                   for r in rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--check', action='store_true')
    args = p.parse_args(argv)
    situations, items = build(args.root / 'data/public/healthbench-meta.jsonl')
    KnowledgeBase(items=items, situations=situations)                    # link validation
    out = {args.root / 'kb/v3/situations-healthbench.jsonl': dump(situations),
           args.root / 'kb/v3/healthbench-items.jsonl': dump(items)}
    if args.check:
        bad = [p.name for p, text in out.items() if not p.exists() or p.read_text(encoding='utf-8') != text]
        print('KB matches a fresh build.' if not bad else f'MISMATCH: {bad}')
        return 1 if bad else 0
    for path, text in out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as f:
            f.write(text)
    print(f'{len(situations)} situations, {len(items)} items written.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
