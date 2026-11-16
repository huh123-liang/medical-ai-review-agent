"""Step 2: rubric knowledge base + local hybrid retrieval (BM25 + dense + RRF). Offline; hash embedder unless the
real model cache is present (that test is skipped otherwise)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from medical_jury.domain import CaseInput, Message
from medical_jury.v3.hybrid import (CachedEmbedder, FastEmbedEmbedder, HashEmbedder, HybridRetriever, open_index,
                                    query_text, tokens)
from medical_jury.v3.retrieval import FilterRetriever, build_query
from medical_jury.v3.schema import CaseMeta, KnowledgeBase, RubricItem, Situation, Source, load_kb, load_profile

from v2_fixtures import load_script

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / 'kb' / 'v3'
SRC = Source(kind='dataset', citation='unit test')


def profile():
    return load_profile(KB_DIR / 'profile-platform-6d.json')


def sit(sid, group, when):
    return Situation(situation_id=sid, exclusive_group=group, label=sid, applies_when=when, source=SRC)


def item(iid, dim, sits=(), mandatory=False, **kw):
    return RubricItem(item_id=iid, level='general', role='scored', dimension=dim, profiles=['platform-6d-1to5'],
                      text=f'criterion {iid}', source=SRC, situations=list(sits), mandatory=mandatory, **kw)


def mini_kb():
    situations = [
        sit('S:emerg:yes', 'G:emerg', 'Severe chest pain, shortness of breath, fainting, stroke signs: emergency now.'),
        sit('S:emerg:no', 'G:emerg', 'Mild cold, runny nose, minor rash: routine self care, no emergency.'),
        sit('S:role:hp', 'G:role', 'A doctor or nurse asks about dosing for a patient on the ward.'),
    ]
    items = [item('M-ACC', 'accuracy', mandatory=True, cap_on_fail=2), item('M-SAF', 'safety', mandatory=True),
             item('I-EY', 'safety', ['S:emerg:yes']), item('I-EN', 'safety', ['S:emerg:no']),
             item('I-HP', 'communication', ['S:role:hp'])]
    return KnowledgeBase(items=items, situations=situations)


def case(user_text, answer='Some answer.', task='Assess the answer.'):
    return CaseInput(case_id='C1', group_id='G1', task=task, context=[Message(role='user', content=user_text)],
                     candidate_output=answer)


def query(c, situations=()):
    return build_query(c, CaseMeta(situations=list(situations)), profile())


def retriever(tmp_path, kb=None, **kw):
    kb = kb or mini_kb()
    emb = HashEmbedder()
    return HybridRetriever(kb, open_index(kb, emb, tmp_path / 'idx'), emb, **kw)


# ---- knowledge base contracts ---------------------------------------------------------------------------------

def test_kb_link_rules():
    s = [sit('S:a', 'GG', 'when something specific happens to the user')]
    with pytest.raises(ValueError, match='unknown situations'):
        KnowledgeBase(items=[item('IT-X', 'safety', ['S:b'])], situations=s)
    with pytest.raises(ValueError, match='without any item'):
        KnowledgeBase(items=[item('IT-X', 'safety')], situations=s)
    with pytest.raises(ValueError, match='mandatory'):
        item('IT-X', 'safety', ['S:a'], mandatory=True)
    with pytest.raises(ValueError, match='only general-level'):
        RubricItem(item_id='D-X', level='disease', role='reference', dimension='safety', diseases=['dm'],
                   profiles=['p'], text='xyz', source=SRC, situations=['S:a'])


def test_shipped_kb_is_sourced_and_linked():
    kb = load_kb(KB_DIR)
    assert len(kb.situations) == 17 and len([i for i in kb.items if i.situational]) == 30
    assert all(i.source.kind != 'project_draft' for i in kb.items)
    assert all(d.source.kind != 'project_draft' for d in profile().dimensions)
    disease = [i for i in kb.items if i.level == 'disease']
    assert disease and all(i.role == 'reference' for i in disease)              # decision 2026-10-25
    mandatory_dims = {i.dimension for i in kb.items if i.mandatory}
    assert mandatory_dims == {d.key for d in profile().dimensions}             # every dimension has a base check
    fixed = next(i for i in kb.items if i.item_id == 'HB:hedging:no-uncertainty:hedges')
    assert 'would be more appropriate? If yes, fail.' in fixed.text and 'polarity corrected' in fixed.source.note


@pytest.mark.skipif(not (ROOT / 'data/public/healthbench-meta.jsonl').exists(), reason='public data not present')
def test_healthbench_kb_is_reproducible():
    assert load_script('build_kb_v3_healthbench').main(['--check']) == 0


# ---- index --------------------------------------------------------------------------------------------------

def test_index_is_reused_only_when_the_manifest_matches(tmp_path):
    kb, emb = mini_kb(), HashEmbedder()
    a = open_index(kb, emb, tmp_path)
    b = open_index(kb, emb, tmp_path)
    assert a.directory == b.directory and (a.directory / 'manifest.json').exists()
    c = open_index(kb, HashEmbedder(dim=64), tmp_path)
    assert c.directory != a.directory                                       # other model -> other index
    manifest = json.loads((a.directory / 'manifest.json').read_text())
    manifest['documents'] = 99
    (a.directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='mismatch'):
        open_index(kb, emb, tmp_path)


def test_partial_index_is_refused(tmp_path):
    kb, emb = mini_kb(), HashEmbedder()
    d = open_index(kb, emb, tmp_path).directory
    (d / 'manifest.json').unlink()
    with pytest.raises(RuntimeError, match='without a manifest'):
        open_index(kb, emb, tmp_path)


def test_retriever_refuses_an_index_built_for_another_kb(tmp_path):
    kb, emb = mini_kb(), HashEmbedder()
    index = open_index(kb, emb, tmp_path)
    other = KnowledgeBase(items=kb.items + [item('M-COM', 'communication', mandatory=True)], situations=kb.situations)
    with pytest.raises(ValueError, match='different KB'):
        HybridRetriever(other, index, emb)


# ---- retrieval behaviour --------------------------------------------------------------------------------------

def test_mandatory_items_are_always_present(tmp_path):
    r = retriever(tmp_path)
    res = r.retrieve(query(case('zzqx unrelated words')))
    ids = {x.item.item_id for x in res.items}
    assert {'M-ACC', 'M-SAF'} <= ids


def test_mandatory_items_are_never_displaced_by_the_cap(tmp_path):
    r = retriever(tmp_path, max_per_dimension=1)
    res = r.retrieve(query(case('severe chest pain shortness of breath fainting'), situations=['S:emerg:yes']))
    safety = [x.item.item_id for x in res.items if x.item.dimension == 'safety']
    assert safety == ['M-SAF'] and res.trace['dropped_by_cap'] == 1


def test_one_situation_per_exclusive_group(tmp_path):
    text = 'severe chest pain shortness of breath, also mild cold runny nose minor rash'
    r = retriever(tmp_path, top_situations=3, situation_policy='retrieved')
    res = r.retrieve(query(case(text)))
    chosen = res.trace['applied_situations']
    groups = [r.situations[s].exclusive_group for s in chosen]
    assert len(groups) == len(set(groups))
    assert not ({'I-EY', 'I-EN'} <= {x.item.item_id for x in res.items})


@pytest.mark.parametrize('mode', ['bm25', 'dense', 'hybrid'])
def test_each_mode_finds_the_clear_situation(tmp_path, mode):
    r = retriever(tmp_path, mode=mode)
    res = r.retrieve(query(case('Sudden severe chest pain and shortness of breath, I feel like fainting')))
    assert res.trace['suggested_situations'] == ['S:emerg:yes'] and res.trace['mode'] == mode
    # default policy: a suggestion is recorded but NOT applied
    assert res.trace['applied_situations'] == [] and 'I-EY' not in {x.item.item_id for x in res.items}


def test_retrieved_policy_applies_the_suggestion(tmp_path):
    r = retriever(tmp_path, situation_policy='retrieved')
    res = r.retrieve(query(case('Sudden severe chest pain and shortness of breath, I feel like fainting')))
    assert res.trace['applied_situations'] == ['S:emerg:yes'] and 'I-EY' in {x.item.item_id for x in res.items}
    assert res.trace['situation_source'] == 'retrieval (experimental)'


def test_metadata_situation_wins_over_the_suggestion(tmp_path):
    r = retriever(tmp_path)
    res = r.retrieve(query(case('Sudden severe chest pain and shortness of breath'), situations=['S:emerg:no']))
    ids = {x.item.item_id for x in res.items}
    assert 'I-EN' in ids and 'I-EY' not in ids
    assert res.trace['applied_situations'] == ['S:emerg:no'] and res.trace['suggested_situations'] == ['S:emerg:yes']
    assert res.trace['situation_source'] == 'metadata'


def test_unknown_metadata_situation_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='unknown situations'):
        retriever(tmp_path).retrieve(query(case('anything'), situations=['S:nope']))


def test_answer_and_task_never_reach_the_query(tmp_path):
    r = retriever(tmp_path)
    plain = case('mild cold and runny nose for two days')
    loaded = case('mild cold and runny nose for two days',
                  answer='EMERGENCY! severe chest pain shortness of breath fainting stroke, call now',
                  task='doctor nurse ward patient dosing')
    q1, q2 = query(plain), query(loaded)
    assert q1.text == q2.text and 'EMERGENCY' not in q2.text and 'dosing' not in q2.text
    assert r.retrieve(q1).trace['suggested_situations'] == r.retrieve(q2).trace['suggested_situations']
    assert r.retrieve(q2).trace['answer_used_in_query'] is False


@pytest.mark.parametrize('text', ['"quoted" NEAR(a b) AND OR NOT * ^ : -x', '???', '', '病人 胸痛 呼吸困难'])
def test_fts_syntax_in_the_conversation_is_inert(tmp_path, text):
    r = retriever(tmp_path)
    res = r.retrieve(query(case(text or ' ')))
    assert {'M-ACC', 'M-SAF'} <= {x.item.item_id for x in res.items}


def test_query_modes():
    c = CaseInput(case_id='C', group_id='G', task='t', candidate_output='a',
                  context=[Message(role='user', content='first question'), Message(role='assistant', content='reply'),
                           Message(role='user', content='follow up')])
    q = query(c)
    assert query_text(q, 'last_user') == 'follow up'
    assert query_text(q, 'user_turns') == 'first question\nfollow up'
    assert 'assistant: reply' in query_text(q, 'full')


def test_filter_retriever_applies_only_metadata_situations():
    res = FilterRetriever(mini_kb()).retrieve(query(case('severe chest pain')))
    assert {x.item.item_id for x in res.items} == {'M-ACC', 'M-SAF'}
    res = FilterRetriever(mini_kb()).retrieve(query(case('severe chest pain'), situations=['S:emerg:yes']))
    assert {x.item.item_id for x in res.items} == {'M-ACC', 'M-SAF', 'I-EY'}


def test_cached_embedder_matches_base():
    base = HashEmbedder()
    cached = CachedEmbedder(base)
    cached.warm(['a b c', 'd e'])
    assert cached.embed(['a b c', 'x y'], 'query') == base.embed(['a b c', 'x y'], 'query')


def test_tokens_drop_stopwords_and_roles():
    assert tokens('User: I have the chest pain, ok?') == ['chest', 'pain', 'ok']


def test_shipped_kb_trace_and_selection_with_hash_embedder(tmp_path):
    kb = load_kb(KB_DIR)
    emb = HashEmbedder()
    r = HybridRetriever(kb, open_index(kb, emb, tmp_path), emb)
    res = r.retrieve(query(case('Please write a discharge summary clinical note for my patient')))
    t = res.trace
    assert t['retriever'] == 'hybrid-v1' and len(t['suggested_situations']) <= 1 and t['index_manifest_sha256']
    assert t['applied_situations'] == [] and not any(x.item.situational for x in res.items)
    assert {i.item_id for i in kb.items if i.mandatory} <= {x.item.item_id for x in res.items}


MODEL_CACHE = ROOT / '.cache' / 'fastembed'


@pytest.mark.skipif(not (MODEL_CACHE / 'models--Qdrant--bge-small-en-v1.5-onnx-Q').exists(),
                    reason='local embedding model not present')
def test_real_local_model_ranks_an_obvious_emergency(tmp_path):
    kb = load_kb(KB_DIR)
    emb = FastEmbedEmbedder('BAAI/bge-small-en-v1.5', MODEL_CACHE)
    r = HybridRetriever(kb, open_index(kb, emb, tmp_path), emb, mode='dense')
    res = r.retrieve(query(case('My father suddenly cannot move his right arm and his speech is slurred. What do I do?')))
    assert res.trace['suggested_situations'] == ['HB:emergency_referrals:emergent']
    assert emb.model_id.startswith('BAAI/bge-small-en-v1.5@')


# ---- CLI end to end -------------------------------------------------------------------------------------------

def test_cli_runs_the_chain_with_hybrid_retrieval(tmp_path, capsys):
    shutil.copytree(KB_DIR, tmp_path / 'kb/v3')
    shutil.copytree(ROOT / 'examples/v3', tmp_path / 'examples/v3')
    cli = load_script('run_review_v3')
    assert cli.main(['--root', str(tmp_path), '--case', 'examples/v3/demo-case.json', '--embedder', 'hash',
                     '--job-id', 'HY1']) == 0
    out = capsys.readouterr().out
    assert 'AwaitingHumanReview' in out
    assert list((tmp_path / 'data/v3-index').glob('*/manifest.json'))
    pending = json.loads(out[out.rindex('{\n  "status"'):])['review_items']
    (tmp_path / 'd.json').write_text(json.dumps({r['dimension']: {'decision': 'accept', 'reviewer': 'LTF'}
                                                 for r in pending}))
    cli2 = load_script('run_review_v3')
    assert cli2.main(['--root', str(tmp_path), '--embedder', 'hash', '--resume', 'HY1', '--decisions',
                      str(tmp_path / 'd.json')]) == 0
    report = json.loads((tmp_path / 'reports/v3-reviews/HY1/report.json').read_text())
    trace = report['snapshot']['retrieval_trace']
    assert trace['retriever'] == 'hybrid-v1' and trace['answer_used_in_query'] is False
    assert 'Applied situations (metadata)' in (tmp_path / 'reports/v3-reviews/HY1/report.html').read_text()
