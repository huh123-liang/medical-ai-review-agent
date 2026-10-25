"""Prepare document-only independent-review packets. No runtime imports or API calls.

Writes only a new iteration-10 review directory. --verify is read-only.
Original case text is allowlist-exported; labels/old judgments never enter packets.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import socket
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/iteration-10-review-pack'
AUDIT = ROOT / 'reports/iteration-9-source-audit'
CASES = ROOT / 'reports/iteration-8-offline/cases.json'
SELECTED = ['fixed-38', 'fixed-62', 'fixed-63', 'fixed-67', 'new-03', 'new-11', 'new-22']
SEED = 102025
DOCS = ['REVIEW-SCORING-SPEC-V1.md', 'REVIEW-EVIDENCE-TEMPLATE-V1.md']

# Source selection is a source-assisted second stage, NOT an unbiased search.
SOURCE_ASSIGNMENT = {
    'fixed-38': ['nice-observations', 'nice-delirium'],
    'fixed-62': ['aripiprazole-label'],
    'fixed-63': [],
    'fixed-67': ['nhs-lumbar-puncture'],
    'new-03': ['pmid-29232171'],
    'new-11': ['nice-epilepsy'],
    'new-22': ['pmid-29229594', 'pmid-33827139'],
}
SOURCE_DESCRIPTIONS = {
    'nice-observations': ('NICE CG50：成人住院观察', 'Recommendations 1.2；观察项目、意识状态及成人急性住院适用范围。', '指南章节；不得自动推断未提供的年龄、基线或病区。'),
    'nice-delirium': ('NICE CG103：谵妄识别与评估', 'Recommendations 中 Indicators / Assessment and diagnosis；注意变化的时间、背景及适用范围。', '指南章节；原病例缺少条件时，不据此直接作诊断。'),
    'aripiprazole-label': ('DailyMed：ABILIFY 药品标签', '8.7 Hepatic and Renal Impairment；并核对页面剂型、剂量章节与日期。', '口服标签快照；不推广到所有剂型或未说明的个体化情境。'),
    'nhs-lumbar-puncture': ('NHS：腰椎穿刺患者信息', 'After / recovering from lumbar puncture，以及紧急求助部分。', '患者信息页，不是全部专科操作指南；需判断适用人群与任务。'),
    'nice-epilepsy': ('NICE NG217：癫痫诊断与评估', '1.2.6；同时阅读相邻的 EEG 建议及适用条件。', '指南局部证据不能证明整份信件或全部医学断言均正确。'),
    'pmid-29232171': ('ASCO/SSO 2018：黑色素瘤区域淋巴结指南', 'PubMed Abstract / Recommendations；核对厚度、溃疡与分期条件及机构归属。', '本地仅存元数据与摘要，非全文；特殊因素不能据摘要穷尽。'),
    'pmid-29229594': ('BV + nivolumab：2018 早期研究报告', 'PubMed 元数据与摘要；分别核对期刊、阶段、入组人群、样本量和终点。', '元数据与摘要，非全文；不要混用不同随访时点或治疗线别。'),
    'pmid-33827139': ('BV + nivolumab：2021 三年随访报告', 'PubMed 元数据与摘要；与早期报告分开记录研究人群、终点与时间点。', '元数据与摘要，非全文；不能据此确认所有其他试验或监管断言。'),
}

STAGE1_README = """# 阶段1：独立初判

先阅读 SCORING-SPEC.md，再按编号查看 cases/ 内7份材料，填写每份 forms/ 中的空白表。review.html 是同一批原文的便读版，JSON 是可核对原文；均不依赖在线资源。原英文未翻译、删节或改写。

不要先看阶段2、协调员目录、原基准标签或以往项目报告。不要调用模型、联网查标签或与另一位评审者交流判分。未知项可以暂缓并列出需查资料；不能猜造。若确需提前外查或认出案例，记录暴露，不能再称该份为未辅助盲判。

先复制表格到自己的工作目录，填写 reviewer_id、专业范围、是否有先前暴露，独立完成。表中所有结论留空，必须由评审者选择。不要修改原始材料；遇到准则含糊、错字或资料缺失，原样保留并记录。

逐项核对原准则的义务、回答位置和解释，不能只填总结果。阶段1先冻结：由协调员保存原表、完成时间及SHA256，之后另建阶段2文件，不能覆盖。

本包不含模型或医生的历史判分。这是研究用材料，不是临床建议，不是经过专家确认的标准答案。部分上下文较长，必须查看完整历史。
"""

STAGE2_README = """# 阶段2：来源辅助复判（初判冻结后才发放）

只在阶段1已独立完成并保存哈希后使用本包。先阅读 sources/ 来源卡，再按对应编号查看 evidence/。卡片提供有限阅读入口与适用范围，没有给出应判何分。

引用的是本项目已有的公开来源记录，未发起新检索或付费调用。链接用于后续人工查阅；生成材料时未重新验证网页现状。抓取时间与来源版本不是同一个概念，哈希只证明确认过的本地快照未变。摘要不是全文，卡片也不是完整医学事实库。

本阶段受研究者选材影响，不是完全无提示的盲审或独立检索实验。欢迎补充反证或更适用的公开来源，记录版本、日期、章节及适用条件。不能仅依来源标题作结论。无预选外部来源不代表事实均正确或无需审查。

使用 forms/ 的新表，保留阶段1原表，填写其哈希、本次判断、是否改变及原因。不得覆盖初判或原医生标签。未决问题允许保留，不强迫达成共识。
"""

COORDINATOR = """# 协调员说明：不要把整个目录发给评审者

本轮交付是规范和材料，不是已经完成盲审。尚未邀请、指定或联系医生；未产生新医学标准答案、模型预测或一致性指标。

## 发放顺序

1. 先确认两位具有相关专业能力的独立评审者，记录专业、利益冲突、既往暴露。看过本项目旧审计/标签的人员不能算独立盲审者；必要时按专科分配。不要向其透露原模型、医生标签或研究建议。
2. 只发 `01-stage1-blind.zip`，不要发根目录、生成脚本、旧报告或 coordinator-only/。评审顺序已固定随机化；两人独立完成。表格是空白，不包含推荐答案。
3. 收回阶段1每人每例的原始表后，先按原样保存、登记时间并计算 SHA256。登记表见 coordinator-only/assignment-log.json；本轮未代填人员或伪造冻结记录。评审在副本里填写，不改分发原件。
4. 冻结后再发 `02-stage2-sources.zip`。复判是来源辅助，不能称完全盲审；允许增补来源，但须记录版本、时间、适用性与修改原因。
5. 收回复判后保留初判、复判和分歧；需要时请第三位专业人员/准则负责人裁定。记录无法裁定的项目，不用多数票抹除事实争议。
6. 只有独立判断全部冻结后，协调员才可另行与原标签、模型结论、旧研究建议对照。本轮私有映射不复制标签；需要对照时仍从冻结原记录读取，不覆盖任何原始结果。

目录隔离和ZIP不是访问权限或加密。持有整个项目的人可自行打开映射；本项目已在对话中讨论过部分案例，因此不能保证任何指定人员未曾看过。仅能承诺**已制成尽量降低锚定的材料**，不能承诺临床评审已独立或绝对盲法。

## 研究边界

这7例是从已见诊断案例中选择的未决项，不是随机抽样、未见验证集或完整覆盖所有错误的集合；不能据此报告总体准确率、Kappa提升或部署安全性。

本轮只新增 docs/ 的两份规范与本报告目录及生成脚本；运行核心、默认Prompt、数据库业务记录、向量库、基准标签和旧平台均不作为写入目标。不存在隐式把人工暂缓状态写回模型schema的步骤。

材料准备的工程核对：原对话/回答/准则逐字一致、仅白名单导出、历史标签/模型判断隔离、来源可追溯、分发ZIP成员检查、账本与核心哈希比较。通过这些检查只证明材料制作过程，不证明医学判分正确。

## 交付后必须暂停

下一步建议（等待用户审批）：先确认独立评审安排，形成准则边界裁定表；再设计单Agent逐项义务/证据候选格式与兼容性测试。未裁定的医学分歧保持NeedsReview，不按7个编号写死答案。未获批准不改核心、不调用付费模型、不集成旧平台。
"""

REFLECTION = """# 第1部分完成后的目标与架构反思

## 符合的部分

- 目标：先提高判断可解释、可复核程度，而非把诊断样本分数调高。原准则原样保留，严格事实性没有被改成“只查重大错误”。
- 架构：仍是单评估Agent、确定性校验/计分和独立审计；本轮只在审计层准备资料，不加Agent、不放宽工具权限、不改核心及原平台。
- 可复现：完整对话、回答、准则逐字导出；随机顺序固定；映射与历史判分隔离；来源使用已有快照和章节入口，不把存在来源当成结论正确。
- 验证诚实性：只完成材料核验，尚无独立临床裁定或新模型结果。人工unable_to_determine不改变基准二元协议。
- 未来平台：人工判分、模型结论、证据状态与复核状态继续分离，便于以后受控接入，不提前改现有数据结构。

## 仍然存在的风险与信息缺口

- 本批是已见且挑选的分歧案例，不可证明总体泛化能力。
- 来源卡由研究者选材，可能遗漏相反资料；阶段2不是完全盲审。目录隔离也不是访问控制。
- 专业评审者尚未确定；一人难以覆盖所有专科，第三人裁定能力也需确认。已有讨论暴露者不具独立盲审条件。
- 一些准则原文极严格或范围含糊，不能靠程序或本轮文字强行消除歧义；部分历史病例缺少适用日期、受众与个人条件。
- 当前候选结果仍需复核。本轮没有证明评分提升，也没有通过临床安全门。

## 判断

**方向符合既定目标和单Agent架构；第1部分达到“规范及盲审材料准备完成”，尚未达到“医学边界经专家确认”。** 此时不宜直接进入付费对照测试或宣称修复判分。

## 下一步建议，等待审批

1. 用户确认独立评审者安排/是否先由准则负责人裁定解释范围；按两阶段流程收集反馈，未能安排时保留全部未决。
2. 经批准后，把无争议的通用义务拆解、任务证据与来源适用性要求设计成候选输出及离线测试；不将研究建议硬编码成病例答案，不在未知项上强制二元放行。
3. 再提交候选改动范围、固定回归与未见样本方案及预算；只有再次批准后才实施核心变动或新付费实验。累计上限仍为30元。

本轮到此停止，等待用户审批；不自动开始上述工作。
"""


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_text(p, text):
    assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')


def write_json(p, value):
    write_text(p, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def ledger():
    with sqlite3.connect(f'file:{(ROOT / "data/jury.sqlite").as_posix()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        costs = [dict(r) for r in db.execute('SELECT * FROM costs ORDER BY id')]
        return {'jobs': db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],
                'cost_records': len(costs), 'costs_sha256': digest(costs),
                'accounted_cny': round(sum(r['accounted'] for r in costs), 6),
                'reserved': sum(r['state'] == 'reserved' for r in costs)}


def protected_hashes():
    # No secret values are read into reports. No source or index file is written.
    paths = set()
    for folder in ['src/medical_jury', 'tests', 'data/prepared', 'data/facts', 'data/vector']:
        paths.update(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ['.env.local', 'pyproject.toml', 'data/private/physician-labels.jsonl',
                 'data/rubric-candidates-for-review.jsonl', 'reports/iteration-8-offline/cases.json']:
        paths.add(ROOT / name)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if p.is_file()}


def case_payload(c, review_id):
    return {'review_id': review_id,
            'context': [{'role': turn['role'], 'content': turn['content']} for turn in c['case']['context']],
            'candidate_output': c['case']['candidate_output'], 'rubric_text': c['rubric']['text']}


def blank_form(review_id, stage):
    return {'review_id': review_id, 'protocol_version': 'review-scoring-v1.0', 'stage': stage,
            'reviewer_id': None, 'specialty_experience': None, 'prior_exposure': None,
            'conflicts_or_scope_limits': None, 'started_at': None, 'completed_at': None,
            'stage1_frozen_sha256': None, 'task_and_audience': None, 'task_evidence_locations': [],
            'rubric_interpretation_questions': [], 'claims': [], 'obligations': [],
            'criterion_decision': None, 'decision_options': ['met', 'not_met', 'unable_to_determine'],
            'decisive_obligation_ids': [], 'rationale': None, 'coverage_and_unresolved': None,
            'confidence_level': None, 'confidence_reason': None, 'needs_professional_review': None,
            'sources_consulted': [], 'stage2_changed_decision': None, 'change_reason': None,
            'adjudication_notes': None}


def printable(title, sections):
    # Escaped preformatted originals: no markdown execution, scripts, images, or remote assets.
    body = ''.join('<section><h2>' + html.escape(name) + '</h2><pre>' + html.escape(text) + '</pre></section>'
                   for name, text in sections)
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#15283d;max-width:1000px;margin:32px auto;padding:0 24px}h1,h2{line-height:1.3}section{border-top:1px solid #ccd5df;padding-top:20px;margin-top:28px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.65 system-ui,sans-serif} @media print{body{margin:0;max-width:none}section{break-before:page}h2{break-after:avoid}pre{orphans:3;widows:3}}</style><h1>' + html.escape(title) + '</h1><p>研究评审材料；完整原文，无预填结论。以同包JSON原文核对。打印请确认所有页面。</p>' + body + '</html>'


def material_text(p):
    sections = ['# ' + p['review_id'], '\n## 原始 Rubric\n\n' + p['rubric_text'], '\n## 完整对话（按时间顺序）']
    for n, turn in enumerate(p['context'], 1):
        fence = '`' * max(4, max((len(s) + 1 for s in re.findall(r'`+', turn['content'])), default=4))
        sections.append(f"\n### Turn {n} · {turn['role']}\n\n{fence}text\n{turn['content']}\n{fence}")
    text = p['candidate_output']
    fence = '`' * max(4, max((len(s) + 1 for s in re.findall(r'`+', text)), default=4))
    sections.append('\n## 待评回答（只评这一份）\n\n' + fence + 'text\n' + text + '\n' + fence)
    return '\n'.join(sections) + '\n'


def build_sources():
    manifest = {r['id']: r for r in read(AUDIT / 'source-manifest.json')['records']}
    papers = {r['id']: r for r in read(AUDIT / 'primary-publications.json')}
    result = {}
    for sid, (name, locator, limit) in SOURCE_DESCRIPTIONS.items():
        r = papers[sid] if sid in papers else manifest[sid]
        raw = r.get('source_snapshot') or r['raw_file']
        expected = r.get('source_sha256') or r['sha256']
        assert sha(ROOT / raw) == expected, sid
        if r.get('text_file'):
            assert sha(ROOT / r['text_file']) == r['text_sha256']
        result[sid] = {'source_id': sid, 'name': name, 'url': r['url'], 'reading_locator': locator,
                       'limits': limit, 'snapshot_sha256': expected, 'snapshot_kind': r['status'],
                       'captured_at': r.get('retrieved_at', '既有快照；本轮未重新抓取'),
                       'snapshot_file_coordinator_only': raw,
                       'publication_date_if_available': r.get('publication_date'), 'doi': r.get('doi', [])}
    return result


def zip_packet(stage, zipname):
    with zipfile.ZipFile(OUT / zipname, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())


def check_materials(freeze):
    source_cases = {c['audit_id']: c for c in read(CASES)}
    mapping = read(OUT / 'coordinator-only/case-map.json')
    assert len(mapping) == 7 and {r['audit_id'] for r in mapping} == set(SELECTED)
    stage1 = OUT / 'reviewer-stage-1'
    stage2 = OUT / 'reviewer-stage-2'
    all_old_ids = [str(c['case'][k]) for c in source_cases.values() for k in ['case_id', 'group_id']]
    all_job_ids = [a['job_id'] for c in source_cases.values() for a in c['arms'].values()]
    banned_keys = ['original_physician_labels', 'model_verdict', 'original_status', 'root_cause',
                   'selection_reason', 'original_line', 'original_error', 'job_id', 'example_id']
    for r in mapping:
        rid, c = r['review_id'], source_cases[r['audit_id']]
        payload = read(stage1 / f'cases/{rid}.json')
        assert payload == case_payload(c, rid), rid
        assert set(payload) == {'review_id', 'context', 'candidate_output', 'rubric_text'}
        assert all(set(t) == {'role', 'content'} for t in payload['context'])
        md = (stage1 / f'cases/{rid}.md').read_text(encoding='utf-8')
        h = (stage1 / 'review.html').read_text(encoding='utf-8')
        for original in [t['content'] for t in payload['context']] + [payload['candidate_output'], payload['rubric_text']]:
            assert original in md and html.escape(original) in h, rid
        for stage in [stage1, stage2]:
            form = read(stage / f'forms/{rid}.json')
            assert form['criterion_decision'] is None and form['prior_exposure'] is None
            assert not form['claims'] and not form['obligations'] and not form['sources_consulted']
    for stage in [stage1, stage2]:
        for p in stage.rglob('*'):
            if not p.is_file():
                continue
            text = p.read_text(encoding='utf-8')
            for marker in SELECTED + all_old_ids + all_job_ids + banned_keys:
                assert marker not in text, (str(p), marker)
            assert not re.search(r'sk-[A-Za-z0-9]{20,}', text), str(p)
        for p in stage.glob('*.html'):
            rendered = p.read_text(encoding='utf-8').lower()
            assert '<script' not in rendered and '<iframe' not in rendered and '<img' not in rendered
    for stage, name in [(stage1, '01-stage1-blind.zip'), (stage2, '02-stage2-sources.zip')]:
        expected = {p.relative_to(stage).as_posix(): sha(p) for p in stage.rglob('*') if p.is_file()}
        with zipfile.ZipFile(OUT / name) as z:
            assert set(z.namelist()) == set(expected) and z.testzip() is None
            assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
            for member, expected_sha in expected.items():
                assert hashlib.sha256(z.read(member)).hexdigest() == expected_sha
    assert protected_hashes() == freeze['protected_files_before'], 'Protected project files changed.'
    current = ledger()
    assert current == freeze['ledger_before'], 'Financial state changed.'
    prior = read(AUDIT / 'scope-freeze.json')
    assert {p.name: sha(p) for p in (ROOT / 'src/medical_jury').glob('*.py')} == prior['core_before']
    for sid, r in build_sources().items():
        assert r == read(OUT / f'coordinator-only/source-provenance.json')[sid]
    return {'status': 'REVIEW_MATERIALS_VERIFIED_NOT_CLINICAL_ADJUDICATION',
            'verified_at': datetime.now(timezone.utc).isoformat(), 'case_count': 7,
            'exact_original_context_candidate_rubric': True, 'original_label_and_judgment_exported': False,
            'stage1_label_model_id_and_old_id_checks': 'passed', 'blank_decision_forms': 14,
            'packet_member_and_byte_checks': 'passed', 'source_records_verified': 8,
            'runtime_core_unchanged_since_iteration9': True, 'protected_files_unchanged': True,
            'protected_file_count': len(freeze['protected_files_before']), 'financial_state': current,
            'new_paid_calls': 0, 'new_model_predictions': 0, 'independent_reviews_received': 0,
            'new_clinical_gold_labels': 0, 'original_platform_accessed_by_generator': False,
            'gate': 'STOP_AND_REQUEST_USER_APPROVAL',
            'limits': ['This checks packet construction, not medical judgment or reviewer independence.',
                       'These seven are seen diagnostic cases, not a new validation set.',
                       'Source selection can bias stage2; no automatic source refresh or new retrieval.',
                       'Folder/ZIP separation is not an access-control guarantee.']}


def main(verify_only=False):
    if verify_only:
        result = check_materials(read(OUT / 'coordinator-only/scope-freeze.json'))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    assert not OUT.exists(), 'Refuse to overwrite review materials; use --verify or a new version.'
    freeze = {'created_at': datetime.now(timezone.utc).isoformat(), 'ledger_before': ledger(),
              'protected_files_before': protected_hashes(), 'source_cases_sha256': sha(CASES),
              'random_order_seed': SEED, 'label_export_allowed': False, 'paid_calls_allowed': False}
    assert freeze['ledger_before'] == read(AUDIT / 'verification.json')['financial_state_unchanged']
    assert sha(CASES) == read(AUDIT / 'scope-freeze.json')['same_cases_sha256']
    OUT.mkdir()
    stage1, stage2 = OUT / 'reviewer-stage-1', OUT / 'reviewer-stage-2'
    write_text(OUT / '开始这里-协调员.md', COORDINATOR)
    write_text(OUT / '目标架构反思与下一步.md', REFLECTION)
    write_json(OUT / 'coordinator-only/scope-freeze.json', freeze)
    write_text(stage1 / 'README.md', STAGE1_README)
    write_text(stage2 / 'README.md', STAGE2_README)
    for stage in [stage1, stage2]:
        write_text(stage / 'SCORING-SPEC.md', (ROOT / 'docs' / DOCS[0]).read_text(encoding='utf-8'))
        write_text(stage / 'EVIDENCE-TEMPLATE.md', (ROOT / 'docs' / DOCS[1]).read_text(encoding='utf-8'))
        write_json(stage / 'evidence-row-shape.json', {
            '_note': '人工评审空白结构示意，不是已填写证据，不是运行wire schema。每项义务、主张和来源可增行。',
            'claim_shape': {'claim_id': None, 'quote': None, 'text_field': None, 'context_turn_1based': None,
                            'start_char_0based': None, 'end_char_exclusive': None, 'occurrence_1based': None,
                            'claim_type': None, 'conditions_units_time': None, 'verification_status': None},
            'obligation_shape': {'obligation_id': None, 'rubric_clause': None, 'claim_ids': [],
                                 'answer_context_locations': [], 'missing_item_checked_scope': None,
                                 'source_links': [], 'obligation_status': None, 'applicability': None,
                                 'reason': None, 'decisive': None, 'unresolved': None},
            'source_link_shape': {'source_id': None, 'url_or_document': None, 'version': None,
                                  'retrieved_at': None, 'section_or_paragraph': None, 'quote_or_paraphrase': None,
                                  'is_verbatim': None, 'source_applicability': None, 'relation': None,
                                  'relation_reason': None, 'claim_ids': []}})
    rows = {c['audit_id']: c for c in read(CASES)}
    order = list(SELECTED)
    random.Random(SEED).shuffle(order)
    mapping, sections = [], []
    template = (ROOT / 'docs' / DOCS[1]).read_text(encoding='utf-8')
    sources = build_sources()
    for n, audit_id in enumerate(order, 1):
        rid = f'REVIEW-{n:02d}'
        p = case_payload(rows[audit_id], rid)
        mapping.append({'review_id': rid, 'audit_id': audit_id, 'source_payload_sha256': digest(p),
                        'context_turns': len(p['context']), 'assigned_source_ids': SOURCE_ASSIGNMENT[audit_id]})
        write_json(stage1 / f'cases/{rid}.json', p)
        write_text(stage1 / f'cases/{rid}.md', material_text(p))
        original_text = '原始 Rubric\n' + p['rubric_text'] + '\n\n完整对话\n'
        original_text += '\n\n'.join(f"Turn {i} · {t['role']}\n{t['content']}" for i, t in enumerate(p['context'], 1))
        original_text += '\n\n待评回答\n' + p['candidate_output']
        sections.append((rid, original_text))
        for stage_num, stage in [(1, stage1), (2, stage2)]:
            write_json(stage / f'forms/{rid}.json', blank_form(rid, stage_num))
            write_text(stage / f'forms/{rid}.md', f'# {rid} · 阶段{stage_num}空白评审表\n\n' + template)
        assigned = SOURCE_ASSIGNMENT[audit_id]
        links = '\n'.join(f'- [{sources[s]["name"]}](../sources/{s}.md)' for s in assigned)
        if not links:
            links = '本例未预选外部来源。请逐项对照阶段1原上下文；如有医学断言需核查，可补充来源并记录。未预选不意味着回答事实正确。'
        write_text(stage2 / f'evidence/{rid}.md', f'# {rid} · 来源辅助复判\n\n阶段1必须已冻结，勿覆盖初判。\n\n{links}\n\n逐项填写：该来源是否适用？支持/反驳/仍不足？影响哪条原准则义务？哪些主张仍未解决？本卡不提供应判结论。\n')
    write_text(stage1 / 'review.html', printable('阶段1 · 独立评审原文', sections))
    source_sections = []
    for sid, r in sources.items():
        card = f'# {r["name"]}\n\n- Source ID：{sid}\n- 原始入口：[打开公开来源]({r["url"]})\n- 阅读定位：{r["reading_locator"]}\n- 材料范围：{r["snapshot_kind"]}\n- 抓取时间：{r["captured_at"]}\n- 本地快照 SHA256：{r["snapshot_sha256"]}\n- 限制：{r["limits"]}\n\n本卡没有给出判分或摘录全文。请记录实际阅读章节、适用条件、支持关系与仍未知项。当前网页可能已更新，不得冒充本地冻结版本。\n'
        if r['doi']:
            card += '\nDOI：' + ', '.join(r['doi']) + '\n'
        write_text(stage2 / f'sources/{sid}.md', card)
        source_sections.append((r['name'], card))
    write_text(stage2 / 'sources.html', printable('阶段2 · 有限来源阅读入口', source_sections))
    write_json(OUT / 'coordinator-only/case-map.json', mapping)
    write_json(OUT / 'coordinator-only/source-provenance.json', sources)
    write_json(OUT / 'coordinator-only/assignment-log.json', {
        'status': 'NOT_ASSIGNED_NO_REVIEWS_RECEIVED', 'reviewers': [],
        'reviewer_row_shape': {'reviewer_id': None, 'specialty': None, 'conflicts': None, 'prior_exposure': None},
        'freezes': [], 'freeze_row_shape': {'review_id': None, 'reviewer_id': None, 'stage': None,
                                          'received_at': None, 'file_path': None, 'sha256': None},
        'adjudications': [], 'adjudication_row_shape': {'review_id': None, 'adjudicator_id': None,
                                                      'decision': None, 'reason': None, 'unresolved': None}})
    zip_packet(stage1, '01-stage1-blind.zip')
    zip_packet(stage2, '02-stage2-sources.zip')
    result = check_materials(freeze)
    write_json(OUT / 'coordinator-only/verification.json', result)
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write_json(OUT / 'coordinator-only/delivery-manifest.json', {
        'files': files, 'manifest_excludes_itself': True,
        'spec_files': {name: sha(ROOT / 'docs' / name) for name in DOCS},
        'generator_sha256': sha(Path(__file__))})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    def no_network(*args, **kwargs):
        raise AssertionError('Network disabled during document preparation.')
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
        main(args.verify)
