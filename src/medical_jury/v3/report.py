"""Self-contained HTML rendering of a v3 report. Every value is HTML-escaped (answers are untrusted)."""
from __future__ import annotations

from html import escape


def _mark(answer: str, spans: list[list[int]]) -> str:
    out, pos = [], 0
    for a, b in sorted({(int(a), int(b)) for a, b in spans}):
        if a < pos or b > len(answer):
            continue
        out.append(escape(answer[pos:a]))
        out.append(f'<mark>{escape(answer[a:b])}</mark>')
        pos = b
    out.append(escape(answer[pos:]))
    return ''.join(out)


def render_html(report: dict, case) -> str:
    spans = [s for d in report['dimensions'] for c in d.get('checks', []) for s in c.get('quote_spans', [])]
    rows = []
    for d in report['dimensions']:
        checks = ''.join(
            f"<li><code>{escape(c['id'])}</code> <b>{escape(c['result'])}</b>"
            f"{' — “' + escape(c['quote']) + '” (' + escape(c['quote_tier']) + ')' if c['quote'] else ''}</li>"
            for c in d.get('checks', []))
        human = d.get('human') or {}
        rows.append(
            f"<section><h3>{escape(d['label'])} <small>[{d['scale'][0]}–{d['scale'][1]}]</small></h3>"
            f"<p>Agent score: <b>{escape(str(d['agent_score']))}</b> · model score {escape(str(d['model_score']))} · "
            f"confidence {escape(str(d['confidence']))} · status {escape(d['status'])}"
            f"{' · failure ' + escape(str(d['failure_code'])) if d.get('failure_code') else ''}</p>"
            f"<p>Final: <b>{escape(str(d['final_score']))}</b> ({escape(d['final_status'])})"
            f"{' · human ' + escape(human.get('decision', '')) + ' by ' + escape(human.get('reviewer', '')) if human else ''}</p>"
            f"<p>Review reasons: {escape(', '.join(d.get('review_reasons', [])) or 'none')}</p>"
            f"<p>Rule adjustments: {escape(str(d.get('rule_adjustments') or 'none'))}</p>"
            f"<details><summary>Analysis and checks</summary><p>{escape(d.get('analysis') or '')}</p><ul>{checks}</ul>"
            f"<p>Reference notes used: {escape(', '.join(d.get('reference_ids', [])) or 'none')}</p></details></section>")
    items = ''.join(f"<li><code>{escape(r['item']['item_id'])}</code> [{escape(r['item']['level'])}/"
                    f"{escape(r['item']['role'])}/{escape(r['item']['dimension'])}] {escape(r['item']['text'][:200])}"
                    f" <small>— {escape(r['item']['source']['citation'])}</small></li>"
                    for r in report['retrieved_items'])
    trace = report.get('snapshot', {}).get('retrieval_trace', {})
    retrieval = (f"<p>Retriever {escape(str(trace.get('retriever')))} ({escape(str(trace.get('mode', 'filter')))}) · "
                 f"Applied situations ({escape(str(trace.get('situation_source', 'metadata')))}): "
                 f"{escape(', '.join(trace.get('applied_situations') or []) or 'none')}</p>")
    # Unapplied retrieval suggestions stay in report.json only: showing a known-unreliable suggestion to the
    # human reviewer would anchor their judgment.
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>Review {escape(report['job_id'])}</title><style>body{{font-family:system-ui,sans-serif;max-width:960px;"
        "margin:auto;padding:16px;line-height:1.5}section{border-top:1px solid #ccc;padding:8px 0}mark{background:#ffe08a}"
        "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px}</style></head><body>"
        f"<h1>Review report</h1><p>Job {escape(report['job_id'])} · case {escape(report['case_id'])} · profile "
        f"{escape(report['profile_id'])} · {escape(report['pipeline'])} · <b>{escape(report['clinical_validation'])}</b></p>"
        f"<p>Safety rule screen: {'review required' if report['rule_screen']['review_required'] else 'no findings'}</p>"
        f"<h2>Scores</h2>{''.join(rows)}<h2>Answer (quoted spans highlighted)</h2>"
        f"<pre>{_mark(case.candidate_output, spans)}</pre><h2>Retrieved rubric items</h2>{retrieval}<ul>{items}</ul>"
        f"<h2>Notes</h2><ul>{''.join('<li>' + escape(n) + '</li>' for n in report['notes'])}</ul></body></html>")
