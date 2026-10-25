"""Narrow, reviewed source cards. Separate from rubrics, labels and vector indexes."""
from __future__ import annotations
import hashlib
import json
import re
from datetime import date
from urllib.parse import urlparse

VERSION='fact-cards-v1'
SPECS=[
    {'id':'F-arip-hepatic-v1','title':'DailyMed ABILIFY tablets, section 8.7',
     'url':'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=c040bd1d-45b7-49f2-93ea-aed7220b30ac',
     'section':'8.7 Hepatic and Renal Impairment','format':'html',
     'excerpt':"No dosage adjustment for ABILIFY is required on the basis of a patient's hepatic function",
     'summary':'For this oral aripiprazole label, hepatic function alone, including Child-Pugh 5-15, does not require dose adjustment. Other factors such as drug interactions have separate rules.',
     'scope':'US oral tablets; not a statement about every formulation or individual treatment decision.',
     'terms':[r'\b(?:aripiprazole|abilify)\b',r'\b(?:hepatic|cirrhosis|liver|child.pugh)\b'],
     'exclusions':[r'\b(?:maintena|asimtufii|long.acting injection)\b']},
    {'id':'F-fentanyl-metabolism-v1','title':'DailyMed fentanyl transdermal system, section 12.3',
     'url':'https://www.dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=bf02ffd0-cc5d-446b-b648-378f6885052f',
     'section':'12.3 Pharmacokinetics / Elimination / Metabolism','format':'html',
     'excerpt':'Fentanyl is metabolized primarily via human cytochrome P450 3A4 isoenzyme system.',
     'summary':'The label describes CYP3A4-mediated metabolism. Urinary elimination of metabolites is not the same as renal metabolism; this card does not establish a safe dose or preferred alternative in liver disease.',
     'scope':'Metabolism claim in transdermal-system label; no dosing recommendation.',
     'terms':[r'\b(?:fentanyl|fentanil)\b',r'\b(?:metaboli\w*|renal|hepatic|liver|cirrhosis|hépatique|rénal)\b'],
     'exclusions':[]},
    {'id':'F-lithium-ras-v1','title':'DailyMed lithium oral solution, section 7.1',
     'url':'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ff9baf45-830f-4a8a-8986-5caeaa1b38cf',
     'section':'7.1 Drug interactions / Renin-Angiotensin System Antagonists','format':'html',
     'excerpt':'Concomitant use increase steady-state serum lithium concentrations.',
     'summary':'Renin-angiotensin system antagonists can raise steady-state lithium concentrations. The label calls for more frequent lithium monitoring and adjustment based on concentration and response, not an interaction-free substitute claim.',
     'scope':'Lithium coadministration with renin-angiotensin system antagonists; not a blanket contraindication.',
     'terms':[r'\blithium\b',r'\b(?:ace inhibitors?|lisinopril|enalapril|captopril|valsartan|renin.angiotensin)\b'],
     'exclusions':[]},
    {'id':'F-lyme-age-v1','title':'CDC Lyme Disease Prophylaxis After Tick Bite',
     'url':'https://www.cdc.gov/lyme/resources/pdfs/lyme-pep-low-ink-p.pdf',
     'section':'Page 1, doxycycline safety decision and dose footnote','format':'pdf',
     'excerpt':'safe for people of all ages, including young children',
     'summary':'The CDC single-dose doxycycline prophylaxis aid includes children of any age when tick-bite and patient-safety conditions are met. Age alone is not the entire eligibility decision; allergy, pregnancy and lactation still need assessment.',
     'scope':'US high-risk Lyme tick-bite prophylaxis, not every tick bite, infection treatment or all antibiotics.',
     'terms':[r'\b(?:lyme|tick|ixodes)\b',r'\b(?:prophylaxis|prophylactic|prophylact|prevent|doxycycline)\w*\b'],
     'exclusions':[]},
]


class FactSourceError(ValueError):pass


def load_cards(root):
    folder=root/'data/facts/v1'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['version']!=VERSION or date.today()>date.fromisoformat(manifest['review_until']):
        raise FactSourceError('Fact source review missing or expired.')
    records={r['id']:r for r in manifest['records']}
    if len(records)!=len(manifest['records']) or set(records)!={s['id'] for s in SPECS}:
        raise FactSourceError('Fact source inventory mismatch.')
    cards=[]
    for spec in SPECS:
        record=records[spec['id']]
        name=record['raw_file']
        if name!=spec['id']+'.'+spec['format'] or '/' in name or '\\' in name or name in {'.','..'}:
            raise FactSourceError('Source path escaped its snapshot folder.')
        raw=(folder/name).read_bytes()
        if record['url']!=spec['url'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise FactSourceError('Reviewed source snapshot mismatch.')
        parsed=urlparse(spec['url'])
        if parsed.scheme!='https' or parsed.hostname not in {'dailymed.nlm.nih.gov','www.dailymed.nlm.nih.gov','www.cdc.gov'}:
            raise FactSourceError('Source host not allowlisted.')
        cards.append({**spec,'source_sha256':record['sha256'],'retrieved_at':record['retrieved_at'],
                      'reviewed_at':manifest['reviewed_at'],'source_version':'sha256:'+record['sha256'],
                      'review_status':'researcher_source_checked_not_clinician_approved',
                      'time_limit':'Current reviewed snapshot; benchmark historical date may be unknown. Do not silently apply retroactively.'})
    return cards


def retrieve_facts(case,root):
    corpus='\n'.join([m.content for m in case.context]+[case.candidate_output])
    selected=[]
    for card in load_cards(root):
        if all(re.search(pattern,corpus,re.I) for pattern in card['terms']) and not any(
            re.search(pattern,corpus,re.I) for pattern in card['exclusions']):
            selected.append({k:v for k,v in card.items() if k not in {'terms','exclusions','format'}})
    return {'version':VERSION,'cards':selected[:4],
            'coverage':'partial_topic_match_needs_scope_and_time_review' if selected else 'not_covered',
            'limits':'No source hit is not a falsehood verdict. Source hits do not verify the entire answer. '
                     'These are not physician labels, reference answers, rubric replacements or clinician-approved guidelines.'}


FACT_POLICY="""
Fact cards are separately sourced excerpts and research summaries, not rubrics or gold labels.
Check formulation, population, indication, jurisdiction and time before applying one. A hit alone
does not prove the selected claim or entire answer correct. Do not retroactively apply current
guidance to an explicitly historical question. Preserve original criterion and verdict independence.
Select only supplied source IDs for a decisive medical claim actually addressed by a card. Do not
invent URLs or claim external browsing. If a claim is outside coverage, use not_covered/uncertain,
not 'verified'. Inability to recall a guideline is not evidence it is false or fabricated.
Each fact_check must bind one selected answer ID to one supplied fact_id, record relation and
brief scope_note. No applicable source means an empty fact_checks list, not invented verification.
For a strict factuality rubric without adequate verification, set review_required=true and give
the required best binary research judgment. Do not automatically pass, reject, or change clinician labels.
"""
