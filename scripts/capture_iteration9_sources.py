"""Operator-run primary-source snapshots for the SAME 12-case audit; no LLM/API fees."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from prepare_iteration8_audit import ROOT,ledger,read,write,sha

OUT=ROOT/'reports/iteration-9-source-audit'
PMIDS=['32417976','30292760','26363333','29723512','29232171','29229594','33827139']
PAGES=[
 ('fda-food-allergy','https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/food-allergies','stop eating'),
 ('gmc-records','https://www.gmc-uk.org/professional-standards/the-professional-standards/good-medical-practice/domain-3-colleagues-culture-and-safety','contemporaneous'),
 ('nhs-lumbar-puncture','https://www.nhs.uk/tests-and-treatments/lumbar-puncture/','severe pain'),
 ('nice-epilepsy','https://www.nice.org.uk/guidance/ng217/chapter/diagnosis-and-assessment-of-epilepsy','Do not use EEG to exclude'),
 ('nice-wells','https://www.nice.org.uk/guidance/ng158/chapter/recommendations','alternative diagnosis'),
 ('nice-delirium','https://www.nice.org.uk/guidance/cg103/chapter/recommendations','confusion'),
 ('nice-observations','https://www.nice.org.uk/guidance/cg50/chapter/recommendations','level of consciousness'),
 ('fda-byfavo','https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/212295s003lbl.pdf',None),
]
HOSTS={'www.fda.gov','www.gmc-uk.org','www.nhs.uk','www.nice.org.uk','www.accessdata.fda.gov','eutils.ncbi.nlm.nih.gov'}


class Parser(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in {'script','style'}:self.skip+=1
    def handle_endtag(self,tag):
        if tag in {'script','style'} and self.skip:self.skip-=1
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def check_url(url):
    p=urlparse(url)
    if p.scheme!='https' or p.hostname not in HOSTS or p.username or p.password:
        raise ValueError('Source URL/redirect outside operator allowlist.')


def fetch(row):
    sid,url,marker=row;check_url(url)
    result={'id':sid,'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat()}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'MedicalJuryResearchAudit/1.0 (public-source-verification)'})
        with urllib.request.build_opener(Redirect()).open(req,timeout=25) as response:
            check_url(response.url);raw=response.read(4*1024*1024+1)
            if len(raw)>4*1024*1024:raise ValueError('Source exceeds bounded snapshot limit.')
            result['resolved_url']=response.url
        suffix='.xml' if sid=='pubmed-primary-records' else '.pdf' if raw.startswith(b'%PDF') else '.html'
        path=OUT/'sources'/(sid+suffix);path.write_bytes(raw)
        result.update(raw_file=str(path.relative_to(ROOT)),sha256=sha(path),bytes=len(raw))
        if suffix=='.html':
            parser=Parser();parser.feed(raw.decode('utf-8',errors='replace'));text=' '.join(' '.join(parser.parts).split())
            textpath=path.with_suffix('.txt');textpath.write_text(text,encoding='utf-8')
            result['text_file']=str(textpath.relative_to(ROOT));result['text_sha256']=sha(textpath)
            if marker and marker.casefold() not in text.casefold():
                result['status']='CAPTURED_BUT_EXPECTED_CONTENT_NOT_CONFIRMED';return result
        result['status']='CAPTURED_PRIMARY_SOURCE'
    except (urllib.error.URLError,TimeoutError,ValueError,OSError) as e:
        result.update(status='UNAVAILABLE_NOT_VERIFIED',error_type=type(e).__name__)
        if isinstance(e,urllib.error.HTTPError):result['http_status']=e.code
    return result


def main():
    OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    before=read(ROOT/'reports/iteration-8-offline/verification.json')
    db=sqlite3.connect('file:'+(ROOT/'data/jury.sqlite').as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert ledger(db)==before['primary_database_financial_state_unchanged'];db.close()
    pre={'same_cases_sha256':sha(ROOT/'reports/iteration-8-offline/cases.json'),
         'selected_case_count':12,'budget_before':before['primary_database_financial_state_unchanged'],
         'network_scope':'Fixed public medical URLs and publication IDs, no patient text, credentials or model requests.',
         'core_before':{p.name:sha(p) for p in (ROOT/'src/medical_jury').glob('*.py')},
         'case_or_label_changes_authorized':False,'paid_calls_authorized':False}
    write(OUT/'scope-freeze.json',pre)
    batch='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='+','.join(PMIDS)+'&retmode=xml'
    rows=PAGES+[('pubmed-primary-records',batch,None)]
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,rows))
    pub=next(r for r in results if r['id']=='pubmed-primary-records')
    articles=[]
    if pub['status']=='CAPTURED_PRIMARY_SOURCE':
        xml=ET.fromstring((ROOT/pub['raw_file']).read_bytes())
        for node in xml.findall('PubmedArticle'):
            pmid=node.findtext('./MedlineCitation/PMID')
            article=node.find('./MedlineCitation/Article')
            title=''.join(article.find('ArticleTitle').itertext())
            abstract=[''.join(a.itertext()) for a in article.findall('./Abstract/AbstractText')]
            articles.append({'id':'pmid-'+pmid,'url':'https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/',
                             'title':title,'journal':article.findtext('./Journal/Title'),
                             'volume':article.findtext('./Journal/JournalIssue/Volume'),
                             'issue':article.findtext('./Journal/JournalIssue/Issue'),
                             'pages':article.findtext('./Pagination/StartPage') or article.findtext('./Pagination/MedlinePgn'),
                             'publication_date':ET.tostring(article.find('./Journal/JournalIssue/PubDate'),encoding='unicode'),
                             'abstract':abstract,'doi':[a.text for a in node.findall('./PubmedData/ArticleIdList/ArticleId') if a.attrib.get('IdType')=='doi'],
                             'source_snapshot':pub['raw_file'],'source_sha256':pub['sha256'],
                             'status':'PRIMARY_METADATA_AND_ABSTRACT_NOT_FULLTEXT'})
        assert {a['id'][5:] for a in articles}==set(PMIDS)
    write(OUT/'primary-publications.json',articles)
    old=ROOT/'data/facts/v1/F-arip-hepatic-v1.html'
    dest=OUT/'sources/aripiprazole-existing-snapshot.html';dest.write_bytes(old.read_bytes())
    assert sha(old)==sha(dest)
    results.append({'id':'aripiprazole-label','url':'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=c040bd1d-45b7-49f2-93ea-aed7220b30ac',
                    'raw_file':str(dest.relative_to(ROOT)),'sha256':sha(dest),
                    'status':'REUSED_UNCHANGED_REVIEWED_PRIMARY_SNAPSHOT',
                    'section':'8.7 Hepatic and Renal Impairment','snapshot_origin':'iteration7 source manifest'})
    write(OUT/'source-manifest.json',{'review_date_local':'2026-10-25','records':results,
        'limits':'Operator source review, not clinician adjudication. No source is automatically applicable to an undated historical case. Unavailable pages are not treated as captured originals.'})
    print(json.dumps({'sources':[{k:r.get(k) for k in ('id','status','http_status')} for r in results],
                      'publication_records':len(articles),'paid_calls':0},indent=2))


if __name__=='__main__':main()
