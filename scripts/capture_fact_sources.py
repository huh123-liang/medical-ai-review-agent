"""Explicit one-time allowlisted public downloads; never called by the Agent."""
import hashlib
import json
import re
from datetime import date,timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse,urljoin
import httpx
from medical_jury.fact_sources import SPECS,VERSION
from medical_jury.store import timestamp

ROOT=Path(__file__).resolve().parents[1]


class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)


def main():
    folder=ROOT/'data/facts/v1';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'manifest.json'
    if target.exists():raise SystemExit('Source manifest already exists; do not overwrite reviewed history.')
    records=[]
    allowed={'dailymed.nlm.nih.gov','www.dailymed.nlm.nih.gov','www.cdc.gov'}
    with httpx.Client(timeout=45,follow_redirects=False,trust_env=False) as client:
        for spec in SPECS:
            url=spec['url']
            for _ in range(4):
                if urlparse(url).scheme!='https' or urlparse(url).hostname not in allowed:
                    raise ValueError('Source redirect left approved HTTPS hosts.')
                response=client.get(url)
                if response.is_redirect:
                    url=urljoin(url,response.headers['location']);continue
                response.raise_for_status();break
            else:raise ValueError('Too many redirects.')
            raw=response.content
            if len(raw)>3_000_000:raise ValueError('Source exceeds bounded snapshot size.')
            check='manual_web_pdf_text_review_2026-10-24'
            if spec['format']=='html':
                parser=Text();parser.feed(response.text)
                norm=lambda x:re.sub(r'\s+',' ',x).replace('’',"'").strip()
                if norm(spec['excerpt']) not in norm(' '.join(parser.parts)):
                    raise ValueError('Reviewed excerpt not found in the official HTML.')
                check='official_html_excerpt_substring_checked'
            elif not raw.startswith(b'%PDF'):
                raise ValueError('Expected CDC PDF snapshot.')
            name=spec['id']+'.'+spec['format']
            with (folder/name).open('xb') as f:f.write(raw)
            records.append({'id':spec['id'],'url':spec['url'],'resolved_url':url,'raw_file':name,
                            'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_at':timestamp(),
                            'section':spec['section'],'review_check':check})
            print(spec['id'],len(raw),check,flush=True)
    manifest={'version':VERSION,'reviewed_at':date.today().isoformat(),
              'review_until':(date.today()+timedelta(days=30)).isoformat(),
              'review_window_note':'Local re-review policy, not a claim that clinical guidance expires on this date.',
              'records':records}
    with target.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print('Four public source snapshots captured. No LLM calls, labels, vectors or old-platform data used.')


if __name__=='__main__':main()
