"""Explicit addendum, preserving failed download attempts and TLS verification."""
from capture_iteration9_sources import HOSTS,fetch,OUT,write


def main():
    HOSTS.add('dailymed.nlm.nih.gov')
    record=fetch(('byfavo-dailymed-pdf',
        'https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid=4e1838f2-b999-41a7-a761-d94f09aa531f&type=pdf',None))
    write(OUT/'source-addendum.json',{'records':[record],
        'reason':'FDA direct download failed certificate verification; TLS checks were not disabled. Public DailyMed PDF is separately retrieved and identified, not claimed to be the same file as FDA PDF.',
        'review_date_local':'2026-10-25'})
    print(record['status'])


if __name__=='__main__':main()
