from pybliometrics.scopus import ScopusSearch

query = 'TITLE-ABS-KEY("high-entropy alloys" AND "hydrogen storage")'
s = ScopusSearch(query, subscriber=True)

print(f"Found {len(s.results)} papers")

for result in s.results[:5]:
    print(result.title)
