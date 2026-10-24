import sys, re, urllib.parse, urllib.request, time

def search(query, max_results=40, sort="relevance"):
    base = "http://export.arxiv.org/api/query?"
    params = urllib.parse.urlencode({
        "search_query": query, "start": 0,
        "max_results": max_results, "sortBy": sort, "sortOrder": "descending"})
    with urllib.request.urlopen(base + params, timeout=60) as f:
        s = f.read().decode()
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", s, re.S):
        g = lambda p: (re.search(p, e, re.S).group(1).strip() if re.search(p, e, re.S) else "")
        t = re.sub(r"\s+", " ", g(r"<title>(.*?)</title>"))
        idu = g(r"<id>(.*?)</id>").split("/abs/")[-1]
        d = g(r"<published>(.*?)</published>")[:7]
        u = g(r"<updated>(.*?)</updated>")[:7]
        cm = re.sub(r"\s+", " ", g(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>"))[:90]
        out.append((d, u, t, idu, cm))
    return out

if __name__ == "__main__":
    for q in sys.argv[1:]:
        print("=" * 30, q)
        try:
            for d, u, t, i, c in search(q):
                print(f"{d} | {t[:105]} | {i} | {c}")
        except Exception as ex:
            print("ERR", ex)
        time.sleep(3)
