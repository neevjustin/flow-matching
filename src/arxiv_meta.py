import sys, re, urllib.request, html

IDS = sys.argv[1:]
url = "http://export.arxiv.org/api/query?id_list=" + ",".join(IDS) + "&max_results=60"
s = urllib.request.urlopen(url, timeout=60).read().decode()
for e in re.findall(r"<entry>(.*?)</entry>", s, re.S):
    g = lambda p, d="": (re.search(p, e, re.S).group(1).strip() if re.search(p, e, re.S) else d)
    t = html.unescape(re.sub(r"\s+", " ", g(r"<title>(.*?)</title>")))
    au = [html.unescape(re.sub(r"\s+", " ", a)) for a in re.findall(r"<name>(.*?)</name>", e, re.S)]
    aid = g(r"<id>(.*?)</id>").split("/abs/")[-1]
    yr = g(r"<published>(.*?)</published>")[:4]
    cm = html.unescape(re.sub(r"\s+", " ", g(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>")))
    jr = html.unescape(re.sub(r"\s+", " ", g(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>")))
    print(f"ID={aid}\nYEAR={yr}\nTITLE={t}\nAUTHORS={' and '.join(au)}\nCOMMENT={cm}\nJOURNAL={jr}\n---")
