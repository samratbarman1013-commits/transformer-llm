"""Swap FineWeb-Edu (now login-gated on HF) -> English Wikipedia in the 550M
Colab notebook. Assert-guarded: any mismatch fails loudly."""
import json, urllib.request

URL = 'https://raw.githubusercontent.com/samratbarman1013-commits/transformer-llm/main/notebooks/train_500m_colab.ipynb'
with urllib.request.urlopen(URL) as r:
    nb = json.loads(r.read().decode())

def set_src(cell, text):
    lines = text.split('\n')
    if lines[-1] == '':
        cell['source'] = [l + '\n' for l in lines[:-1]]
    else:
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]

c0 = ''.join(nb['cells'][0]['source'])
c0 = c0.replace(
    "Quality English corpus: **FineWeb-Edu (~5 GB) + Python code (~1.5 GB) + TinyStories (~1 GB)**.",
    "Quality English corpus: **English Wikipedia (~5 GB) + Python code (~1.5 GB) + TinyStories (~1 GB)**.")
assert 'English Wikipedia' in c0, 'markdown swap failed'
set_src(nb['cells'][0], c0)

c3 = ''.join(nb['cells'][3]['source'])
OLD = "        # 1) FineWeb-Edu - high-quality educational web text (~5 GB)\n        from datasets import load_dataset\n        cap, n = 5_000_000_000, 0\n        ds = load_dataset('HuggingFaceTB/fineweb-edu_', split='train', streaming=True)\n        print('downloading fineweb-edu (5 GB)... several minutes - progress every 0.5 GB below')\n        last = 0\n        for ex in ds:\n            t = (ex.get('text') or '').strip()\n            if len(t) < 200:\n                continue\n            out.write(t + NL); n += len(t) + 1\n            if n >= cap:\n                break\n            if n - last >= 500_000_000:\n                last = n; print('  fineweb-edu:', round(n/1e9, 2), 'GB downloaded')\n        print('fineweb-edu:', round(n/1e9, 2), 'GB')\n"
NEW = "        # 1) English Wikipedia (high-quality reference text, ~5 GB)\n        from datasets import load_dataset\n        cap, n = 5_000_000_000, 0\n        ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)\n        print('downloading english wikipedia (5 GB)... several minutes - progress every 0.5 GB below')\n        last = 0\n        for ex in ds:\n            t = (ex.get('text') or '').strip()\n            if len(t) < 200:\n                continue\n            out.write(t + NL); n += len(t) + 1\n            if n >= cap:\n                break\n            if n - last >= 500_000_000:\n                last = n; print('  wikipedia:', round(n/1e9, 2), 'GB downloaded')\n        print('wikipedia:', round(n/1e9, 2), 'GB')\n"
assert OLD in c3, 'fineweb block anchor mismatch'
c3 = c3.replace(OLD, NEW)
assert "load_dataset('wikimedia/wikipedia', '20231101.en'" in c3, 'wikipedia swap failed'2 in c3, 'wikipedia swap failed'
assert 'fineweb' not in c3 and 'GB downloaded' in c3
set_src(nb['cells'][3], c3)

open('notebooks/train_500m_colab.ipynb', 'w').write(json.dumps(nb, indent=1))
print('notebook rebuilt with wikipedia source')
