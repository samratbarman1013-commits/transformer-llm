"""Swap login-gated FineWeb-Edu -> English Wikipedia in the 550M notebook."""
import json, urllib.request

URL = 'https://raw.githubusercontent.com/samratbarman1013-commits/transformer-llm/main/notebooks/train_500m_colab.ipynb'
with urllib.request.urlopen(URL) as r:
    nb = json.loads(r.read().decode())

def set_src(cell, text):
    lines = text.split('\n')
    if lines[-1]:
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    else:
        cell['source'] = [l + '\n' for l in lines[:-1]]

c0 = ''.join(nb['cells'][0]['source']).replace('FineWeb-Edu', 'English Wikipedia')
assert 'English Wikipedia' in c0, 'markdown swap failed'
set_src(nb['cells'][0], c0)

c3 = ''.join(nb['cells'][3]['source'])
c3 = c3.replace("load_dataset('HuggingFaceTB/fineweb-edu', split='train'",
                "load_dataset('wikimedia/wikipedia', '20231101.en', split='train'")
assert "'20231101.en'" in c3, 'dataset swap failed'
c3 = c3.replace('FineWeb-Edu - high-quality educational web text',
               'English Wikipedia - high-quality reference text')
c3 = c3.replace('fineweb-edu', 'wikipedia')
assert 'fineweb' not in c3, 'fineweb still present'
set_src(nb['cells'][3], c3)

open('notebooks/train_500m_colab.ipynb', 'w').write(json.dumps(nb, indent=1))
print('notebook rebuilt with wikipedia source')
