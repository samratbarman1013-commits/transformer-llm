"""One-shot builder: rebuild notebooks/train_500m_colab.ipynb and
transformer/tokenizer_bpe.py from PINNED commits, applying the self-heal
patches. Every patch is assert-guarded so a mismatch fails loudly instead
of silently corrupting files."""
import json, urllib.request

REPO = 'samratbarman1013-commits/transformer-llm'

def get(sha, path):
    url = f'https://raw.githubusercontent.com/{REPO}/{sha}/{path}'
    with urllib.request.urlopen(url) as r:
        return r.read().decode()

def set_src(cell, text):
    lines = text.split('\n')
    if lines[-1] == '':
        cell['source'] = [l + '\n' for l in lines[:-1]]
    else:
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]

# ---------- notebook ----------
nb = json.loads(get('6ebb5e0b2e9be5a34a0863df2f72b8068f6f836d', 'notebooks/train_500m_colab.ipynb'))

c3 = ''.join(nb['cells'][3]['source'])
c3 = c3.replace(
    "elif os.path.exists(CORPUS):\n    print('corpus.txt present:', round(os.path.getsize(CORPUS)/1e9, 2), 'GB')\nelse:\n",
    "elif os.path.exists(CORPUS) and os.path.getsize(CORPUS) > 5_000_000_000:\n    print('corpus.txt present:', round(os.path.getsize(CORPUS)/1e9, 2), 'GB')\nelse:\n    if os.path.exists(CORPUS):\n        print('incomplete corpus found (', round(os.path.getsize(CORPUS)/1e9, 2), 'GB) - redownloading')\n")
assert 'getsize(CORPUS) > 5_000_000_000' in c3 and 'redownloading' in c3
c3 = c3.replace(
    "        ds = load_dataset('HuggingFaceTB/fineweb-edu', split='train', streaming=True)\n        for ex in ds:\n",
    "        ds = load_dataset('HuggingFaceTB/fineweb-edu', split='train', streaming=True)\n        print('downloading fineweb-edu (5 GB)... several minutes - progress every 0.5 GB below')\n        last = 0\n        for ex in ds:\n")
assert 'downloading fineweb-edu (5 GB)' in c3
c3 = c3.replace(
    "            if n >= cap:\n                break\n        print('fineweb-edu:', round(n/1e9, 2), 'GB')\n",
    "            if n >= cap:\n                break\n            if n - last >= 500_000_000:\n                last = n; print('  fineweb-edu:', round(n/1e9, 2), 'GB downloaded')\n        print('fineweb-edu:', round(n/1e9, 2), 'GB')\n")
assert 'GB downloaded' in c3
c3 = c3.replace(
    "        for shard in ('file-000000000001.json.gz', 'file-000000000002.json.gz'):\n            path = f'{DATA}/{shard}'\n",
    "        for i, shard in enumerate(('file-000000000001.json.gz', 'file-000000000002.json.gz'), 1):\n            print(f'downloading python code shard {i}/2 (~1 GB)... a few quiet minutes')\n            path = f'{DATA}/{shard}'\n")
assert 'code shard {i}/2' in c3
c3 = c3.replace(
    "        ts = f'{DATA}/ts.txt'\n",
    "        print('downloading TinyStories (1 GB)... a few quiet minutes')\n        ts = f'{DATA}/ts.txt'\n")
assert 'downloading TinyStories' in c3
set_src(nb['cells'][3], c3)
assert "print('corpus ready')" in c3

c4 = ''.join(nb['cells'][4]['source'])
c4 = c4.replace(
    "if not os.path.exists(TOK):\n",
    "if not (os.path.exists(TOK) and os.path.getsize(TOK) > 500_000):\n    if os.path.exists(TOK): print('stub tokenizer found - retraining')\n")
assert 'getsize(TOK) > 500_000' in c4 and "reusing tokenizer from Drive" in c4
set_src(nb['cells'][4], c4)

open('notebooks/train_500m_colab.ipynb', 'w').write(json.dumps(nb, indent=1))

# ---------- tokenizer_bpe.py ----------
src = get('4c62e2e709dae73f33afa97ea3f2eeea9a709f75', 'transformer/tokenizer_bpe.py')
src = src.replace(
    "def encode_file(text_path: str, tokenizer_path: str, out_dir: str,",
    "ENCODE_PROGRESS_BYTES = 200_000_000  # print progress every ~200MB of text\n\ndef encode_file(text_path: str, tokenizer_path: str, out_dir: str,")
old_ef = """    t = BpeTokenizer(tokenizer_path)
    parts = []
    with open(text_path, encoding="utf-8") as f:
        buf = []
        size = 0
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= 2 << 20:  # ~2MB chunks (small for low-RAM machines)
                parts.append(np.array(t.encode("".join(buf)), dtype=np.uint32))
                buf, size = [], 0
        if buf:
            parts.append(np.array(t.encode("".join(buf)), dtype=np.uint32))
    ids = np.concatenate(parts)
"""
new_ef = """    t = BpeTokenizer(tokenizer_path)
    parts = []
    total = last_p = 0
    with open(text_path, encoding="utf-8") as f:
        buf = []
        size = 0
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= 2 << 20:  # ~2MB chunks (small for low-RAM machines)
                parts.append(np.array(t.encode("".join(buf)), dtype=np.uint32))
                total += size
                buf, size = [], 0
                if total - last_p >= ENCODE_PROGRESS_BYTES:
                    last_p = total
                    print(f"  tokenized {total/1e9:.1f} GB of text so far...")
        if buf:
            parts.append(np.array(t.encode("".join(buf)), dtype=np.uint32))
    if not parts:
        raise RuntimeError(
            f"corpus file is empty or unreadable: {text_path} "
            "(an earlier download was interrupted - delete the file and re-run the corpus cell)")
    ids = np.concatenate(parts)
"""
assert old_ef in src, 'encode_file anchor mismatch'
src = src.replace(old_ef, new_ef)
assert 'ENCODE_PROGRESS_BYTES' in src and 'corpus file is empty or unreadable' in src
assert 'SPECIALS = ["' in src  # specials untouched
open('transformer/tokenizer_bpe.py', 'w').write(src)
print('builder: wrote notebooks/train_500m_colab.ipynb + transformer/tokenizer_bpe.py')
