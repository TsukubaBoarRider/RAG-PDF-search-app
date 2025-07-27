import os, pickle, tiktoken, time
from pathlib import Path
from openai import OpenAI
import pdfplumber
import faiss, numpy as np

# --------- パラメータ ---------
DOC_DIR   = Path("doc")
VEC_PATH  = Path("data/index.faiss")
META_PATH = Path("data/meta.pkl")
CHUNK_SIZE = 800
BATCH_SIZE = 128 
# ------------------------

client = OpenAI()
enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str):
    toks = enc.encode(text)
    for i in range(0, len(toks), CHUNK_SIZE):
        yield enc.decode(toks[i:i+CHUNK_SIZE])

def read_pdf_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

def embed(texts, batch=BATCH_SIZE):
    vecs = []
    for i in range(0, len(texts), batch):
        part = [t for t in texts[i:i+batch] if t.strip()]
        if not part:
            continue
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=part
        )
        vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.5)
    return vecs

# ------------ ファイルの読み込み、スライス処理------------
vecs, metas = [], []
print(f"📂 スキャン {DOC_DIR.resolve()}")
for pdf in sorted(DOC_DIR.glob("*.pdf*")):
    text = read_pdf_text(pdf)
    if not text.strip():
        print(f"⚠️ {pdf.name} 抽出できる文字がなく、スキップする")
        continue
    for ck in chunk_text(text):
        vecs.append(ck)
        metas.append((pdf.name, ck))
print("Total chunks =", len(vecs))

if not vecs:
    raise RuntimeError("❌ ドキュメントが読み取れない。フォルダーおよびPDFの内容を確認してください")
# ------------ Vector Embedding ------------
print("🚀 Starting Embedding…（ 1~2 分が必要）")
embeds = embed(vecs)
print("Embeds =", len(embeds))

# ------------  Building Vector Store ------------
index = faiss.IndexFlatL2(len(embeds[0]))
index.add(np.asarray(embeds, dtype="float32"))
VEC_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, VEC_PATH.as_posix())
pickle.dump(metas, META_PATH.open("wb"))
print("✅ Vector index completed！")
