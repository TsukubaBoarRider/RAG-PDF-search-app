import pickle, faiss, numpy as np, streamlit as st
from openai import OpenAI

VEC_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"

st.set_page_config(page_title="RAG機能アプリ & まとめ Demo")
st.title("📄 RAG機能アプリ / まとめ")

@st.cache_resource
def load_resources():
    index = faiss.read_index(VEC_PATH)
    metas = pickle.load(open(META_PATH, "rb"))
    client = OpenAI()
    return index, metas, client
index, metas, client = load_resources()

query = st.text_input("質問：", "")
topk = st.slider("最初の数件のドキュメントを表示", 1, 5, 3)
if query:
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding
    D,I = index.search(np.array([q_emb]).astype("float32"), topk)
    for rank, idx in enumerate(I[0], 1):
        fname, chunk = metas[idx]
        with st.expander(f"#{rank}　{fname}"):
            # LLM まとめ
            summ = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"user",
                     "content":f"150字內の日本語でまとめてください：\n\n{chunk}"}
                ]
            ).choices[0].message.content
            st.markdown("**まとめ：** " + summ)
            if st.checkbox("参考文章", key=idx):
                st.code(chunk[:2000])
