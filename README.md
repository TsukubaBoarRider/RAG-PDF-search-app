# 事前準備
RAGに使用するPDFを「doc」フォルダーに入れる。

# 1回目のみ
$ python -m venv .venv && .venv\Scripts\activate
$ pip install -r requirements.txt

# 2回目以降(1回目の時は省略)
$ .venv\Scripts\activate

# API KEYを設定
$ set "OPENAI_API_KEY=【Your API key】"

# 1️⃣ まずベクトル化を実行する（ファイルを入れ替えた場合は再実行）
$ python build_index.py

# 2️⃣ Webアプリを起動する
$ streamlit run app.py
