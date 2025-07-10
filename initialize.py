# initialize.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

# 環境変数読み込み
load_dotenv()

# ✅ APIキー確認
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OpenAI APIキーが設定されていません。`.env` ファイルまたは環境変数をご確認ください。")

# 対応ファイルの設定（必要に応じて拡張可能）
SUPPORTED_FILES = [
    "data/法人顧客との会話スクリプト.txt",
    "data/法人顧客との会話スクリプト.pdf",
    "data/法人顧客との会話スクリプト.csv"
]

# 文書読み込み関数
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".txt":
            return TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".pdf":
            return PyPDFLoader(file_path).load()
        elif ext == ".csv":
            return CSVLoader(file_path).load()
        else:
            raise ValueError(f"❌ 未対応のファイル形式です: {ext}")
    except Exception as e:
        st.error(f"📂 ファイル読み込みエラー: {e}")
        raise RuntimeError(e)

# メイン関数：ベクトルストア生成
def initialize_documents():
    for file_path in SUPPORTED_FILES:
        if os.path.exists(file_path):
            docs = load_document(file_path)
            if not docs:
                raise ValueError("📂 読み込んだ文書が空です。")
            break
    else:
        st.error("❌ 対応する文書ファイルが見つかりません。")
        raise FileNotFoundError("文書ファイルが存在しません。")

    # 分割してベクトル化
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        splitted_docs,
        embedding=embeddings,
        persist_directory="./.db"
    )
    return vectorstore
