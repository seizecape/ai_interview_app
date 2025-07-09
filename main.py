# main.py
import streamlit as st
from initialize import initialize_documents
from conversation import run_chat

st.set_page_config(page_title="AI面談シミュレーション", layout="wide")

# ユーザー入力（レベル・テーマ）
st.sidebar.title("設定")
level = st.sidebar.selectbox("レベルを選択", ["初級", "中級", "上級"])
theme = st.sidebar.selectbox("面談テーマを選択", ["新規提案", "金利交渉", "与信審査"])
input_mode = st.sidebar.radio("入力方式", ["テキスト", "音声ファイルアップロード"])

# ドキュメント初期化（RAG用）
vectorstore = initialize_documents()

# ユーザー入力
st.title("🧑‍💼 AI面談シミュレーション")
if input_mode == "テキスト":
    user_input = st.text_input("あなたの質問を入力してください", "")
elif input_mode == "音声ファイルアップロード":
    audio_file = st.file_uploader("音声ファイルをアップロード", type=["mp3", "wav"])
    user_input = ""  # 音声認識で変換（後述）

# 実行ボタン
if st.button("送信") and (user_input or audio_file):
    run_chat(user_input, level, theme, vectorstore)
