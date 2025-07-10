import streamlit as st
from conversation import run_agent_conversation, run_chat
from speech_utils import record_audio
from initialize import initialize_documents

st.set_page_config(page_title="AI面談アプリ", layout="centered")
st.title("💼 生成AI営業面談シミュレーター")

# -------------------------------
# ✅ サイドバー（設定）
# -------------------------------
st.sidebar.header("🛠 設定")
mode = st.sidebar.radio("🤖 対話モード", ["Agent（自由会話）", "RAG（資料参照）"])
level = st.sidebar.selectbox("🧠 顧客レベル", ["初級", "中級", "上級"])

# -------------------------------
# メイン画面：テーマ入力
# -------------------------------
theme = st.text_input("📝 面談テーマ（例：融資相談、設備投資など）")

# RAGモードでのみ文書初期化
if mode == "RAG（資料参照）":
    vectorstore = initialize_documents()

# -------------------------------
# 入力タブ：テキスト or 音声
# -------------------------------
tab1, tab2 = st.tabs(["⌨ テキスト入力", "🎤 音声入力"])

with tab1:
    user_input = st.text_input("発言を入力してください", key="text_input")
    if st.button("送信", key="text_button") and user_input:
        if mode == "Agent（自由会話）":
            run_agent_conversation(user_input, level, theme)
        else:
            run_chat(user_input, level, theme, vectorstore)

with tab2:
    if st.button("🎙 話しかける"):
        recognized = record_audio()
        if recognized:
            if mode == "Agent（自由会話）":
                run_agent_conversation(recognized, level, theme)
            else:
                run_chat(recognized, level, theme, vectorstore)

# -------------------------------
# 会話ログ表示
# -------------------------------
with st.expander("📜 会話履歴"):
    if "chat_display_history" in st.session_state:
        for i, (q, a) in enumerate(st.session_state.chat_display_history, 1):
            st.markdown(f"**{i}. あなた:** {q}")
            st.markdown(f"**{i}. 顧客:** {a}")
