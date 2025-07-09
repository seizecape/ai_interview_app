# conversation.py
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import streamlit as st

def run_chat(user_input, level, theme, vectorstore):
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    # 会話履歴
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # レベル別プロンプト調整（カスタム応答に使用可能）
    prompt_prefix = {
        "初級": "あなたは銀行の取引先である法人の社長です。新入社員向けに丁寧な言葉で、やさしく答えてください。",
        "中級": "あなたは銀行の取引先である法人の社長です。法人担当経験者相手への回答となるため、必要な時は専門用語も使ってください。",
        "上級": "あなたは銀行の取引先である法人の社長です。長年法人担当をしていたベテラン行員への回答となるため、要点を端的にプロフェッショナルに回答してください。"
    }

    final_input = f"{prompt_prefix[level]} テーマ：{theme} 入力：{user_input}"

    result = qa_chain.run({"question": final_input, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((user_input, result))

    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("ai"):
        st.write(result)
