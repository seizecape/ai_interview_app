from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# -------------------------
# Agentベースの自由対話
# -------------------------
def run_agent_conversation(user_input, level, theme):
    # ✅ 表示用履歴（テキストのタプル）と内部用メモリ（LangChain用）を分離
    if "chat_display_history" not in st.session_state:
        st.session_state.chat_display_history = []

    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True  # <- BaseMessage形式を内部的に返すための指定
        )

    prompt_map = {
        "初級": "あなたは地方の中小企業の社長です。未熟な銀行の営業担当に対して、優しく丁寧に答えてください。",
        "中級": "あなたは地方の製造業の会社社長です。銀行の営業担当に対して、現実的かつ正確に答えます。",
        "上級": "あなたは地方でも有名な売上高100億円のオーナー社長です。経験豊富な銀行の営業担当に対して、論理的かつ厳しく質問に答えます。"
    }

    system_prompt = f"{prompt_map[level]} テーマ: {theme}"
    full_input = f"{system_prompt} 入力: {user_input}"

    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=st.session_state.chat_memory,
        verbose=False
    )

    response = agent.run(full_input)

    # 表示用履歴に保存（文字列ベース）
    st.session_state.chat_display_history.append((user_input, response))

    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("ai"):
        st.write(response)

# -------------------------
# RAGベースの検索付き対話
# -------------------------
def run_chat(user_input, level, theme, vectorstore):
    if "chat_display_history" not in st.session_state:
        st.session_state.chat_display_history = []

    if "chat_history_rag" not in st.session_state:
        st.session_state.chat_history_rag = []

    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    prompt_prefix = {
        "初級": "あなたはある銀行の取引先である法人の社長です。銀行の２年目営業担当に対してやさしく丁寧な言葉で答えてください。",
        "中級": "あなたはある銀行と取引のある法人の社長です。銀行の営業担当に対して現実的かつ誠実に答えてください。",
        "上級": "あなたはある銀行と取引のある老舗企業の社長です。経験豊富な銀行の営業担当に対して専門性を持って要点だけを伝えます。"
    }

    final_input = f"{prompt_prefix[level]} テーマ: {theme} 入力: {user_input}"

    result = qa_chain.run({
        "question": final_input,
        "chat_history": st.session_state.chat_history_rag
    })

    # 履歴保存
    st.session_state.chat_history_rag.append((user_input, result))
    st.session_state.chat_display_history.append((user_input, result))

    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("ai"):
        st.write(result)
