import streamlit as st
import os

from langchain.memory import ConversationBufferMemory
from utils import qa_agent
from dotenv import load_dotenv

load_dotenv()

index_name = os.environ.get("INDEX_NAME")
if not index_name:
    raise ValueError("環境變數 'INDEX_NAME' 未設定")

print(index_name)

st.title("📑 智慧PDF")

with st.sidebar:
    openai_api_key = st.text_input("請輸入OpenAI API金鑰：", type="password")
    st.markdown("[OpenAI API key 說明](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

uploaded_file = st.file_uploader("上傳你的PDF文件：", type="pdf")
question = st.text_input("對PDF內容進行提問", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("輸入API key 金鑰")

if uploaded_file and question and openai_api_key:
    with st.spinner("AI思考中，請稍候..."):
        response = qa_agent(openai_api_key, st.session_state["memory"],
                            index_name,
                            uploaded_file, question)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("歷史訊息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
