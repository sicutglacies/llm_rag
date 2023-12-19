from app.rag.rag_engine import rag_chain_with_source


import streamlit as st


st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation")
st.header("Помощник по базе знаний о дебетовой карте Тинькофф")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Задайте Ваш вопрос"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = None
        for response in rag_chain_with_source.stream(st.session_state.messages[-1]['content']):
            if sources is None:
                sources = response.get('documents')
            else:
                full_response += (response.get('answer') or "")
            message_placeholder.markdown(full_response + "▌")

        full_response += "\n"
        full_response += "#### Использованные источники\n"
        for n, source in enumerate(sources):
            full_response += f"{n+1}. {'/'.join(list(source.values()))}\n"
        
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
