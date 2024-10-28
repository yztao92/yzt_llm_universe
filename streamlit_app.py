import streamlit as st

import sys
sys.path.append("/Users/yangzhentao/Documents/AI/llm-universe/notebook/C3 搭建知识库") # 将父目录放入系统路径中
#--------------------------------------------------------------------------------------------------------------------------------
# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from zhipuai_llm import ZhipuAILLM



#构建向量库
def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = '/Users/yangzhentao/Documents/AI/llm-universe/data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

#定义返回的值
def generate_response(input_text, api_key):
    zhipuai_model = ZhipuAILLM(model="glm-4", temperature=0.5, api_key=api_key)
    response = zhipuai_model(input_text)
    return response

#带有历史记录的问答链
def get_chat_qa_chain(question:str,api_key:str):
    vectordb = get_vectordb()
    zhipuai_model = ZhipuAILLM(model="glm-4", temperature=0.5, api_key=api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        zhipuai_model,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,api_key:str):
    vectordb = get_vectordb()
    zhipuai_model = ZhipuAILLM(model="glm-4", temperature=0.5, api_key=api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(zhipuai_model,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

# Streamlit 应用程序界面
def main():
    st.title('YZT_AI')
    zhipuai_api_key = st.sidebar.text_input('Zhipuai API Key', type='password')

    #添加一个单选按钮部件st.radio，选择进行问答的模式：
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])
    

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=860)
    if prompt := st.chat_input("你好,我有什么可以帮你的"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        # 调用 respond 函数获取回答
        answer = generate_response(prompt, zhipuai_api_key)
        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

main()