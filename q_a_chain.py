import sys
sys.path.append("/Users/yangzhentao/Documents/AI/llm-universe/notebook/C3 搭建知识库") # 将父目录放入系统路径中
#--------------------------------------------------------------------------------------------------------------------------------
# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(find_dotenv())    # read local .env file

zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']

# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 向量数据库持久化路径
persist_directory = '/Users/yangzhentao/Documents/AI/llm-universe/data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

#print(f"向量库中存储的数量：{vectordb._collection.count()}")
#question = "什么是prompt engineering?"
#docs = vectordb.similarity_search(question,k=3)
# print(f"检索到的内容数：{len(docs)}")
# for i, doc in enumerate(docs):
#     print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")



#--------------------------------------------------------------------------
#调用 zhipuapi
from zhipuai_llm import ZhipuAILLM

from dotenv import find_dotenv, load_dotenv
import os

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息
zhipuai_model = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = api_key)  #model="glm-4-0520",
# print(zhipuai_model.invoke("请你自我介绍一下自己！"))

#构建检索问答链
from langchain.prompts import PromptTemplate

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(zhipuai_model,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

# 再创建一个基于模板的检索链：
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(zhipuai_model,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

#检索问答链效果测试
# question_1 = "什么是南瓜书？"
# question_2 = "王阳明是谁？"

# result = qa_chain({"query": question_1})
# print("大模型+知识库后回答 question_1 的结果：")
# print(result["result"])

# result = qa_chain({"query": question_2})
# print("大模型+知识库后回答 question_2 的结果：")
# print(result["result"])

# # 大模型自己回答的效果
# prompt_template = """请回答下列问题:
#                             {}""".format(question_1)

# ### 基于大模型的问答
# print(zhipuai_model.predict(prompt_template))

#——————————————————————————————————————————————————————————————————————————————————————————
#添加历史对话的记忆功能
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)

#对话检索链（ConversationalRetrievalChain）
from langchain.chains import ConversationalRetrievalChain

retriever=vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    zhipuai_model,
    retriever=retriever,
    memory=memory
)
question = "我可以学习到关于提示工程的知识吗？"
result = qa({"question": question})
print(result['answer'])

question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
print(result['answer'])