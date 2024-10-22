import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。
# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

from zhipuai import ZhipuAI
def zhipu_embedding(text: str):

    api_key = os.environ['ZHIPUAI_API_KEY']
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response

text = '要生成 embedding 的输入文本，字符串形式。'
response = zhipu_embedding(text=text)

# print(f'response类型为：{type(response)}')
# print(f'embedding类型为：{response.object}')
# print(f'生成embedding的model为：{response.model}')
# print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
# print(f'embedding（前10）为: {response.data[0].embedding[:10]}')

#文档读取
from langchain.document_loaders.pdf import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("/Users/yangzhentao/Documents/AI/llm-universe/data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()
print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")

pdf_page = pdf_pages[1]
# print(f"每一个元素的类型：{type(pdf_page)}.", 
#     f"该文档的描述性数据：{pdf_page.metadata}", 
#     f"查看该文档的内容:\n{pdf_page.page_content}", 
#     sep="\n------\n")

#数据清洗
import re
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
# print(pdf_page.page_content)
pdf_page.page_content = pdf_page.page_content.replace('•', '')
pdf_page.page_content = pdf_page.page_content.replace(' ', '')
print(pdf_page.page_content)

#文档分割
