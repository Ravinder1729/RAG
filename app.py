from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from IPython.display import Markdown as md
from langchain_community.document_loaders import PyPDFLoader
import nltk
import re
from markupsafe import Markup 
nltk.download('punkt')
from flask import Flask,render_template,request

app = Flask(__name__)

# Load and process documents
loader = PyPDFLoader(r"C:\Users\ravin\Downloads\LORA.pdf")
data = loader.load()

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyBVkdA9-VOamlERftuPcUxhLxfzLiiRFRk", model="models/embedding-001")
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_model = ChatGoogleGenerativeAI(google_api_key='AIzaSyDlU5sqyfQ-DOPWJ2jI6_ouKdsDDRjSbEg', model="gemini-1.5-pro-latest")

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

output_parser = StrOutputParser()

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""  # Initialize user_input with an empty string or a default value
    
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            response = rag_chain.invoke(user_input)
            return render_template("index.html", response=response, user_input=user_input)
    
    return render_template("index.html", user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)