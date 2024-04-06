from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from InstructorEmbedding import INSTRUCTOR
import os
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    # loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    # data = loader.load()
    # ###

    # urls=["https://www.wrike.com/blog/ultimate-guide-team-building-activities"]
    # st.sidebar.title("News Article URLs")

    urls = []
    # for i in range(3):
    #     url = st.sidebar.text_input(f"URL {i+1}")
    #     urls.append(url)
    # process_url_clicked = st.sidebar.button("Process URLs")
    urls=["https://blog.hubspot.com/marketing/creative-team-outing-ideas"]
    if True:
        # load data
        loader = UnstructuredURLLoader(urls=urls)
        # main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n'],
            chunk_size=1000
        )
    # process_url_clicked = st.sidebar.button("Process URLs")
    # loader = UnstructuredURLLoader(urls=urls)
    # main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    # main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print(data)


    ####

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=docs,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    # If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    # QUESTION: {question}
    prompt_template = """Given the following team_size, generate an answer based on this team_size only.
    
    TEAM_SIZE: {team_size}

    """

    PROMPT = PromptTemplate(
        # template=prompt_template, input_variables=["context", "question"]
        template=prompt_template, input_variables=["team_size"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("write a fun task with size of team 5?"))