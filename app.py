import streamlit as st
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
load_dotenv()

llm = ChatOpenAI()
# create a heading
st.header("Chat with a pdf")

# Load a pdf file

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
# get the content from the file
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    number_of_pages = len(reader.pages)
    text_content = ""
    for page_ind in range(number_of_pages):
        text_content = text_content + (reader.pages[page_ind].extract_text())

    # Split the document using text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,
                                                   chunk_overlap=200,
                                                   length_function=len,
                                                   separators="\n")
    split_docs = text_splitter.split_text(text_content)
    st.write("Document uploaded and processed!")
    # st.write(split_docs)

    # create embeddings object and then create a vector database
    embeddings = OpenAIEmbeddings()

    # Initialise a vector database and pass the embeddings model and the documents
    if 'db' not in st.session_state or True:
        pinecone_index = "rag-application"
        st.session_state.db = PineconeVectorStore.from_texts(
            texts=split_docs, embedding=embeddings, index_name=pinecone_index)
        # st.session_state.db = Chroma.from_texts(split_docs, embeddings)
        st.write("Knowledgebase created !")

        # for collection in st.session_state.db._client.list_collections():
        #     ids = collection.get()['ids']
        #     print('REMOVE %s document(s) from %s collection' %
        #           (str(len(ids)), collection.name))
        #     if len(ids):
        #         collection.delete(ids)
        #     print("deleted")

    # vector_store = DocArrayInMemorySearch.from_texts(
    #     split_docs,
    #     embedding=embeddings
    # )
    # pinecone_index = "rag-application"
    # docsearch = PineconeVectorStore.from_texts(
    #     texts=split_docs, embedding=embeddings, index_name=pinecone_index)

    # Create a retriever from the datasource
    retriever = st.session_state.db.as_retriever()

    # response = retriever.invoke("What is the need for langsmith?")
    # print("***", response)

    # creating the prompt
    template = '''
    I am going to provide you some context and ask you a question. You are supposed to answer the question based on the context provided. PLease do not make 
    anything up and answer only from the context. If you dont know the answer then just say "I dont know".


    Context = {context}
    Question = {input}

    '''
    prompt = ChatPromptTemplate.from_template(template=template)

    # create a retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    query = st.text_input("Enter your question!")
    st.write(f"query : {query}")
    if query is not None and query != "":
        response = rag_chain.invoke({
            "input": query
        })

        answer = response["answer"]
        sources = response["context"]
        st.write(f"Answer: {answer}")

        st.subheader("Sources::")
        # print(sources)

        for source in sources:
            st.write(source)

        print("*******", response)


print("file uploaded!")
