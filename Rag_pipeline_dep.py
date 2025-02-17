import os
import boto3
from dotenv import load_dotenv
import pickle
from io import BytesIO
import streamlit as st
import base64
from base64 import b64decode  # Needed for parsing base64 strings
import uuid

# LangChain and related imports
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import chromadb  # Importing chromadb to access its functionalities

# Load environment variables
load_dotenv()

# Set environment variables (S3 and API keys)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
if S3_BUCKET_NAME is None:
    raise ValueError("S3_BUCKET_NAME is not defined in the environment variables.")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

def download_pickle_from_s3(file_name, local_path):
    """Downloads a pickle file from S3 to a local directory."""
    try:
        s3_client.download_file(S3_BUCKET_NAME, file_name, local_path)
        with open(local_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        return None

def load_cached_data():
    """Loads all pickle files from S3."""
    files = [
        "tables.pkl", "texts.pkl", "images.pkl",
        "text_summaries.pkl", "table_summaries.pkl", "image_summaries.pkl"
    ]
    local_cache_dir = "./s3_cache"
    os.makedirs(local_cache_dir, exist_ok=True)

    data = []
    for file in files:
        local_path = os.path.join(local_cache_dir, file)
        data.append(download_pickle_from_s3(file, local_path))
    
    if None in data:
        print("One or more files failed to load.")
        return None
    else:
        print("Loaded cached data from S3.")
        return tuple(data)

# Attempt to load the cached data
cached_data = load_cached_data()

# Check cached data and initialize variables
tables, texts, images, text_summaries, table_summaries, image_summaries = (
    ([], [], [], [], [], []) if cached_data is None else cached_data
)

# Initialize the vectorstore and retriever
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=OpenAIEmbeddings()
)
store = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

# Add texts to the retriever
if texts:
    try:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))
    except Exception as e:
        print(f"Error adding documents to retriever: {e}")

# Add image summaries to the retriever
if images:
    try:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]})
            for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images)))
    except Exception as e:
        print(f"Error adding image documents: {e}")

def parse_docs(docs):
    """Split base64-encoded images and texts."""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = ""

    if docs_by_type.get("texts"):
        context_text = "".join([text_element.text for text_element in docs_by_type["texts"]])

    # Construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """
    
    prompt_content = [{"type": "text", "text": prompt_template}]
    
    if docs_by_type.get("images"):
        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })

    return ChatPromptTemplate.from_messages(
        [HumanMessage(content=prompt_content)]
    )

chain_with_sources = {
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
)

# Initialize Streamlit app
st.set_page_config(
    page_title="RAG Pipeline for Research Related to Renal Diseases",
    page_icon="🧬"
)
st.title("RAG Pipeline for Research Related to Renal Diseases")

# Welcome prompt
st.markdown("### Welcome!")
st.write(
    "This RAG pipeline responds to queries related to research on renal diseases. "
    "For complete context, scroll down. Click on the submit button twice to see full context if not available."
)

# Input box for user query
user_input = st.text_input("Type your query here:")

# Function to display base64 images in Streamlit
def display_base64_image(base64_code):
    if base64_code:
        formatted_image = f"data:image/jpeg;base64,{base64_code}"  # Ensure the image format is correct
        st.image(formatted_image, caption="Context Image", use_column_width=True)

if st.button("Submit"):
    if user_input:
        with st.spinner("Processing your query... Please wait..."):
            # Invoke the RAG chain
            response = chain_with_sources.invoke(user_input)

        if response is not None:
            # Clear system cache for chromadb after each invocation
            chromadb.PersistentClient().reset()


            # Display the response
            st.write("### Response:")
            st.markdown(response['response'], unsafe_allow_html=True)

            # Display context images if they exist
            st.write("### Context Images:")
            if response['context']['images']:
                for image in response['context']['images']:
                    display_base64_image(image)
            else:
                st.write("No context images available.")

            # Display the context text and page numbers
            st.write("### Context:")
            for text in response['context']['texts']:
                st.write(text.text)
                st.write("Page number:", text.metadata.page_number)
        else:
            st.warning("No valid response returned from RAG chain.")
    else:
        st.warning("Please enter a query.")
