import os
import boto3
from dotenv import load_dotenv
import pickle
import streamlit as st
import base64
import uuid
from io import BytesIO

from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Configure environment
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REQUIRED_FILES = [
    "tables.pkl", "texts.pkl", "images.pkl",
    "text_summaries.pkl", "table_summaries.pkl", "image_summaries.pkl"
]

# Validate environment variables
if not S3_BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME is not defined in environment variables.")

# Initialize S3 client with error handling
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
except Exception as e:
    st.error(f"Failed to initialize S3 client: {e}")
    st.stop()

def load_from_s3(file_name):
    """Load pickle file directly from S3 without local storage"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_name)
        return pickle.loads(response['Body'].read())
    except Exception as e:
        st.error(f"Error loading {file_name} from S3: {e}")
        return None

def initialize_retriever():
    """Initialize and populate the retriever with error handling"""
    vectorstore = Chroma(collection_name="multi_modal_rag", 
                        embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id"
    )

    # Load data with progress
    data = {}
    progress_bar = st.progress(0)
    for i, file in enumerate(REQUIRED_FILES):
        data[file.split('.')[0]] = load_from_s3(file)
        progress_bar.progress((i + 1) / len(REQUIRED_FILES))
    
    # Add documents to retriever
    for data_type in ["texts", "tables", "images"]:
        items = data.get(f"{data_type}", [])
        summaries = data.get(f"{data_type}_summaries", [])
        
        if items and summaries:
            try:
                doc_ids = [str(uuid.uuid4()) for _ in items]
                summary_docs = [
                    Document(page_content=s, metadata={"doc_id": doc_ids[i]})
                    for i, s in enumerate(summaries)
                ]
                retriever.vectorstore.add_documents(summary_docs)
                retriever.docstore.mset(list(zip(doc_ids, items)))
            except Exception as e:
                st.error(f"Error adding {data_type} to retriever: {e}")
    
    return retriever

def parse_docs(docs):
    """Organize documents by type with proper base64 handling"""
    organized = {"images": [], "texts": []}
    for doc in docs:
        if isinstance(doc, Document):
            content = doc.page_content
            # Check for base64 pattern
            if len(content) > 100 and 'image' in doc.metadata.get('type', ''):
                organized["images"].append(content)
            else:
                organized["texts"].append(doc)
    return organized

def build_prompt(context, question):
    """Construct multimodal prompt with proper formatting"""
    prompt_content = [{"type": "text", "text": f"Answer this question: {question}"}]
    
    # Add text context
    if context["texts"]:
        text_context = "\n".join([d.page_content for d in context["texts"]])
        prompt_content[0]["text"] += f"\nContext:\n{text_context}"

    # Add images
    for img in context["images"][:3]:  # Limit to 3 images
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

# Streamlit UI Configuration
st.set_page_config(
    page_title="Renal Disease Research Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .response-box {
        padding: 20px;
        background: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    .context-section {
        margin: 15px 0;
        padding: 15px;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("Renal Disease Research Assistant")
st.markdown("""
    Welcome to the Renal Disease Research Assistant. 
    Ask questions about recent research in nephrology and related fields.
""")

# Initialize retriever
retriever = initialize_retriever()

# Build processing chain
processing_chain = (
    {"context": retriever | RunnableLambda(parse_docs), "question": RunnablePassthrough()}
    | RunnablePassthrough().assign(
        response=(
            RunnableLambda(lambda x: build_prompt(x["context"], x["question"]))
            | ChatOpenAI(model="gpt-4-turbo")
            | StrOutputParser()
        )
    )
)

# Query Interface
query = st.text_input("Enter your research question:", key="query_input")
if st.button("Analyze", type="primary"):
    if query:
        with st.spinner("Analyzing research documents..."):
            try:
                response = processing_chain.invoke(query)
                
                # Display main response
                with st.container():
                    st.markdown("### Research Insights")
                    st.markdown(f'<div class="response-box">{response["response"]}</div>', 
                               unsafe_allow_html=True)

                # Display context
                with st.expander("View Supporting Context"):
                    # Text context
                    if response["context"]["texts"]:
                        st.markdown("#### Text References")
                        for doc in response["context"]["texts"]:
                            st.markdown(f"""
                                <div class="context-section">
                                    <p>{doc.page_content}</p>
                                    <small>Source: Page {doc.metadata.get('page_number', 'N/A')}</small>
                                </div>
                            """, unsafe_allow_html=True)

                    # Image context
                    if response["context"]["images"]:
                        st.markdown("#### Visual References")
                        cols = st.columns(3)
                        for i, img in enumerate(response["context"]["images"][:3]):
                            with cols[i % 3]:
                                st.image(
                                    f"data:image/jpeg;base64,{img}",
                                    caption=f"Relevant diagram {i+1}",
                                    use_column_width=True
                                )
                                
            except Exception as e:
                st.error(f"Error processing request: {e}")
    else:
        st.warning("Please enter a research question.")
