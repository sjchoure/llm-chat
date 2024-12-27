import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub

# Helper function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Avoid adding None or empty text
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
    return text

# Helper function to split text into smaller chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Helper function to create a vectorstore using embeddings
def get_vectorstore(text_chunk):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(text_chunk, embedding=embeddings)
        return vectorstore
    except ImportError:
        st.error("Required dependencies for HuggingFaceInstructEmbeddings are not installed.")
        st.stop()
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        st.stop()

def get_conversation_chain(vectorstore):        
    llm = HuggingFaceHub(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_p": 0.95,
            "repetition_penalty": 1.15
        }
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False,
        verbose=True
    )
    return conversation_chain

def handle_userinput(user_question):
    try:
        # Get response from conversation chain
        response = st.session_state.conversation({"question": user_question})
        
        # Extract and display the bot's message
        bot_message = response.get('answer', 'I am not sure about that.')
        
        # Extract only the helpful answer after "Helpful Answer:" if present
        if "Helpful Answer:" in bot_message:
            bot_message = bot_message.split("Helpful Answer:")[-1].strip()
        
        # Display the cleaned message
        st.write(bot_template.replace("{{MSG}}", bot_message), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        if "timeout" in str(e).lower():
            st.info("The model is taking too long to respond. Please try again in a moment.")

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Pookie Chat :books:")
    
    user_query = st.text_input("Ask a question about your documents:")
    if user_query:
        handle_userinput(user_query)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return

            with st.spinner("Processing..."):
                # Get text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vectorstore
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("PDFs processed successfully!")


if __name__ == '__main__':
    main()
