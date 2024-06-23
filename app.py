# load all the dependencies
import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit page configuration
st.set_page_config(
    page_title='PDF Summarizer',
    page_icon=':pencil2:',
    layout='wide',
    initial_sidebar_state='expanded'
)

# function to load the lottie file
def load_lottiefile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

st.title('PDF Summarizer')
description = '''
Text Summarization is a natural language processing (NLP) task that creates a concise and informative summary of a longer text, 
which can be challenging and time-consuming. Large language models (LLMs) can generate summaries of news articles, literature, technical documents, and other types of text. 
This lightweight web application is designed to quickly summarize any document in PDF format using an open-source LLM, 
making the process more efficient and less tedious.  
'''
st.write(description)

st.sidebar.title('File Uploader')
uploaded_file = st.sidebar.file_uploader('Upload a PDF File', type='pdf', accept_multiple_files=False)

cont = st.container(border=True)
col1, col2 = cont.columns(2, gap='small')
# load the lottie animation    
lottie_cover = load_lottiefile('img/bot.json')
with col1:
    #st.write(description)
    st.lottie(lottie_cover, speed=1, reverse=False, loop=True, quality='low', height=800, key='first_animate')

# load the environment variables
load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# load the llm model, in this case, we use llama3 model
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

chunks_prompt = '''
Please summarize the following document:
Document: '{text}'
Summary:
'''
map_prompt_template = PromptTemplate(input_variables=['text'],
                                     template=chunks_prompt)

final_combine_prompt = '''
Provide a final summary of the entire document with these important points.
Add a document title,
Provide the summary in bulletin points for the document.
Document: '{text}'
'''
final_combine_prompt_template = PromptTemplate(input_variables=['text'],
                                               template=final_combine_prompt)

summary_chain = load_summarize_chain(
    llm = llm,
    chain_type = 'map_reduce',
    map_prompt = map_prompt_template,
    combine_prompt = final_combine_prompt_template,
    verbose=False
)

with col2:
    container = st.container(border=True)
    container.subheader('Summary of the Uploaded PDF File', divider='blue')

    if uploaded_file:
        pdfReader = PdfReader(uploaded_file)
        text = ''
        for i, page in enumerate(pdfReader.pages):
            content = page.extract_text()
            if content:
                text += content

        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        chunks = text_splitter.create_documents([text])

        output = summary_chain.run(chunks)

        container.write(output)


