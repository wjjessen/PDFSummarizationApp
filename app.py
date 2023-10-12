import base64
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from transformers import T5Tokenizer, T5ForConditionalGeneration

# model and tokenizer
# offload_folder = "offload"
checkpoint = "MBZUAI/LaMini-Flan-T5-77M"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, truncation=True, legacy=False, model_max_length=1000
)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    #    device_map="auto",
    torch_dtype=torch.float32,
    #    offload_folder=offload_folder,
)

# notes
# https://huggingface.co/docs/transformers/pad_truncation


# file loader and preprocessor
def file_preprocessing(file, skipfirst):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    print("")
    print("# pages[0] ##########")
    print("")
    print(pages[0])
    print("")
    print("# pages ##########")
    print("")
    print(pages)
    # if skipping the first page, remove pages[0]
    if skipfirst == 1:
        del pages[0]
    else:
        pages = pages
    print("")
    print("# pages after loop ##########")
    print("")
    print(pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        #    print(text)
        final_texts = final_texts + text.page_content
    return final_texts


# llm pipeline
def llm_pipeline(filepath, skipfirst):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=1000,
        min_length=300,
        truncation=True,
    )
    # print("Default number of cpu threads: {}".format(torch.get_num_threads()))
    # torch.set_num_threads(16)
    # print("Modified number of cpu threads: {}".format(torch.get_num_threads()))
    input_text = file_preprocessing(filepath, skipfirst)
    result = pipe_sum(input_text)
    result = result[0]["summary_text"]
    return result


@st.cache_data
# function to display the PDF
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    # embed pdf in html
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    # display file
    st.markdown(pdf_display, unsafe_allow_html=True)


# streamlit code
st.set_page_config(layout="wide")


def main():
    st.title("RASA: Research Article Summarization App")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        skipfirst = st.checkbox("Skip first page")
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded PDF")
                pdf_viewer = displayPDF(filepath)
            with col2:
                st.info("PDF Summary")
                with st.spinner("Please wait..."):
                    summary = llm_pipeline(filepath, skipfirst)
                st.success(summary)


if __name__ == "__main__":
    main()
