from transformers import pipeline
import base64
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# notes
# https://huggingface.co/docs/transformers/pad_truncation


# file loader and preprocessor
def file_preprocessing(file, skipfirst, skiplast):
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
    # skip page(s)
    if (skipfirst == 1) & (skiplast == 0):
        del pages[0]
    elif (skipfirst == 0) & (skiplast == 1):
        del pages[-1]
    elif (skipfirst == 1) & (skiplast == 1):
        del pages[0]
        del pages[-1]
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
def llm_pipeline(tokenizer, base_model, filepath, skipfirst, skiplast):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=300,
        truncation=True,
    )
    input_text = file_preprocessing(filepath, skipfirst, skiplast)
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
        st.subheader("Options")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model_names = [
                "Flan T5 small",
                "LaMini GPT-2 124M",
                "Fine-Tuned T5 Small for Text Summarization",
            ]
            selected_model = st.radio("Select a model to use", model_names)
            if selected_model == "LaMini GPT-2 124M":
                checkpoint = "MBZUAI/LaMini-GPT-124M"
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint,
                    truncation=True,
                    legacy=False,
                    model_max_length=1000,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    checkpoint, torch_dtype=torch.float32
                )
            elif selected_model == "Fine-Tuned T5 Small for Text Summarization":
                checkpoint = "Falconsai/text_summarization"
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint,
                    truncation=True,
                    legacy=False,
                    model_max_length=1000,
                )
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    checkpoint, torch_dtype=torch.float32
                )
            else:  # default Flan T5 small
                checkpoint = "MBZUAI/LaMini-Flan-T5-77M"
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint,
                    truncation=True,
                    legacy=False,
                    model_max_length=1000,
                )
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    checkpoint, torch_dtype=torch.float32
                )
        with col2:
            st.write("Skip any pages?")
            skipfirst = st.checkbox("Skip first page")
            skiplast = st.checkbox("Skip last page")
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
                    summary = llm_pipeline(
                        tokenizer, base_model, filepath, skipfirst, skiplast
                    )
                st.success(summary)


st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 1rem;
    font-weight: 400;
}
    </style>
    """,
    unsafe_allow_html=True,
)


if __name__ == "__main__":
    main()
