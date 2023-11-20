from transformers import pipeline
import base64
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# notes
# https://huggingface.co/docs/transformers/pad_truncation


# file loader and preprocessor
def file_preprocessing(file, skipfirst, skiplast):
    loader = PyMuPDFLoader(file)
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # number of characters
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # default list
    )
    # https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts


def preproc_count(filepath, skipfirst, skiplast):
    input_text = file_preprocessing(filepath, skipfirst, skiplast)
    text_length = len(input_text)
    return input_text, text_length


def postproc_count(summary):
    text_length = len(summary)
    return text_length


# llm pipeline
def llm_pipeline(tokenizer, base_model, input_text):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=600,
        min_length=300,
        truncation=True,
    )
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
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            model_names = [
                "T5-Small",
                "BART",
            ]
            selected_model = st.radio("Select a model to use:", model_names)
            if selected_model == "BART":
                checkpoint = "ccdv/lsg-bart-base-16384-pubmed"
                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint,
                    truncation=True,
                    legacy=False,
                    model_max_length=1000,
                    trust_remote_code=True,
                )
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    checkpoint, torch_dtype=torch.float32, trust_remote_code=True
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
        with col3:
            st.write("Background information (links open in a new window)")
            st.write(
                "Model class: [BART](https://huggingface.co/docs/transformers/main/en/model_doc/bart)"
                "&nbsp;&nbsp;|&nbsp;&nbsp;Specific model: [MBZUAI/LaMini-Flan-T5-77M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-77M)"
            )
            st.write(
                "Model class: [T5-Small](https://huggingface.co/docs/transformers/main/en/model_doc/t5)"
                "&nbsp;&nbsp;|&nbsp;&nbsp;Specific model: [ccdv/lsg-bart-base-16384-pubmed](https://huggingface.co/ccdv/lsg-bart-base-16384-pubmed)"
            )
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                input_text, text_length = preproc_count(filepath, skipfirst, skiplast)
                st.info(
                    "Uploaded PDF&nbsp;&nbsp;|&nbsp;&nbsp;Number of words: "
                    f"{text_length:,}"
                )
                pdf_viewer = displayPDF(filepath)
            with col2:
                with st.spinner("Please wait..."):
                    summary = llm_pipeline(tokenizer, base_model, input_text)
                    text_length = postproc_count(summary)
                st.info(
                    "PDF Summary&nbsp;&nbsp;|&nbsp;&nbsp;Number of words: "
                    f"{text_length:,}"
                )
                st.success(summary)


st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 1rem;
    font-weight: 400;
}
div[class*="stMarkdown"] > div[data-testid="stMarkdownContainer"] > p {
    margin-bottom: -15px;
}
div[class*="stCheckbox"] > label {
    margin-bottom: -15px;
}
body > a {
    text-decoration: underline;
}
    </style>
    """,
    unsafe_allow_html=True,
)


if __name__ == "__main__":
    main()
