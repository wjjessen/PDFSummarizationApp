import base64
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.docstore.document import Document

# model and tokenizer
# offload_folder = "offload"

from transformers import T5Tokenizer, T5ForConditionalGeneration

# text2textgen 990mb pytorch_model
# fine-tuned version of google/flan-t5-base on LaMini-instruction dataset
# that contains 2.58M samples for instruction fine-tuning.
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(
    checkpoint, truncation=True, legacy=False, model_max_length=1000
)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    #    device_map="auto",
    torch_dtype=torch.float32,
    #    offload_folder=offload_folder,
)

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# summarization 1.63gb pytorch_model
# BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder
# and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an
# arbitrary noising function, and (2) learning a model to reconstruct the original text.
# BART is particularly effective when fine-tuned for text generation (e.g. summarization,
# translation) but also works well for comprehension tasks (e.g. text classification,
# question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail,
# a large collection of text-summary pairs.
# Evaluation results
# ROUGE-1 on cnn_dailymail self-reported 42.949
# ROUGE-2 on cnn_dailymail self-reported 20.815
# ROUGE-L on cnn_dailymail self-reported 30.619
# ROUGE-LSUM on cnn_dailymail self-reported 40.038
# loss on cnn_dailymail self-reported 2.529
# gen_len on cnn_dailymail self-reported 78.587
# https://paperswithcode.com/sota/summarization-on-cnn-dailymail
# checkpoint = "bart-large-cnn"
# tokenizer = AutoTokenizer.from_pretrained(
#    checkpoint, model_max_length=512, truncation=True, legacy=False
# )
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#    checkpoint,
#    #    device_map="auto",
#    torch_dtype=torch.float32,
#    offload_folder=offload_folder,
# )

# notes
# https://huggingface.co/docs/transformers/pad_truncation


# file loader and preprocessor
def file_preprocessing(file, skipfirst):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    print("")
    print("# pages[0] ########################################################")
    print("")
    print(pages[0])
    print("")
    print("# pages ########################################################")
    print("")
    print(pages)
    # if skipping the first page, then remove pages[0]
    if skipfirst == 1:
        del pages[0]
    else:
        pages = pages
    print("")
    print("# pages after loop ########################################################")
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
    st.title("PDF Summarization")
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
