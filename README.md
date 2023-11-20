# RASA: Research Article Summarization App

## Description

This application summarizes an uploaded research article PDF using the large language models "LaMini-Flan-T5-77M" or "lsg-bart-base-16384-pubmed". LaMini-Flan-T5-77M is a fine-tuned version of google/flan-t5-small on LaMini-instruction dataset that contains 2.58M samples for instruction fine-tuning. lsg-bart-base-16384-pubmed is a fine-tuned version of ccdv/lsg-bart-base-4096-pubmed on the scientific_papers pubmed dataset. 

https://huggingface.co/MBZUAI/LaMini-Flan-T5-77M

https://huggingface.co/ccdv/lsg-bart-base-16384-pubmed

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation

Create a virtual python environment. To install the required python application packages, type "pip install -r requirements.txt" in a terminal window within the virtual python environment.

## Usage

To run locally, navigate to the project folder and in a terminal window type "streamlit run app.py". 

## Credits

Written by Walter Jessen

Based on https://www.youtube.com/watch?v=GIbar_kZzwk

## MIT License

Copyright (c) 2023 Walter Jessen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.