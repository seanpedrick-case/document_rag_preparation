---
title: Document RAG preparation
emoji: 📖
colorFrom: yellow
colorTo: purple
sdk: docker
app_file: app.py
pinned: true
license: apache-2.0
---

# Document RAG preparation

Extract text from documents and convert into tabular format using the Unstructured package. The outputs can then be used downstream for e.g. RAG/other processes that require tabular data. Currently supports the following file types: .pdf, .docx, .odt, .pptx, .html, text files (.txt, .md., .rst), image files (.png, .jpg, .heic), email exports (.msg, .eml), tabular files (.csv, .xlsx),  code files (.py, .js etc.). Outputs csvs and files in a 'Document' format commonly used as input to vector databases e.g. ChromaDB, or Langchain embedding datastore integrations. See [here](https://docs.unstructured.io/open-source/core-functionality/overview) for more details about what is going on under the hood.