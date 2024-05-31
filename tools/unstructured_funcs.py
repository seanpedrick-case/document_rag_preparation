from unstructured.partition.auto import partition 
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements 
from unstructured.documents.elements import Element, Title, CompositeElement
from unstructured.staging.base import convert_to_dataframe
from typing import Type, List, Literal, Tuple

from unstructured.cleaners.core import replace_unicode_quotes, clean_non_ascii_chars, clean_ordered_bullets, group_broken_paragraphs, replace_unicode_quotes, clean, clean_trailing_punctuation, remove_punctuation, bytes_string_to_string
import gradio as gr
import time
import pandas as pd
import re
import gzip
import pickle
from pydantic import BaseModel, Field

from tools.helper_functions import get_file_path_end, get_file_path_end_with_ext

# Creating an alias for pandas DataFrame using Type
PandasDataFrame = Type[pd.DataFrame]

# %%
# pdf partitioning strategy vars
pdf_partition_strat = "ocr_only" # ["fast", "ocr_only", "hi_res"]

# %%
# Element metadata modification vars
meta_keys_to_filter = ["file_directory", "filetype"]
element_types_to_filter = ['UncategorizedText', 'Header']

# %%
# Clean function vars

bytes_to_string=False
replace_quotes=True 
clean_non_ascii=False 
clean_ordered_list=True 
group_paragraphs=True
trailing_punctuation=False
all_punctuation=False
clean_text=True 
extra_whitespace=True 
dashes=True 
bullets=True 
lowercase=False

# %%
# Chunking vars

minimum_chunk_length = 2000
start_new_chunk_after_end_of_this_element_length = 2000
hard_max_character_length_chunks = 3000
multipage_sections=True
overlap_all=True
include_orig_elements=True

# %%
class Document(BaseModel):
    """Class for storing a piece of text and associated metadata. Implementation adapted from Langchain code: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py"""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"

# %%
def create_title_id_dict(elements:List[Element]):

    # Assuming the object is stored in a variable named 'elements_list'
    titles = [item.text for item in elements if isinstance(item, Title)]

    #### Get all elements under these titles
    chapter_ids = {}
    for element in elements:
        for chapter in titles:
            if element.text == chapter and element.category == "Title":
                chapter_ids[element._element_id] = chapter
                break

    chapter_to_id = {v: k for k, v in chapter_ids.items()}

    return chapter_ids, chapter_to_id

# %%
def filter_elements(elements:List[Element], excluded_elements: List[str] = ['']):
    """
    Filter out elements from a list based on their categories.

    Args:
        elements: The list of elements to filter.
        excluded_elements: A list of element categories to exclude.

    Returns:
        A new list containing the filtered elements.
    """
    filtered_elements = []
    for element in elements:
        if element.category not in excluded_elements:
            filtered_elements.append(element)
    return filtered_elements

# %%
def remove_keys_from_meta(
    elements: List[Element], 
    meta_remove_keys: List[str], 
    excluded_element_types: List[str] = []
) -> List[Element]:
    '''
    Remove specified metadata keys from an Unstructured Element object
    '''

    for element in elements:
        if element.category not in excluded_element_types:
            for key in meta_remove_keys:
                try:
                    del element.metadata.__dict__[key]  # Directly modify metadata
                except KeyError:
                    print(f"Key '{key}' not found in element metadata.")

    return elements

def filter_elements_and_metadata(
    elements: List[Element],
    excluded_categories: List[str] = [],
    meta_remove_keys: List[str] = [],
) -> List[Element]:
    """
    Filters elements based on categories and removes specified metadata keys.

    Args:
        elements: The list of elements to process.
        excluded_categories: A list of element categories to exclude.
        meta_remove_keys: A list of metadata keys to remove.

    Returns:
        A new list containing the processed elements.
    """

    filtered_elements = []
    for element in elements:
        if element.category not in excluded_categories:
            for key in meta_remove_keys:
                try:
                    del element.metadata.__dict__[key]
                except KeyError:
                    # Better logging/error handling instead of just printing
                    # Use a proper logger or raise a warning/exception
                    pass 
            filtered_elements.append(element)

    return filtered_elements

# %%
def add_parent_title_to_meta(elements:List[Element], chapter_ids:List[str], excluded_element_types:List[str]=['']) -> List[Element]:
    '''
    Add parent title to Unstructured metadata elements
    
    '''
    for element in elements:
        if element.category in excluded_element_types:
            pass

        else:
            meta = element.metadata.to_dict()
            
            if "parent_id" in meta and meta["parent_id"] in chapter_ids and "title_name" not in meta:
                title_name = chapter_ids[meta["parent_id"]]
                # Directly modify the existing element metadata object
                element.metadata.title_name = title_name

    return elements


def chunk_all_elements(elements:List[Element], file_name_base:str, chunk_type:str = "Basic_chunking",  minimum_chunk_length:int=minimum_chunk_length, start_new_chunk_after_end_of_this_element_length:int=start_new_chunk_after_end_of_this_element_length, hard_max_character_length_chunks:int=hard_max_character_length_chunks, multipage_sections:bool=multipage_sections, overlap_all:bool=overlap_all, include_orig_elements:bool=include_orig_elements):

    '''
    Use Unstructured.io functions to chunk an Element object by Title or across all elements.
    '''
    output_files = []
    output_summary = ""

    chapter_ids, chapter_to_id = create_title_id_dict(elements)
    
    ### Break text down into chunks

    try:

        if chunk_type == "Chunk within title":
            chunks = chunk_by_title(
                elements,
                include_orig_elements=include_orig_elements,
                combine_text_under_n_chars=minimum_chunk_length,
                new_after_n_chars=start_new_chunk_after_end_of_this_element_length,
                max_characters=hard_max_character_length_chunks,
                multipage_sections=multipage_sections,
                overlap_all=overlap_all
            )

        else:
            chunks = chunk_elements(
                elements,
                include_orig_elements=include_orig_elements,
                new_after_n_chars=start_new_chunk_after_end_of_this_element_length,
                max_characters=hard_max_character_length_chunks,
                overlap_all=overlap_all
            )
    
    except Exception as output_summary:
        print(output_summary)
        return output_summary, output_files, file_name_base

    chunk_sections, chunk_df, chunks_out = element_chunks_to_document(chunks, chapter_ids)

    file_name_suffix = "_chunk"

    # The new file name does not overwrite the old file name as the 'chunked' elements are only used as an output, and not an input to other functions
    output_summary, output_files, file_name_base_new = export_elements_as_table_to_file(chunks_out, file_name_base, file_name_suffix, chunk_sections)

    return output_summary, output_files, file_name_base

# %%
def element_chunks_to_document(chunks:CompositeElement, chapter_ids:List[str]) -> Tuple[List[Document], PandasDataFrame, List[str]]:
    '''
    Take an Unstructured.io chunk_by_title output with the original parsed document elements and turn it into a Document format commonly used by vector databases, and a Pandas dataframe. 
    '''
    chunk_sections = []
    current_title_id = ''
    current_title = ''
    last_page = ''
    chunk_df_list = []

    for chunk in chunks:
        chunk_meta = chunk.metadata.to_dict()
        true_element_ids = []
        element_categories = []
        titles = []
        titles_id = []        

        if "page_number" in chunk_meta:
            last_page = chunk_meta["page_number"]

        chunk_text = chunk.text
        #chunk_page_number = chunk.metadata.to_dict()["page_number"]

        # If the same element text is found, add the element_id to the chunk (NOT PERFECT. THIS WILL FAIL IF THE SAME TEXT IS SEEN MULTIPL TIMES)
        for element in chunk.metadata.orig_elements:
            
            #element_text = element.text
            element_id = element._element_id
            element_category = element.category
            element_meta = element.metadata.to_dict()

            if "page_number" in element_meta:
                element_page_number = element_meta["page_number"]
                last_page = element_page_number

            true_element_ids.append(element_id)
            element_categories.append(element_category)
            

        # Set new metadata for chunk
        if "page_number" in element_meta:
            chunk_meta["last_page_number"] = last_page
        
        chunk_meta["true_element_ids"] = true_element_ids        

        for loop_id in chunk_meta['true_element_ids']:
            if loop_id in chapter_ids:
                current_title = chapter_ids[loop_id]
                current_title_id = loop_id

                titles.append(current_title)
                titles_id.append(current_title_id)        
                
        chunk_meta['titles'] = titles
        chunk_meta['titles_id'] = titles_id

        # Remove original elements data for documents
        chunk_meta.pop('orig_elements')

        chunk_dict_for_df = chunk_meta.copy()
        chunk_dict_for_df['text'] = chunk.text

        chunk_df_list.append(chunk_dict_for_df)

        
        chunk_doc = [Document(page_content=chunk_text, metadata=chunk_meta)]
        chunk_sections.extend(chunk_doc)

        ## Write metadata back to elements
        chunk.metadata.__dict__ = chunk_meta

    chunk_df = pd.DataFrame(chunk_df_list)

    # print("Doc format: ", chunk_sections)

    return chunk_sections, chunk_df, chunks

# %%
def write_elements_to_documents(elements:List[Element]):
    '''
    Take Unstructured.io parsed elements and write it into a 'Document' format commonly used by vector databases
    '''

    doc_sections = []

    for element in elements:
        meta = element.metadata.to_dict()

        meta["type"] = element.category
        meta["element_id"] = element._element_id

        element_doc = [Document(page_content=element.text, metadata= meta)]
        doc_sections.extend(element_doc)

        #print("Doc format: ", doc_sections)


    return doc_sections

# %%
def clean_elements(elements:List[Element], dropdown_options: List[str] = [''], 
                       output_name:str = "combined_elements",
                       bytes_to_string:bool=False,
                       replace_quotes:bool=True, 
                       clean_non_ascii:bool=False, 
                       clean_ordered_list:bool=True, 
                       group_paragraphs:bool=True,
                       trailing_punctuation:bool=False,
                       all_punctuation:bool=False,
                       clean_text:bool=True, 
                       extra_whitespace:bool=True, 
                       dashes:bool=True, 
                       bullets:bool=True, 
                       lowercase:bool=False) -> List[Element]:
    
    '''
    Apply Unstructured cleaning processes to a list of parse elements.
    '''

    out_files = []
    output_summary = ""

    # Set variables to True based on dropdown selections
    for option in dropdown_options:
        if option == "Convert bytes to string":
            bytes_to_string = True
        elif option == "Replace quotes":
            replace_quotes = True
        elif option == "Clean non ASCII":
            clean_non_ascii = True
        elif option == "Clean ordered list":
            clean_ordered_list = True
        elif option == "Group paragraphs":
            group_paragraphs = True
        elif option == "Remove trailing punctuation":
            trailing_punctuation = True
        elif option == "Remove all punctuation":
            all_punctuation = True
        elif option == "Clean text":
            clean_text = True
        elif option == "Remove extra whitespace":
            extra_whitespace = True
        elif option == "Remove dashes":
            dashes = True
        elif option == "Remove bullets":
            bullets = True
        elif option == "Make lowercase":
            lowercase = True
           

    cleaned_elements = elements.copy()

    for element in cleaned_elements:

        try:
            if element:  # Check if element is not None or empty
                if bytes_to_string:
                    element.apply(bytes_string_to_string)
                if replace_quotes:
                    element.apply(replace_unicode_quotes)
                if clean_non_ascii:
                    element.apply(clean_non_ascii_chars)
                if clean_ordered_list:
                    element.apply(clean_ordered_bullets)
                if group_paragraphs:
                    element.apply(group_broken_paragraphs)
                if trailing_punctuation:
                    element.apply(clean_trailing_punctuation)
                if all_punctuation:
                    element.apply(remove_punctuation)
                if group_paragraphs:
                    element.apply(group_broken_paragraphs)
                if clean_text:
                    element.apply(lambda x: clean(x, extra_whitespace=extra_whitespace, dashes=dashes, bullets=bullets, lowercase=lowercase))
        except Exception as e:
            print(e)
            element = element

    alt_out_message, out_files, output_file_base = export_elements_as_table_to_file(cleaned_elements, output_name, file_name_suffix="_clean")

    output_summary = "Text elements successfully cleaned."
    print(output_summary)

    return cleaned_elements, output_summary, out_files, output_file_base

# %% [markdown]
def export_elements_as_table_to_file(elements:List[Element], file_name_base:str, file_name_suffix:str="", chunk_documents:List[Document]=[]):
    '''
    Export elements as as a table.
    '''
    output_summary = ""
    out_files = []

    # Convert to dataframe format
    out_table = convert_to_dataframe(elements)

    # If the file suffix already exists in the output file name, don't add it again.
    if file_name_suffix not in file_name_base:
        out_file_name_base = file_name_base + file_name_suffix

    else:
        out_file_name_base = file_name_base
        
    out_file_name = "output/" + out_file_name_base + ".csv"

    out_table.to_csv(out_file_name)
    out_files.append(out_file_name)

    # Convert to document format
    if chunk_documents:
        out_documents = chunk_documents
    else:
        out_documents = write_elements_to_documents(elements)

    

    out_file_name_docs = "output/" + out_file_name_base + "_docs.pkl.gz"
    with gzip.open(out_file_name_docs, 'wb') as file:
        pickle.dump(out_documents, file)

    out_files.append(out_file_name_docs)

    output_summary = "File successfully exported."

    return output_summary, out_files, out_file_name_base

# # Partition PDF

def get_file_type(filename):
    pattern = r"\.(\w+)$"  # Match a dot followed by one or more word characters at the end of the string

    match = re.search(pattern, filename)
    if match:
        file_type = match.group(1)  # Extract the captured file type (without the dot)
        print(file_type)  # Output: "png"
    else:
        print("No file type found.")

    return file_type 

# %%
def partition_file(filenames:List[str], pdf_partition_strat:str = pdf_partition_strat, progress = gr.Progress()):
    '''
    Partition document files into text elements using the Unstructured package. Currently supports PDF, docx, pptx, html, several image file types, text document types, email messages, code files.
    '''

    out_message = ""
    combined_elements = []
    out_files = []

    for file in progress.tqdm(filenames, desc="Partitioning files", unit="files"):

        try:

            tic = time.perf_counter()
            print(file)

            file_name = get_file_path_end_with_ext(file)
            file_name_base = get_file_path_end(file)
            file_type = get_file_type(file_name)

            image_file_type_list = ["jpg", "jpeg", "png", "heic"]

            if file_type in image_file_type_list:
                print("File is an image. Using OCR method to partition.")
                file_elements = partition(file, strategy="ocr_only")
            else:
                file_elements = partition(file, strategy=pdf_partition_strat)

            toc = time.perf_counter()


            new_out_message = f"Successfully partitioned file: {file_name} in {toc - tic:0.1f} seconds\n"
            print(new_out_message)

            out_message = out_message + new_out_message
            combined_elements.extend(file_elements)

        except Exception as e:
            new_out_message = f"Failed to partition file:  {file_name} due to {e}. Partitioning halted."
            print(new_out_message)
            out_message = out_message + new_out_message
            break

    out_table = convert_to_dataframe(combined_elements)

    # If multiple files, overwrite default file name for outputs
    if len(filenames) > 1:
        file_name_base = "combined_files"

    alt_out_message, out_files, output_file_base = export_elements_as_table_to_file(combined_elements, file_name_base, file_name_suffix="_elements")

    return out_message, combined_elements, out_files, output_file_base, out_table
        
# %%
def modify_metadata_elements(elements_out_cleaned:List[Element], meta_keys_to_filter:List[str]=meta_keys_to_filter, element_types_to_filter:List[str]=element_types_to_filter) -> List[Element]:

    '''
    Take an element object, add parent title names to metadata. Remove specified metadata keys or element types from element list.
    '''

    chapter_ids, chapter_to_id = create_title_id_dict(elements_out_cleaned.copy())
    elements_out_meta_mod = add_parent_title_to_meta(elements_out_cleaned.copy(), chapter_ids)
    elements_out_meta_mod_meta_filt = remove_keys_from_meta(elements_out_meta_mod.copy(), meta_keys_to_filter)
    elements_out_filtered_meta_mod = filter_elements(elements_out_meta_mod_meta_filt, element_types_to_filter)

    return elements_out_filtered_meta_mod
# %%
# file_stub = "C:/Users/SPedrickCase/OneDrive - Lambeth Council/Apps/doc_rag_prep/examples/"
# filenames = []
# pdf_filename = [file_stub + "Lambeth_2030-Our_Future_Our_Lambeth_foreword.pdf"]
# filenames.extend(pdf_filename)

# html_filename = [file_stub + "transport-strategy.html"]
# filenames.extend(html_filename)

# docx_filename = [file_stub + "FINAL Policy and Procedure for Writing Housing Policies.docx"]
# filenames.extend(docx_filename)

# out_message, elements_parse = partition_file(filenames=filenames, pdf_partition_strat="ocr_only")

# for element in elements_parse[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")
#     elements_out = elements_parse.copy()

# %% [markdown]
# ###  Process with document layout detection - fast strategy
# 
# The "fast" strategy will extract the text using pdfminer and process the raw text with partition_text. If the PDF text is not extractable, partition_pdf will fall back to "ocr_only". We recommend using the "fast" strategy in most cases where the PDF has extractable text.
# elements_out_parse = partition_pdf(filename=filename, strategy="fast")
# for element in elements_out_parse[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")
#  elements_out = elements_out_parse.copy()
# ### OCR only
# 
# The "ocr_only" strategy runs the document through Tesseract for OCR and then runs the raw text through partition_text. Currently, "hi_res" has difficulty ordering elements for documents with multiple columns. If you have a document with multiple columns that does not have extractable text, we recommend using the "ocr_only" strategy. "ocr_only" falls back to "fast" if Tesseract is not available and the document has extractable text.
#  elements_out_parse = partition_pdf(filename=filename, strategy="ocr_only")
#  for element in elements_out_parse[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")
#     elements_out = elements_out_parse.copy()
# ### Hi-res partitioning
# 
# The "hi_res" strategy will identify the layout of the document using detectron2. The advantage of “hi_res” is that it uses the document layout to gain additional information about document elements. We recommend using this strategy if your use case is highly sensitive to correct classifications for document elements. If detectron2 is not available, the "hi_res" strategy will fall back to the "ocr_only" strategy.
# elements_out = partition_pdf(filename=filename, strategy="hi_res")
# for element in elements_out[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")

# %% [markdown]
# ## Clean data

# %%
# elements_out_cleaned = clean_elements(elements_out.copy(), bytes_to_string=False,
# replace_quotes=True ,
# clean_non_ascii=False, 
# clean_ordered_list=True ,
# group_paragraphs=True,
# trailing_punctuation=False,
# all_punctuation=False,
# clean_text=True ,
# extra_whitespace=True, 
# dashes=True ,
# bullets=True ,
# lowercase=False)

# %% [markdown]
# ## Add/remove elements to/from metadata



# %% [markdown]
# ### Write to table, dictionary, document format

# %%
### Dataframe format

# elements_out_filtered_df = convert_to_dataframe(elements_out_filtered_meta_mod)

# elements_out_filtered_df.to_csv("table.csv")
# elements_out_filtered_df.head(6)

# # %%
# ### Dictionary format

# elements_out_filtered_dict = convert_to_dict(elements_out_filtered_meta_mod)
# elements_out_filtered_dict[20]

# # %% [markdown]
# # ### Document format for embeddings

# # %%
# doc_sections = write_elements_to_documents(elements_out_filtered_meta_mod, element_types_to_filter)

# doc_sections[0:10]

# # %% [markdown]
# # ### Break text down into chunks

# # %%
# chunks_by_title = chunk_by_title(
#     elements_out_filtered_meta_mod,
#     include_orig_elements=True,
#     combine_text_under_n_chars=minimum_chunk_length,
#     new_after_n_chars=start_new_chunk_after_end_of_this_element_length,
#     max_characters=hard_max_character_length_chunks,
#     multipage_sections=True,
#     overlap_all=True
# )

# chunk_sections, chunk_df = element_chunks_to_document(chunks_by_title, chapter_ids)
# chunk_df.to_csv("chunked_df.csv")
# print(chunk_sections[2])

# # %%
# chunks_basic = chunk_elements(
#     elements_out_filtered_meta_mod,
#     include_orig_elements=True,
#     new_after_n_chars=start_new_chunk_after_end_of_this_element_length,
#     max_characters=hard_max_character_length_chunks,
#     overlap_all=True
# )

# chunk_basic_sections, chunk_basic_df = element_chunks_to_document(chunks_basic, chapter_ids)
# chunk_basic_df.to_csv("chunked_basic_df.csv")

# %% [markdown]
# # Partition Word document
# 
# You cannot get location metadata for bounding boxes from word documents

# %%
# word_filename = "../examples/FINAL Policy and Procedure for Writing Housing Policies.docx"

# # %%
# docx_elements = partition(filename=word_filename)
# for element in docx_elements:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")

# # %%
# docx_elements[5].text

# # %%
# docx_elements[5].category

# # %%
# docx_elements[5].metadata.to_dict()

# # %% [markdown]
# # ## Find elements associated with chapters

# # %%
# chapter_ids, chapter_to_id = create_title_id_dict(docx_elements)

# chapter_ids

# # %%
# doc_sections = write_elements_to_documents(docx_elements.copy(), chapter_ids)

# # %%
# doc_sections

# # %% [markdown]
# # ### Chunk documents

# # %%
# chunks = chunk_by_title(
#     docx_elements,
#     include_orig_elements=False,
#     combine_text_under_n_chars=0,
#     new_after_n_chars=500,
#     max_characters=1000,
#     multipage_sections=True,
#     overlap_all=True
# )

# # %%
# print(chunks)

# # %%
# chunk_sections = element_chunks_to_document(chunks.copy(), docx_elements.copy(), chapter_ids)

# # %%
# chunk_sections[5].page_content

# # %%
# chunk_sections[5].metadata["true_element_ids"]

# # %%
# for element in docx_elements:
#     if element._element_id in chunk_sections[5].metadata["true_element_ids"]:
#         print(element.text)

# # %% [markdown]
# # # Partition PPTX document

# # %%
# pptx_filename = "../examples/LOTI presentation Jan 2024.pptx"

# # %%
# pptx_elements = partition(filename=pptx_filename)
# for element in pptx_elements[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")

# # %%
# chapter_ids, chapter_to_id = create_title_id_dict(pptx_elements)
# chapter_ids

# # %%
# pptx_sections = write_elements_to_documents(pptx_elements.copy(), chapter_ids)

# # %%
# pptx_sections

# # %%
# pptx_chunks = chunk_by_title(
#     pptx_elements,
#     include_orig_elements=False,
#     combine_text_under_n_chars=0,
#     new_after_n_chars=500,
#     max_characters=1000,
#     multipage_sections=True,
#     overlap_all=True
# )

# # %%
# pptx_chunk_sections = element_chunks_to_document(pptx_chunks.copy(), pptx_elements.copy(), chapter_ids)

# # %% [markdown]
# # ### Load documents into a vectorDB (Not necessary)

# # %%
# import chromadb

# # %%
# client = chromadb.PersistentClient(path="chroma_tmp", settings=chromadb.Settings(allow_reset=True))
# client.reset()

# # %%
# collection = client.create_collection(
#     name="policy_statements",
#     metadata={"hnsw:space": "cosine"}
# )

# # %%
# chapter_ids

# # %%
# for element in docx_elements:
#     parent_id = element.metadata.parent_id
#     #print(element.text)
#     #print(parent_id)
#     #print(element.metadata.to_dict())
#     if parent_id:
#         try:
#             print(parent_id)
#             chapter = chapter_ids[parent_id]
#             print(chapter)
#         except KeyError:
#             chapter = "None"
#     else:
#         chapter = "None"
#     collection.add(
#         documents=[element.text],
#         ids=[element._element_id],
#         metadatas=[{"chapter": chapter}]
#     )

# # %% [markdown]
# # #### See the elements in the VectorDB and perform hybrid search

# # %%
# results = collection.peek()
# print(results["documents"])

# # %%
# print(collection.metadata)

# # %%
# import json

# result = collection.query(
#     query_texts=["What should policies do?"],
#     n_results=2,
#     where={"chapter": '3.0  Policy Statements'},
# )
# print(json.dumps(result, indent=2))

# # %%
# collection = client.create_collection(
#     name="policy_statements_chunk",
#     metadata={"hnsw:space": "cosine"}
# )

# # %%
# for element in chunks:
#     parent_id = element.metadata.parent_id
#     #print(element.text)
#     #print(parent_id)
#     #print(element.metadata.to_dict())
#     if parent_id:
#         try:
#             print(parent_id)
#             chapter = chapter_ids[parent_id]
#             print(chapter)
#         except KeyError:
#             chapter = "None"
#     else:
#         chapter = "None"

#     print(element._element_id)
#     collection.add(
#         documents=[element.text],
#         ids=[element.orig_elements],
#         metadatas=[{"chapter": chapter}]
#     )

# # %% [markdown]
# # # Partition HTML

# # %%
# html_filename = "../examples/transport-strategy.html"

# # %%
# html_elements = partition(filename=html_filename)
# for element in html_elements[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")

# # %% [markdown]
# # # Partition image

# # %%
# img_filename = "../examples/example_complaint_letter.jpg"

# # %%
# img_elements = partition(filename=img_filename)
# for element in img_elements[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")

# # %% [markdown]
# # # Partition XLSX

# # %%
# xlsx_filename = "../examples/fuel-poverty-sub-regional-tables-2020-2018-data.xlsx"

# # %%
# xlsx_elements = partition(filename=xlsx_filename)
# for element in xlsx_elements[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")

# # %% [markdown]
# # # Partition .py

# # %%
# py_filename = "../examples/app.py"

# # %%
# py_elements = partition(filename=py_filename)
# for element in py_elements[:10]:
#     print(f"{element.category.upper()}: {element.text} - Metadata: {element.metadata.to_dict()}")


