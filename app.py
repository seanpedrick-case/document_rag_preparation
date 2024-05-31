import os

# By default TLDExtract will try to pull files from the internet. I have instead downloaded this file locally to avoid the requirement for an internet connection.
os.environ['TLDEXTRACT_CACHE'] = 'tld/.tld_set_snapshot'

from tools.helper_functions import ensure_output_folder_exists, add_folder_to_path, custom_regex_load
from tools.unstructured_funcs import partition_file, clean_elements, export_elements_as_table_to_file, filter_elements_and_metadata, chunk_all_elements, minimum_chunk_length, start_new_chunk_after_end_of_this_element_length, hard_max_character_length_chunks, multipage_sections, overlap_all
#from tools.aws_functions import load_data_from_aws
from tools.clean_funcs import pre_clean, full_entity_list, chosen_redact_entities
import gradio as gr
import pandas as pd
import numpy as np
from typing import Type, List
from unstructured.documents.elements import Element

# Creating an alias for pandas DataFrame using Type
PandasDataFrame = Type[pd.DataFrame]

add_folder_to_path("_internal/tesseract/")
add_folder_to_path("_internal/poppler/poppler-24.02.0/Library/bin/")

ensure_output_folder_exists()

language = 'en'
default_meta_keys_to_filter=["file_directory", "filetype"]
default_element_types_to_filter = ['UncategorizedText', 'Header']


def get_element_metadata(elements, prefix=""):
    """Recursively retrieves element names and metadata in the desired format."""
    result = []

    for element in elements:
        # print("Element metadata: ", element.metadata)
        # print("Element metadata dict: ", element.metadata.__dict__)

        if hasattr(element, 'metadata') and isinstance(element.metadata.__dict__, dict):
            for key, value in element.metadata.__dict__.items():  # Iterate over key-value pairs in metadata dictionary
                    new_prefix = f"{prefix}." if prefix else ""
                    if isinstance(value, dict):  # Nested metadata
                        result.extend(get_element_metadata([value], new_prefix))  # Recurse with the nested dictionary as a single-item list
                    else:  # Leaf element
                        meta_element_to_add = f"{new_prefix}{key}"
                        if meta_element_to_add not in result:
                            result.append(meta_element_to_add)
        else:
            print(f"Warning: Element {element} does not have a metadata dictionary.")  # Handle elements without metadata gracefully

    return result

def update_filter_dropdowns(elements_table:PandasDataFrame, elements:List[Element]):
    if 'text' in elements_table.columns:
        elements_table_filt = elements_table.drop('text', axis=1)
    else:
        elements_table_filt = elements_table

    # Error handling for missing 'type' column
    if 'type' not in elements_table_filt.columns:
        print("Warning: 'type' column not found in the DataFrame.")
        return gr.Dropdown(label="Element types (not available)"), gr.Dropdown(label="Metadata properties (not available)")

    element_types_to_filter = elements_table_filt['type'].unique().tolist()
    meta_keys_to_filter = get_element_metadata(elements)

    #print("Element types:", element_types_to_filter)
    #print("Meta keys:", meta_keys_to_filter)

    element_types_to_filter_shortlist = [x for x in default_element_types_to_filter if x in element_types_to_filter]
    meta_keys_to_filter_shortlist = [x for x in default_meta_keys_to_filter if x in meta_keys_to_filter]

    return gr.Dropdown(
        value=element_types_to_filter_shortlist, choices=element_types_to_filter, multiselect=True, interactive=True, label="Choose element types to exclude from element list"
    ), gr.Dropdown(
        value=meta_keys_to_filter_shortlist, choices=meta_keys_to_filter, multiselect=True, interactive=True, label="Choose metadata keys to filter out"
    )

# Create the gradio interface

block = gr.Blocks(theme = gr.themes.Base())

with block:

    elements_state = gr.State([])
    elements_table_state = gr.State(pd.DataFrame())
    metadata_keys_state = gr.State([])
    output_image_files_state = gr.State([])
    output_file_list_state = gr.State([])
    in_colnames_state = gr.State("text")

    data_state = gr.State(pd.DataFrame())
    embeddings_state = gr.State(np.array([]))
    embeddings_type_state = gr.State("")
    topic_model_state = gr.State()
    assigned_topics_state = gr.State([])
    custom_regex_state = gr.State(pd.DataFrame())
    docs_state = gr.State()
    data_file_name_no_ext_state = gr.State()
    label_list_state = gr.State(pd.DataFrame())
    output_name_state = gr.State("")

    gr.Markdown(
    """
    # Document RAG preparation
    Extract text from documents and convert into tabular format using the Unstructured package. The outputs can then be used downstream for e.g. RAG/other processes that require tabular data. Currently supports the following file types: .pdf, .docx, .odt, .pptx, .html, text files (.txt, .md., .rst), image files (.png, .jpg, .heic), email exports (.msg, .eml), tabular files (.csv, .xlsx), or code files (.py, .js, etc.). Outputs csvs and files in a 'Document' format commonly used as input to vector databases e.g. ChromaDB, or Langchain embedding datastore integrations. See [here](https://docs.unstructured.io/open-source/core-functionality/overview) for more details about what is going on under the hood.
    """)

    with gr.Tab("Partition document"):
    
        with gr.Accordion("Upload files - accepts .pdf, .docx, .odt, .pptx, .html, text files (.txt, .md., .rst), image files (.png, .jpg, .heic), email exports (.msg, .eml), tabular files (.csv, .xlsx),  or code files (.py, .js, etc.)", open = True):
            in_file = gr.File(label="Choose file", file_count= "multiple", height=100)
            in_pdf_partition_strategy = gr.Radio(label="PDF partition strategy", value = "fast", choices=["fast", "ocr_only", "hi_res"])
        
        partition_btn = gr.Button("Partition documents (outputs appear below)", variant='primary')

        with gr.Accordion("Clean, anonymise, or filter text elements", open = False):
            with gr.Accordion("Filter element types from text and information from metadata", open = False):
                element_types_to_filter = gr.Dropdown(value=default_element_types_to_filter, choices=default_element_types_to_filter, multiselect=True, interactive=True, label = "Choose element types to exclude from element list")
                meta_keys_to_filter = gr.Dropdown(value=default_meta_keys_to_filter, choices=default_meta_keys_to_filter, multiselect=True, interactive=True, label = "Choose metadata keys to filter out")                

                filter_meta_btn = gr.Button("Filter elements/metadata")

            with gr.Accordion("Clean/anonymise text", open = False):
                with gr.Row():
                    clean_options = gr.Dropdown(choices = ["Convert bytes to string","Replace quotes","Clean non ASCII","Clean ordered list", "Group paragraphs",
                    "Remove trailing punctuation", "Remove all punctuation","Clean text","Remove extra whitespace", "Remove dashes","Remove bullets",
                    "Make lowercase"],
                    value=["Clean ordered list", "Group paragraphs", "Clean non ASCII", "Remove extra whitespace", "Remove dashes",  "Remove bullets"],
                    label="Clean options", multiselect=True, interactive=True)                    

                with gr.Accordion("Clean with custom regex", open = False):
                    gr.Markdown("""Import custom regex - csv table with one column of regex patterns with header. Example pattern: (?i)roosevelt for case insensitive removal of this term.""")
                    clean_text = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Remove custom regex.")
                    with gr.Row():
                        custom_regex = gr.UploadButton(label="Import custom regex file", file_count="multiple")
                        custom_regex_text = gr.Textbox(label="Custom regex load status")

                with gr.Accordion("Anonymise text", open = False):
                    anonymise_drop = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Anonymise data. Personal details are redacted - not 100% effective. Please check results afterwards!")
                    with gr.Row():
                        anon_strat = gr.Dropdown(value = "redact", choices=["redact", "replace"], multiselect=False, label="Anonymisation strategy. Choose from redact (simply remove text), or replace with entity type (e.g. <PERSON>)")
                        anon_entities_drop = gr.Dropdown(value=chosen_redact_entities, choices=full_entity_list, multiselect=True, label="Choose entities to find and anonymise in your open text")                        

                unstructured_clean_btn = gr.Button("Clean data")       
                
        with gr.Accordion("Chunk text", open = False):
            with gr.Row():
                chunking_method_rad = gr.Radio(value = "Chunk within title", choices = ["Chunk within title", "Basic chunking"], interactive=True)
                multipage_sections_drop =gr.Dropdown(choices=["Yes", "No"], value = "Yes", label = "Continue chunk over page breaks.", interactive=True)
                overlap_all_drop =gr.Dropdown(choices=["Yes", "No"], value = "Yes", label="Overlap over adjacent element text if needed.", interactive=True)
            with gr.Row():
                minimum_chunk_length_slide = gr.Slider(value = minimum_chunk_length, minimum=100, maximum=10000, step = 100, label= "Minimum chunk character length. Chunk will overlap next title if character limit not reached.", interactive=True)
                start_new_chunk_after_end_of_this_element_length_slide = gr.Slider(value = start_new_chunk_after_end_of_this_element_length, minimum=100, maximum=10000, step = 100, label = "'Soft' maximum chunk character length - chunk will continue until end of current element when length reached")
                hard_max_character_length_chunks_slide = gr.Slider(value = hard_max_character_length_chunks, minimum=100, maximum=10000, step = 100, label = "'Hard' maximum chunk character length. Chunk will not be longer than this.", interactive=True)

            chunk_btn = gr.Button("Chunk document")

        # Save chunked data to file
        with gr.Accordion("File outputs", open = True):
            with gr.Row():
                output_summary = gr.Textbox(label="Output summary")
                output_file = gr.File(label="Output file")

    # AWS functions not yet implemented in this app
    # with gr.Tab(label="AWS data load"):
    #     with gr.Accordion(label = "AWS data access", open = True):
    #         aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
    #         with gr.Row():
    #             in_aws_file = gr.Dropdown(label="Choose file to load from AWS (only valid for API Gateway app)", choices=["None", "Lambeth borough plan"])
    #             load_aws_data_button = gr.Button(value="Load data from AWS", variant="secondary")
                
    #         aws_log_box = gr.Textbox(label="AWS data load status")
    
    # Partition data, then Update filter dropdowns from loaded data
    partition_btn.click(fn = partition_file, inputs=[in_file, in_pdf_partition_strategy],
                    outputs=[output_summary, elements_state, output_file, output_name_state, elements_table_state], api_name="partition").\
                    then(fn = update_filter_dropdowns, inputs=[elements_table_state, elements_state], outputs=[element_types_to_filter, meta_keys_to_filter])

    # Clean data
    ## Filter metadata
    
    filter_meta_btn.click(fn=filter_elements_and_metadata, inputs=[elements_state, element_types_to_filter, meta_keys_to_filter], outputs=[elements_state]).\
    then(fn=export_elements_as_table_to_file, inputs=[elements_state, output_name_state], outputs=[output_summary, output_file])

    ## General text clean and anonymisation

    ### Custom regex load
    custom_regex.upload(fn=custom_regex_load, inputs=[custom_regex], outputs=[custom_regex_text, custom_regex_state])

    unstructured_clean_btn.click(fn=clean_elements, inputs=[elements_state, clean_options, output_name_state], outputs=[elements_state, output_summary, output_file, output_name_state]).\
    then(fn=pre_clean, inputs=[elements_state, in_colnames_state, custom_regex_state, clean_text, output_name_state, anonymise_drop, anon_strat, anon_entities_drop], outputs=[output_summary, output_file, elements_state, output_name_state])

    ## Chunk data
    chunk_btn.click(fn = chunk_all_elements, inputs=[elements_state, output_name_state, chunking_method_rad, minimum_chunk_length_slide, start_new_chunk_after_end_of_this_element_length_slide, hard_max_character_length_chunks_slide, multipage_sections_drop, overlap_all_drop], outputs=[output_summary, output_file, output_name_state])
    
    # Loading AWS data - not yet implemented in this app
    # load_aws_data_button.click(fn=load_data_from_aws, inputs=[in_aws_file, aws_password_box], outputs=[in_file, aws_log_box])
    
# Simple run
block.queue().launch(ssl_verify=False) # root_path="/address-match", debug=True, server_name="0.0.0.0", server_port=7861
