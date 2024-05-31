from spacy.cli import download
import spacy
from tools.presidio_analyzer_custom import analyze_dict
from tools.load_spacy_model_custom_recognisers import nlp_analyser
from typing import List
from unstructured.documents.elements import Element

spacy.prefer_gpu()

def spacy_model_installed(model_name):
    try:
        import en_core_web_lg
        en_core_web_lg.load()
        print("Successfully imported spaCy model")
        #nlp = spacy.load("en_core_web_sm")
        #print(nlp._path)
    except:
        download(model_name)
        spacy.load(model_name)
        print("Successfully imported spaCy model")
    #print(nlp._path)


#if not is_model_installed(model_name):
#    os.system(f"python -m spacy download {model_name}")
model_name = "en_core_web_lg"
spacy_model_installed(model_name)

#spacy.load(model_name)
# Need to overwrite version of gradio present in Huggingface spaces as it doesn't have like buttons/avatars (Oct 2023)
#os.system("pip uninstall -y gradio")
#os.system("pip install gradio==3.50.0")
#os.system("python -m spacy download en_core_web_lg")

import re
import secrets
import base64
import time

import pandas as pd

from faker import Faker

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine, BatchAnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig



def anon_consistent_names(df):
    # ## Pick out common names and replace them with the same person value
    df_dict = df.to_dict(orient="list")

    analyzer = AnalyzerEngine()
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    analyzer_results = batch_analyzer.analyze_dict(df_dict, language="en")
    analyzer_results = list(analyzer_results)

    # + tags=[]
    text = analyzer_results[3].value

    # + tags=[]
    recognizer_result = str(analyzer_results[3].recognizer_results)

    # + tags=[]
    recognizer_result

    # + tags=[]
    data_str = recognizer_result  # abbreviated for brevity

    # Adjusting the parse_dict function to handle trailing ']'
    # Splitting the main data string into individual list strings
    list_strs = data_str[1:-1].split('], [')

    def parse_dict(s):
        s = s.strip('[]')  # Removing any surrounding brackets
        items = s.split(', ')
        d = {}
        for item in items:
            key, value = item.split(': ')
            if key == 'score':
                d[key] = float(value)
            elif key in ['start', 'end']:
                d[key] = int(value)
            else:
                d[key] = value
        return d

    # Re-running the improved processing code

    result = []

    for lst_str in list_strs:
        # Splitting each list string into individual dictionary strings
        dict_strs = lst_str.split(', type: ')
        dict_strs = [dict_strs[0]] + ['type: ' + s for s in dict_strs[1:]]  # Prepending "type: " back to the split strings
        
        # Parsing each dictionary string
        dicts = [parse_dict(d) for d in dict_strs]
        result.append(dicts)

    #result

    # + tags=[]
    names = []

    for idx, paragraph in enumerate(text):
        paragraph_texts = []
        for dictionary in result[idx]:
            if dictionary['type'] == 'PERSON':
                paragraph_texts.append(paragraph[dictionary['start']:dictionary['end']])
        names.append(paragraph_texts)

    # + tags=[]
    # Flatten the list of lists and extract unique names
    unique_names = list(set(name for sublist in names for name in sublist))
    
    # + tags=[]
    fake_names = pd.Series(unique_names).apply(fake_first_name)

    # + tags=[]
    mapping_df = pd.DataFrame(data={"Unique names":unique_names,
                    "Fake names": fake_names})

    # + tags=[]
    # Convert mapping dataframe to dictionary
    # Convert mapping dataframe to dictionary, adding word boundaries for full-word match
    name_map = {r'\b' + k + r'\b': v for k, v in zip(mapping_df['Unique names'], mapping_df['Fake names'])}

    # + tags=[]
    name_map

    # + tags=[]
    scrubbed_df_consistent_names = df.replace(name_map, regex = True)

    # + tags=[]
    scrubbed_df_consistent_names

    return scrubbed_df_consistent_names

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)

def anonymise_script(text_list:List[Element], anon_strat:str, nlp_analyser=None):

    #print(df.shape)

    #df_chosen_col_mask = (df[chosen_col].isnull()) | (df[chosen_col].str.strip() == "")
    #print("Length of input series blank at start is: ", df_chosen_col_mask.value_counts())

    # DataFrame to dict
    df_dict = pd.DataFrame(data={"text":text_list}).to_dict(orient="list")   

    if nlp_analyser:
        analyzer = nlp_analyser
    else:
        analyzer = AnalyzerEngine()

    # Add titles to analyzer list
    titles_recognizer = PatternRecognizer(supported_entity="TITLE",
                                      deny_list=["Mr","Mrs","Miss", "Ms", "mr", "mrs", "miss", "ms"])

    analyzer.registry.add_recognizer(titles_recognizer)

    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    anonymizer = AnonymizerEngine()

    batch_anonymizer = BatchAnonymizerEngine(anonymizer_engine = anonymizer)

    print("Identifying personal data")
    analyse_tic = time.perf_counter()
    #analyzer_results = batch_analyzer.analyze_dict(df_dict, language="en")
    analyzer_results = analyze_dict(batch_analyzer, df_dict, language="en")
    #print(analyzer_results)
    analyzer_results = list(analyzer_results)

    analyse_toc = time.perf_counter()
    analyse_time_out = f"Analysing the text took {analyse_toc - analyse_tic:0.1f} seconds."
    print(analyse_time_out)

    # Generate a 128-bit AES key. Then encode the key using base64 to get a string representation
    key = secrets.token_bytes(16)  # 128 bits = 16 bytes 
    key_string = base64.b64encode(key).decode('utf-8')

    # Create faker function (note that it has to receive a value)
    
    fake = Faker("en_UK")

    def fake_first_name(x):
        return fake.first_name()

    # Set up the anonymization configuration WITHOUT DATE_TIME
    replace_config = eval('{"DEFAULT": OperatorConfig("replace")}')
    redact_config = eval('{"DEFAULT": OperatorConfig("redact")}')
    hash_config = eval('{"DEFAULT": OperatorConfig("hash")}')
    mask_config = eval('{"DEFAULT": OperatorConfig("mask", {"masking_char":"*", "chars_to_mask":100, "from_end":True})}')
    people_encrypt_config = eval('{"PERSON": OperatorConfig("encrypt", {"key": key_string})}') # The encryption is using AES cypher in CBC mode and requires a cryptographic key as an input for both the encryption and the decryption.
    fake_first_name_config = eval('{"PERSON": OperatorConfig("custom", {"lambda": fake_first_name})}')


    if anon_strat == "replace": chosen_mask_config = replace_config
    if anon_strat == "redact": chosen_mask_config = redact_config
    if anon_strat == "hash": chosen_mask_config = hash_config
    if anon_strat == "mask": chosen_mask_config = mask_config
    if anon_strat == "encrypt": chosen_mask_config = people_encrypt_config
    elif anon_strat == "fake_first_name": chosen_mask_config = fake_first_name_config

    # I think in general people will want to keep date / times - NOT FOR TOPIC MODELLING
    #keep_date_config = eval('{"DATE_TIME": OperatorConfig("keep")}')

    #combined_config = {**chosen_mask_config, **keep_date_config}
    combined_config = {**chosen_mask_config}#, **keep_date_config}
    combined_config

    print("Anonymising personal data")
    anonymizer_results = batch_anonymizer.anonymize_dict(analyzer_results, operators=combined_config)

    #print(anonymizer_results)

    scrubbed_df = pd.DataFrame(data={"text":anonymizer_results["text"]})

    scrubbed_series = scrubbed_df["text"]

    #print(scrubbed_series[0:6])

    #print("Length of output series is: ", len(scrubbed_series))
    #print("Length of input series at end is: ", len(df[chosen_col]))

    
    #scrubbed_values_mask = (scrubbed_series.isnull()) | (scrubbed_series.str.strip() == "")
    #df_chosen_col_mask = (df[chosen_col].isnull()) | (df[chosen_col].str.strip() == "")

    #print("Length of input series blank at end is: ", df_chosen_col_mask.value_counts())
    #print("Length of output series blank is: ", scrubbed_values_mask.value_counts())
    

    # Create reporting message
    out_message = "Successfully anonymised"
    
    if anon_strat == "encrypt":
        out_message = out_message + ". Your decryption key is " + key_string + "."
    
    return scrubbed_series, out_message

def do_anonymise(in_file:str, anon_strat:str, chosen_cols:List[str]):
    
    # Load file
    
    anon_df = pd.DataFrame()
    
    if in_file: 
        for match_file in in_file:
            match_temp_file = pd.read_csv(match_file.name, delimiter = ",", low_memory=False)#, encoding='cp1252')
            anon_df = pd.concat([anon_df, match_temp_file])
    
    # Split dataframe to keep only selected columns
    all_cols_original_order = list(anon_df.columns)
    anon_df_part = anon_df[chosen_cols]
    anon_df_remain = anon_df.drop(chosen_cols, axis = 1)
    
    # Anonymise the selected columns
    anon_df_part_out, out_message = anonymise_script(anon_df_part, anon_strat)
        
    # Rejoin the dataframe together
    anon_df_out = pd.concat([anon_df_part_out, anon_df_remain], axis = 1)
    anon_df_out = anon_df_out[all_cols_original_order]
    
    # Export file
    out_file_part = re.sub(r'\.csv', '', match_file.name)
                
    anon_export_file_name = out_file_part + "_anon_" + anon_strat + ".csv"
    
    anon_df_out.to_csv(anon_export_file_name, index = None)   
    
    return out_message, anon_export_file_name
