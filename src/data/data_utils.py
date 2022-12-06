import re
from difflib import SequenceMatcher

import codecs
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from text_unidecode import unidecode
from tqdm.notebook import tqdm

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)



def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def clean_text(text):
    text = text.replace(u'\x9d', u' ')
    text = resolve_encodings_and_normalize(text)
    # text = text.replace(u'\xa0', u' ')
    # text = text.replace(u'\x85', u'\n')
    text = text.strip()
    return text

def add_text_to_df(test_df,data_folder):
    mapper = {}
    for idx in tqdm(test_df.essay_id.unique()):
        with open(data_folder/f'{idx}.txt','r') as f:
            texte = clean_text(f.read())
            # texte = resolve_encodings_and_normalize(f.read())
            # texte = texte.strip() 
        mapper[idx] = texte

    test_df['discourse_ids'] = np.arange(len(test_df))
    test_df['essay_text'] = test_df['essay_id'].map(mapper)
    test_df['discourse_text'] = test_df['discourse_text'].transform(clean_text)
    test_df['discourse_text'] = test_df['discourse_text'].str.strip()

    test_df['previous_discourse_end'] = 0
    test_df['st_ed'] = test_df.apply(get_start_end('discourse_text'),axis=1)
    test_df['discourse_start'] = test_df['st_ed'].transform(lambda x:x[0])
    test_df['discourse_end'] = test_df['st_ed'].transform(lambda x:x[1])
    test_df['previous_discourse_end'] = test_df.groupby("essay_id")['discourse_end'].transform(lambda x:x.shift(1).fillna(0)).astype(int)
    test_df['st_ed'] = test_df.apply(get_start_end('discourse_text'),axis=1)
    test_df['discourse_start'] = test_df['st_ed'].transform(lambda x:x[0]) #+ test_df['previous_discourse_end']
    test_df['discourse_end'] = test_df['st_ed'].transform(lambda x:x[1]) #+ test_df['previous_discourse_end']

    if 'target' in test_df.columns:
        classe_mapper = {'Ineffective':0,"Adequate":1,"Effective":2}
        test_df['target'] = test_df['discourse_effectiveness'].map(classe_mapper)
        
    else:
        test_df['target'] = 1 

    return test_df


def get_text_start_end(txt,s,search_from=0):
    txt = txt[int(search_from):]
    try:
        idx = txt.find(s)
        if idx>=0:
            st=idx
            ed = st+len(s)
        else:
            raise ValueError('Error')
    except:                
        res = [(m.start(0), m.end(0)) for m in re.finditer(s, txt)]
        if len(res):
            st,ed = res[0][0],res[0][1]
        else:
            m = SequenceMatcher(None, s,txt).get_opcodes()
            for tag,i1,i2,j1,j2 in m:
                if tag=='replace':
                    s = s[:i1]+txt[j1:j2]+s[i2:]
                if tag=="delete":
                    s = s[:i1]+s[i2:]
            
            res = [(m.start(0), m.end(0)) for m in re.finditer(s,txt)]
            if len(res):
                st,ed = res[0][0],res[0][1]
            else:
                idx = txt.find(s)
                if idx>=0:
                    st=idx
                    ed = st+len(s)
                else:
                    st,ed = 0,0
    return st+search_from,ed+search_from

def get_start_end(col):
    def search_start_end(row):
        txt = row.essay_text
        search_from = row.previous_discourse_end
        s = row[col]
        # print(search_from)
        return get_text_start_end(txt,s,search_from)
    return search_start_end

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def text_to_words(text):
    word = text.split()
    word_offset = []

    start = 0
    for w in word:
        r = text[start:].find(w)

        if r==-1:
            raise NotImplementedError
        else:
            start = start+r
            end   = start+len(w)
            word_offset.append((start,end))
        start = end

    return word, word_offset

def text_to_sentence(text):
    sentences = re.split(r' *[\.\?!\n][\'"\)\]]* *', text)
    sentences = [x for x in sentences if x!=""]
    
    sentence_offset = []
    start = 0
    for w in sentences:
        r = text[start:].find(w)

        if r==-1:
            raise NotImplementedError
        else:
            start = start+r
            end   = start+len(w)
            sentence_offset.append((start,end))
        start = end

    return sentences,sentence_offset

def text_to_paragraph(text):
    sentences = re.split(r' *[\n][\'"\)\]]* *', text)
    sentences = [x for x in sentences if x!=""]
    
    sentence_offset = []
    start = 0
    for w in sentences:
        r = text[start:].find(w)

        if r==-1:
            raise NotImplementedError
        else:
            start = start+r
            end   = start+len(w)
            sentence_offset.append((start,end))
        start = end

    return sentences,sentence_offset


def get_span_from_text(text,span_type="words"):
    
    if span_type=="words":
        spans,spans_offset = text_to_words(text)
    elif span_type=="sentences":
        spans,spans_offset = text_to_sentence(text)
    else:
        spans,spans_offset = text_to_paragraph(text)
    
    return spans,spans_offset