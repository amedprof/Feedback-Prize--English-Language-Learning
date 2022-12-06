import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from data.data_utils import clean_text,get_span_from_text

## =============================================================================== ##
class FeedbackDataset(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 max_length,
                 target = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar','conventions'],
                 spans = "",
                ):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_tokens = [tokenizer.cls_token_id,
                               tokenizer.sep_token_id,
                               tokenizer.pad_token_id,
                               tokenizer.mask_token_id
                              ]
        
        self.target = target
        df = self.prepare_df(df)
        
        self.items = []
        self.texts = []
        self.text_id = []
        self.targets = []
        self.spans_offset = []


        if len(self.tokenizer.encode("\n\n"))==2:
            df["full_text"] = df['full_text'].transform(lambda x:x.str.replace("\n\n"," | "))

        for text_id,g in df.sort_values(["full_text_len"]).groupby('text_id',sort=False):
            self.texts.append(g['full_text'].values[0])
            self.text_id.append(text_id)
            self.targets.append(g[self.target].values.tolist())
            if spans!="":
                self.spans_offset.append(get_span_from_text(g['full_text'].values[0],spans)[1])

        for idx in tqdm(range(len(self.texts))):
            self.items.append(self.make_one_item(idx))
    
        self.df = df


    def prepare_df(self,df):
        df["full_text"] = df['full_text'].astype(str).fillna('').apply(clean_text)
        if "cohesion" not in df.columns:
            df[self.target] = 0
            
        df['full_text_len'] = df['full_text'].astype(str).fillna('').apply(len)

        return df

    def make_one_item(self,idx):

        encoding = self.tokenizer(
                                    self.texts[idx],
                                    truncation=True if self.max_length else False,
                                    max_length=self.max_length,
                                    add_special_tokens = True,
                                    return_offsets_mapping=False if not len(self.spans_offset) else True,
                                )
        
        outputs = dict(**encoding)
        
        outputs['label'] = self.targets[idx]*len(outputs['input_ids'])

        if len(self.spans_offset):
            target_idx = np.zeros(len(outputs['offset_mapping']))-1
            for lab_id,(start_span, end_span) in enumerate(self.spans_offset[idx]):
                for i,(s,e) in enumerate(outputs['offset_mapping']):
                    if min(end_span, e) - max(start_span, s) > 0:
                        target_idx[i] = lab_id+1

            outputs['span_labels'] = target_idx.tolist()
        else:
            outputs['span_labels'] = [1]*len(outputs['input_ids'])

        return outputs
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self,idx):
        return self.items[idx]


## =============================================================================== ##
## =============================================================================== ##
class CustomCollator():
    def __init__(self, tokenizer,num_target=6,inference=False):
        self.tokenizer = tokenizer
        self.num_target = num_target
        self.inference = inference

    def __call__(self, batch):
        output = dict()

        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if not self.inference:
            output["label"] = [sample["label"] for sample in batch]
        output["span_labels"] = [sample["span_labels"] for sample in batch]


        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            if not self.inference:
            
                output["label"] = [s + (batch_max - len(s)) * [[-1.]*self.num_target] for s in output["label"]]
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            output["span_labels"] = [s + (batch_max - len(s)) * [-1] for s in output["span_labels"]]

        else:
            if not self.inference:
                
                output["label"] = [(batch_max - len(s)) * [[-1.]*self.num_target] + s for s in output["label"]]
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]
            output["span_labels"] = [(batch_max - len(s)) * [-1] + s for s in output["span_labels"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if not self.inference:
            output["label"] = torch.tensor(output["label"], dtype=torch.float)
        output["span_labels"] = torch.tensor(output["span_labels"], dtype=torch.long)
        return output