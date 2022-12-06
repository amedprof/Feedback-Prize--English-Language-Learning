import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.utils.checkpoint
import torch.nn.functional as F
from model_zoo.pooling import NLPPooling

import gc
       
class FeedbackModel(nn.Module):
    def __init__(self,
                 model_name,
                 num_labels,
                 config_path=None,
                 use_gradient_checkpointing = False,
                 pooling_params={},
                 spans_pooling_params = {},
                 window_size = 1024,
                 spans = "",
                 striding_wind = False
                 ):
        super().__init__()
        self.striding_wind = striding_wind
        self.edge_len = 64
        self.inner_len = 384
        self.window_size = window_size
        self.spans = spans
        self.span_pool = False if spans=="" else True


        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True) if not config_path else torch.load(config_path)
        self.config.update(
                                
                        {
                            "hidden_dropout_prob": 0.0,
                            "attention_probs_dropout_prob": 0.0,
                        }
                            )
        
        self.backbone = AutoModel.from_pretrained(model_name,config=self.config) if not config_path else AutoModel.from_config(self.config)
        


        self.pooling_params = pooling_params
        self.pooling_params.update({"in_features":self.config.hidden_size,
                                    "out_features":self.config.hidden_size
                                    })
        self.pool_ly = NLPPooling(**self.pooling_params)
        if self.span_pool:
            print(f"span pooler")
            self.spans_pooling_params = spans_pooling_params
            self.spans_pooling_params.update({"in_features":self.config.hidden_size,
                                          "out_features":self.config.hidden_size
                                    })
            self.spans_pool = NLPPooling(**self.spans_pooling_params)

        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        self._init_weights(self.fc)

    def _from_token_to_span(self,preds,labels_ids,attention_mask):
        TOK,SEQ = preds.shape
        predictions = []
        ids = torch.unique(labels_ids)
        for idx in ids:
            if idx!=-1:
                mask = labels_ids==idx
                p = preds[mask].reshape(1,-1,SEQ)
                att = attention_mask[mask].reshape(1,-1)
                predictions.append(self.spans_pool(p,att))
        return torch.cat(predictions)

    def from_token_to_span(self,preds,labels_ids,attention_mask):
        BS,_,SEQ = preds.shape
        if BS>1:
            print('Span pooler only support batch size = 1')
            
        predictions = []
        for p,l,att in zip(preds,labels_ids,attention_mask):
            predictions.append(self._from_token_to_span(p,l,att))
        
        return torch.cat(predictions).reshape(BS,-1,SEQ)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(3.0)

    def forward(self,b):

        # Apply Slidding window if # tokens > 512   
        if self.striding_wind:
            B,L=b["input_ids"].shape # BS x token_size
            if L<=self.window_size:
                x=self.backbone(input_ids=b["input_ids"],attention_mask=b["attention_mask"]).last_hidden_state
            else:
                # Slidding window
                segments=(L-self.window_size)//self.inner_len
                if (L-self.window_size)%self.inner_len>self.edge_len:
                    segments+=1
                elif segments==0:
                    segments+=1
                x=self.backbone(input_ids=b["input_ids"][:,:self.window_size],
                                attention_mask=b["attention_mask"][:,:self.window_size]).last_hidden_state
                    
                for i in range(1,segments+1):
                    start=self.window_size-self.edge_len+(i-1)*self.inner_len
                    end=self.window_size-self.edge_len+(i-1)*self.inner_len+self.window_size
                    end=min(end,L)
                    x_next=b["input_ids"][:,start:end]
                    mask_next=b["attention_mask"][:,start:end]
                    x_next=self.backbone(input_ids=x_next,attention_mask=mask_next).last_hidden_state
                    if i==segments:
                        x_next=x_next[:,self.edge_len:]
                    else:
                        x_next=x_next[:,self.edge_len:self.edge_len+self.inner_len]
                    x=torch.cat([x,x_next],1)
        else:
            x = self.backbone(input_ids=b["input_ids"],attention_mask=b["attention_mask"]).last_hidden_state
            
        x = self.dropout(x) # BS x token_size x Emb 
        
        

        ## Pooling on embeddings
        if self.span_pool:
            x = self.from_token_to_span(x,b['span_labels'],b['attention_mask'])
            b['attention_mask'] = b['attention_mask']*0 +1 
            b['attention_mask'] = b['attention_mask'][:,:x.shape[1]]

        x = self.pool_ly(x,b['attention_mask'])
        x = self.fc(x) # BS x num_classes 
        return x