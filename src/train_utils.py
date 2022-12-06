import re
import os
import math
import time
import json
import random
import numpy as np
import pandas as pd

from pathlib import Path

import torch 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader
from data.data_utils import batch_to_device
from data.dataset import FeedbackDataset,CustomCollator
from transformers import AutoTokenizer,AutoConfig

from model_zoo.models import FeedbackModel
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup


from tqdm.auto import tqdm
import gc
import torch.utils.checkpoint
from metrics_loss.loss_function import RMSELoss,mcrmse,comp_metric,FeedbackLoss




# ------------------------------------------ seeds ------------------------------------------- #
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ------------------------------------------  ------------------------------------------- #
# ------------------------------------------  ------------------------------------------- #

class AutoSave:
  def __init__(self, top_k=3,metric_track="mae_val",mode="min", root=None):
    
    self.top_k = top_k
    self.logs = []
    self.metric = metric_track
    self.mode = -1 if mode=='min' else 1
    self.root = Path(root)
    assert self.root.exists()

    self.top_models = []
    self.top_metrics = []
    self.texte_log = []

  def log(self, model, metrics):
    metric = metrics[self.metric]
    rank = self.rank(self.mode*metric)

    self.top_metrics.insert(rank+1, self.mode*metric)
    if len(self.top_metrics) > self.top_k:
      self.top_metrics.pop(0)


    self.logs.append(metrics)
    self.save(model, rank, metrics)


  def save(self, model,rank, metrics):
    val_text = " "
    for k,v in metrics.items():
        if k in ["fold","epoch",'step','train_loss','val_loss']:
            if k in ["fold","epoch",'step']:
                val_text+=f"_{k}={v:.0f} "
            else:
                val_text+=f"_{k}={v:.4f} "

    name = val_text.strip()
    name = name+".pth"
    name = name.replace('=',"_")
    path = self.root.joinpath(name)

    old_model = None
    self.top_models.insert(rank+1, name)
    if len(self.top_models) > self.top_k:
      old_model = self.root.joinpath(self.top_models[0])
      self.top_models.pop(0)      

    torch.save(model.state_dict(), path.as_posix())

    if old_model is not None:
      old_model.unlink()


  def rank(self, val):
    r = -1
    for top_val in self.top_metrics:
      if val <= top_val:
        return r
      r += 1

    return r


# # ----------------- Opt/Sched --------------------- #
def get_optim_sched(model,args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.optimizer["params"]['weight_decay'],
        "lr": args.optimizer["params"]['lr'],
                                    },
                                    {
                                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                        "weight_decay": 0.0,
                                        "lr": args.optimizer["params"]['lr'],
                                    }]

    if args.optimizer['name']=="optim.AdamW":
        optimizer = eval(args.optimizer['name'])(optimizer_grouped_parameters,lr=args.optimizer["params"]['lr'])
    else:
        optimizer = eval(args.optimizer['name'])(model.parameters(), **args.optimizer['params'])

    # if 'scheduler' in args:
    if args.scheduler['name'] == 'poly':

        params = args.scheduler['params']

        power = params['power']
        lr_end = params['lr_end']

        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))

        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)

    elif args.scheduler['name'] in ['linear','cosine']:
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        if args.scheduler['name']=="linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, training_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
            
    elif args.scheduler['name'] in ['optim.lr_scheduler.OneCycleLR']:
        max_lr = args.optimizer['params']['lr']
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        scheduler = eval(args.scheduler['name'])(optimizer,max_lr=max_lr,
                                                 epochs=args.trainer['epochs'],
                                                 steps_per_epoch=training_steps,
                                                 pct_start = args.scheduler['warmup']
                                                 )

    return optimizer,scheduler

# # ----------------- One Step --------------------- #
def training_step(args,model,criterion,data):
    model.train()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data = batch_to_device(data, device)

    if args.trainer['use_amp']:
        with amp.autocast():
            pred = model(data)
    else:
        pred = model(data)

    loss = criterion(pred,data['label'][:,0,:])
    return loss,{"train_loss":loss.item()}

# ------------------------------------------ ------------------------------------------- #
def evaluate_step(args,model,criterion,val_loader):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    criterion = eval(args.model['loss'])(reduction="none").to(args.device)

    model.eval()
    ypred = []
    ytrue = []
    loss = []
    with torch.no_grad():
        for data in val_loader:
            data = batch_to_device(data, device)
            pred = model(data)
            ytrue.append(data['label'][:,0,:])
            ypred.append(pred)

    
    ytrue = torch.cat(ytrue,dim=0).detach().cpu()#.numpy() 
    ypred = torch.cat(ypred,dim=0).detach().cpu()#.numpy() 

    m,c = comp_metric(ytrue,ypred) 
    met = {"val_loss":m}   
    cols = args.model['target']
    for i,col in enumerate(cols):
        met[col] = c[i]

    return met
# ------------------------------------------ ------------------------------------------- #
# #----------------------------------- Training Steps -------------------------------------------------#

def fit_net(
                model,
                train_dataset,
                val_dataset,
                args,
                fold,
                tokenizer,
    ):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    
    data_collator = CustomCollator(tokenizer,num_target=args.model["num_labels"])
    criterion_tr = eval(args.model['loss'])(loss_name = args.model['sub_loss'],
                        loss_param = args.model['sub_loss_param'],
                        reduction=args.model['loss_reduction'],
                        weights = args.model['target_weights'],
                        device = device
                        ).to(device)


    train_loader = DataLoader(train_dataset,**args.train_loader,collate_fn=data_collator)
    val_loader = DataLoader(val_dataset,**args.val_loader,collate_fn=data_collator)
    
    args.len_train_loader = len(train_loader)
    args.dataset_size = len(train_dataset)

    mode_ = -1 if args.callbacks["mode"]=='max' else 1
    best_epoch = mode_*np.inf
    best = mode_*np.inf

    es = args.callbacks['es']
    es_step = 0
    patience = args.callbacks['patience']

    if args.callbacks["save"]:
        Path(args.checkpoints_path).mkdir(parents=True,exist_ok=True)

    saver = AutoSave(root=args.checkpoints_path,metric_track=args.callbacks['metric_track'],top_k=args.callbacks['top_k'],mode=args.callbacks['mode'])

  
    if args.trainer['use_amp'] and ("cuda" in str(device)):
        scaler = amp.GradScaler()
        print("Using Amp")
    else:
        scaler = None

    optimizer,scheduler = get_optim_sched(model,args)


    for epoch in range(args.trainer['epochs']):
        # Init
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        # Init Metrics
        trn_metric = {}
        for k in ["train_loss"]:
            trn_metric[k]=0
        

        nb_step_per_epoch = args.len_train_loader
        step_val = int(np.round(nb_step_per_epoch*args.callbacks['epoch_pct_eval']))
        nstep_val = int(1/args.callbacks['epoch_pct_eval'])
        if args.callbacks['epoch_eval_dist']=="uniforme":
            evaluation_steps = [(nb_step_per_epoch//2)+x for x in np.arange(0,nb_step_per_epoch//2,nb_step_per_epoch//(2*nstep_val))][1:]
        else:
            evaluation_steps = [x for x in np.arange(nb_step_per_epoch) if (x + 1) % step_val == 0][1:]

        trn_loss = []
        pbar = tqdm(train_loader)
        for step,data in enumerate(pbar):
            if step==epoch and step==0:
                print('\n')
                print(" ".join(train_dataset.tokenizer.convert_ids_to_tokens(data['input_ids'][0])))
                print('\n')
            loss,tr_sc= training_step(args,model,criterion_tr,data)
            pbar.set_postfix(tr_sc)
            trn_loss.append(tr_sc['train_loss'])
            trn_metric["train_loss"] = np.mean(trn_loss)
 

            if args.trainer['use_amp']:
                scaler.scale(loss).backward()
                 # gradient clipping
                if args.trainer['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                                                        parameters=model.parameters(), max_norm=args.trainer['max_norm']
                                                    )

                scaler.step(optimizer)
                scaler.update()
                

            else:
                loss.backward()
                # gradient clipping
                if args.trainer['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                                                    parameters=model.parameters(), max_norm=args.trainer['max_norm']
                                                )
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            # Evaluation
            if (((step + 1) in evaluation_steps) or (step + 1 == nb_step_per_epoch)) and (epoch>=args.callbacks["start_eval_epoch"]):
                metric_val = evaluate_step(args,model,criterion_tr,val_loader)

                if es:
                    if args.callbacks['mode']=='min':
                        if (metric_val[args.callbacks['metric_track']]<best):
                            best = metric_val[args.callbacks['metric_track']]
                    else:
                        if (metric_val>best):
                            best = metric_val[args.callbacks['metric_track']]

                metrics = {
                    "fold":fold,
                    "epoch": epoch+1,
                    "step": int(step),
                    "global_step":step+(epoch*nb_step_per_epoch),
                    "best":best
                    
                }
                metrics.update(metric_val)
                metrics.update(trn_metric)
                saver.log(model, metrics)
        
                elapsed_time = time.time() - start_time
                elapsed_time = elapsed_time * args.callbacks['verbose_eval']

                lr = scheduler.get_lr()[0]
                
                val_text = " "
                for k,v in metric_val.items():
                    val_text+=f" {k}={v:.4f} "

                trn_text = " "
                for k,v in trn_metric.items():
                    trn_text+=f" {k}={v:.4f} "

                metrics.update({"lr":lr})

                texte = f"Epoch {epoch + 1}.{int(np.ceil((step+1)/step_val))}/{args.trainer['epochs']} lr={lr:.6f} t={elapsed_time:.0f}s "
                texte = texte+trn_text+val_text
                print(texte)
                metric_val = metric_val[args.callbacks['metric_track']] 


        if es:
            if args.callbacks['mode']=='min':
                if (best<best_epoch):
                    best_epoch = best
                    es_step = 0
                else:
                    es_step+=1
                    print(f"es step {es_step}")
            else:
                if (best>best_epoch):
                    best_epoch = best
                    es_step = 0
                else:
                    es_step+=1
                    print(f"es step {es_step}")

            if (es_step>patience):
                break

    torch.cuda.empty_cache()
    

# #----------------------------------- Training Folds -------------------------------------------------#
def train_one_fold(args,tokenizer,train_df,valid_df,fold):    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


    train_dataset = FeedbackDataset(train_df,
                                    tokenizer,
                                    args.model["max_len"],
                                    target=args.model['target'],
                                    spans=args.model['spans'],
                                    )

    val_dataset = FeedbackDataset(valid_df,
                                    tokenizer,
                                    args.model["max_len_eval"],
                                    target=args.model['target'],
                                    spans=args.model['spans'],
                                    )
    
    
        
    model = FeedbackModel(args.model['model_name'],
                        args.model['num_labels'],
                        config_path = args.model['pretrained_config'],
                        use_gradient_checkpointing = args.trainer['use_gradient_checkpointing'] ,
                        pooling_params=args.model['pooling_params'],
                        spans_pooling_params=args.model['spans_pooling_params'],
                        window_size = args.model["max_len"] if "deberta" in args.model['model_name'] else 512,
                        spans=args.model['spans'],
                        striding_wind = False if "deberta" in args.model['model_name'] else True,
                        ).to(device) 
        
    model.zero_grad()    

    pred_val = fit_net(
        model,
        train_dataset,
        val_dataset,
        args,
        fold,
        tokenizer,
    )
    return pred_val


def kfold(args,df):
    

    k = len((df[df[args.kfold_name]!=-1][args.kfold_name].unique()))
    tokenizer = AutoTokenizer.from_pretrained(args.model['model_name'])
    tokenizer.save_pretrained(Path(args.checkpoints_path)/'tokenizer/')
    config = AutoConfig.from_pretrained(args.model['model_name'])
    torch.save(config, Path(args.checkpoints_path)/'config.pth')

    print(f"----------- {args.kfold_name} ---------")
    for i in args.selected_folds:
        
       
        print(f"\n-------------   Fold {i+1} / {k}  -------------\n")
        if args.trainer['sample']:
            train_df = df[df[args.kfold_name]!=i].reset_index(drop=True).sample(100)
            valid_df = df[df[args.kfold_name]==i].reset_index(drop=True).sample(100)

        else:
            train_df = df[~df[args.kfold_name].isin([-1,i])].reset_index(drop=True)#.sample(100)
            valid_df = df[df[args.kfold_name]==i].reset_index(drop=True)#.sample(100)

        config = {"model":args.model}
        
        config.update({"optimizer":args.optimizer})
        config.update({'scheduler':args.scheduler})
        config.update({"train_loader":args.train_loader})
        config.update({"val_loader":args.val_loader})
        config.update({"trainer":args.trainer})
        config.update({"callbacks":args.callbacks})
        
        with open(args.checkpoints_path+'/params.json', 'w') as f:
            json.dump(config, f)

        train_one_fold(args,tokenizer,train_df,valid_df,i)




# ------------------------------------------ ------------------------------------------- #
def prediction_step(args,df_test,checkpoints,weights=None):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model['pretrained_tokenizer'])
    data_collator = CustomCollator(tokenizer,num_target=args.model['num_labels'],inference=True)

    test_dataset = FeedbackDataset(df_test,
                                    tokenizer,
                                    args.model["max_len_eval"],
                                    target=args.model['target'],
                                    spans=args.model['spans'],
                                    )
                                    
    test_loader = DataLoader(test_dataset,**args.val_loader,collate_fn=data_collator)

    if not weights:
        weights = [1/len(checkpoints)]*len(checkpoints)

    ypred = 0
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    for j,(checkpoint,weight) in enumerate(zip(checkpoints,weights)):
        net = FeedbackModel(args.model['model_name'],
                            args.model['num_labels'],
                            config_path = args.model['pretrained_config'],
                            use_gradient_checkpointing = args.trainer['use_gradient_checkpointing'] ,
                            pooling_params=args.model['pooling_params'],
                            spans_pooling_params=args.model['spans_pooling_params'],
                            window_size = args.model["max_len_eval"] if "deberta" in args.model['model_name'] else 512,
                            spans=args.model['spans'],
                            striding_wind = False if "deberta" in args.model['model_name'] else True,
                            )
        

        net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
        net = net.to(device)
        net.eval()


        ypred_model = []
        with torch.no_grad():
            for data in test_loader:
                data = batch_to_device(data, device)
                pred = net(data)*weight
                ypred_model.append(pred)

        ypred+= torch.cat(ypred_model,dim=0).detach().cpu().numpy() 

        del net
        del ypred_model
        del data
        del pred
        torch.cuda.empty_cache()
        gc.collect()
    
    dc = {"text_id":test_dataset.text_id}
    for i,name in enumerate(args.model['target']):
        dc.update({name:ypred[:,i]})

    sub = pd.DataFrame(dc)

    del test_dataset
    del test_loader
    del data_collator
    del tokenizer
    del dc

    gc.collect()
    torch.cuda.empty_cache()
    return sub