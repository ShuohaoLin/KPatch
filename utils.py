import codecs, json, os
import re
from typing import List
from tqdm import tqdm

import random

import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

from configs import URL_REGEX, PLM_MAX_SEQ_LEN

def operate_json(file_path: str, content=None):
    operate_mode = 'r' if content is None else 'w'
    with codecs.open(file_path, operate_mode, 'utf-8') as f:
        if operate_mode == 'w':
            json.dump(content, f, ensure_ascii=False, indent=4)
        else:
            return json.load(f)

def param_init(module, init_func=torch.nn.init.kaiming_uniform_):
   for name, param in module.named_parameters():
      if "weight" in name:
         if len(param.shape) == 2:
            init_func(param)
      else:
         torch.nn.init.zeros_(param)

def check_text_is_useless(text):
    return text == "" or len(text.split(" ")) < 4 if text.isascii() else len(text) < 11

def clean_text(text):
    return "\n".join([
        re.sub(URL_REGEX, "", paragraph).strip()
        for paragraph in text.split("\n")
        if not check_text_is_useless(paragraph.strip())
    ]) if isinstance(text, str) else ""

def root_path_transfer(path:str):
    return os.path.expanduser(path)

def tokenize_func(params, texts:List[str], max_length:int=PLM_MAX_SEQ_LEN):
    return params.tokenizer(
        texts, 
        padding=True, 
        return_tensors="pt", 
        truncation=True,
        max_length=max_length
    )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_metrics(
        params,
        groud_true:torch.Tensor, pred_labels:torch.Tensor, 
        return_float:bool=False
    ):
    if params.f1_score_type == "avg":
        _, _, f1scores, _ = prfs(
            groud_true.cpu().numpy(), pred_labels.cpu().numpy(), labels=[0, 1],
            average=None,
            zero_division=0
        )
        f1score = sum(f1scores) / 2
        pro_f1, con_f1 = f1scores
    else:
        _, _, f1score, _ = prfs(
            groud_true.cpu().numpy(), pred_labels.cpu().numpy(), 
            average=params.f1_score_type, 
            zero_division=0
        )
        pro_f1, con_f1 = 0, 0
    if return_float:
        return [_ for _ in [pro_f1, con_f1, f1score]]
    else:
        return {
            "pro_f1": f"{pro_f1 * 100:.4}%",
            "con_f1": f"{con_f1 * 100:.4}%",
            "f1score": f"{f1score * 100:.4}%",
        }
    
def validate(model:nn.Module, dataloader, params):
    model.eval()
    bar = tqdm(dataloader)
    all_preds, all_gts = [], []
    for pak in bar:
        input_tensors = pak[:-1]
        labels = pak[-1]
        with torch.no_grad():
            pred = model(*input_tensors)
            if isinstance(pred, tuple):
                pred = pred[0]
        
        if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels)
        labels:torch.Tensor
        
        all_gts.append(labels.type(torch.LongTensor).to(pred.device))
        all_preds.append(torch.argmax(pred, dim=-1))
    return cal_metrics(
        params,
        torch.cat(all_gts, dim=0), torch.cat(all_preds, dim=0), 
        True
    )