import pickle, os, re
from typing import List
from string import punctuation

from configs import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])
emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
CHINESE_PUNCTUATION = "'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'"
ALL_PUNCTUATION = CHINESE_PUNCTUATION + punctuation

def collate_fn(items):
    sgs, labels = list(zip(*items))
    return sgs, torch.stack(labels, dim=0)

class CustomDataset(Dataset):
    def __init__(self, data:List, local_rank:int, process_func=None):
        self.data = data
        self.len = len(data)
        self.local_rank = local_rank
        self.process_func = process_func
    
    def __getitem__(self, index):
        return self.data[index] if self.process_func is None else self.process_func(self.data[index])

    def __len__(self):
        return self.len

def read_data(dataset:str=None, status:str=None, split2items:bool=True, root_dir:str=".", data_file_name:str="corpus.pkl", **kwargs):
    assert status in ["train", "valid", "test"], "status is invalid..."
    dir_path = DATASET_PATH_TEMPLATE.substitute(root_dir=root_dir, dataset=dataset, status=status)
    path = os.path.join(dir_path, data_file_name)
    
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        if not split2items:
            data = list(zip(*data))
    return data, dir_path

def preprocess_tweet(tweet):
    tweet = re.sub(URL_REGEX, "", tweet).replace("#SemST", "").strip().replace("&amp", "").encode("ascii", "ignore").decode()

    if tweet[:2] == "RT":
        tweet = tweet[2:]
    
    tweet_toks = tweet.split(" ")
    final_tweet_toks = []
    for i in range(len(tweet_toks)):
        if tweet_toks[i].startswith("#"):
            hashtag = tweet_toks[i]
            hashtag = hashtag[1:]
            split_hashtag = re.findall('[0-9]+|[A-Z][a-z]+|[A-Z][A-Z]+|[a-z]+', hashtag)
            final_tweet_toks = final_tweet_toks + split_hashtag
        else:
            final_tweet_toks.append(tweet_toks[i])
    tweet = " ".join(final_tweet_toks)
    return tweet

def get_sd_dataset(
        params, 
        batch_size:int,
        local_rank:int,
        status:str="train",
        multi_gpu:bool=False,
        **kwargs
    ):
    assert status in ["train", "valid", "test"], f"{status} is invalid..."
    _corpus_list, _ = read_data(dataset=params.dataset, status=status, **kwargs)

    corpus_list = []
    for tweet, target, label in tqdm(_corpus_list):
        tweet = preprocess_tweet(tweet)
        processed_text = f"{tweet} {params.tokenizer.sep_token} {target}"
        corpus_list.append([processed_text, label])

    corpus_list = list(zip(*corpus_list))
    corpus_list = list(zip(*[
        corpus_list[0], 
        torch.tensor(corpus_list[1], dtype=torch.float)
    ]))

    dataset = CustomDataset(corpus_list, local_rank, kwargs.get("process_func", None))

    if multi_gpu:
        sampler = DistributedSampler(dataset)
        return DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=sampler, 
            collate_fn=kwargs.get("collate_fn", collate_fn)
        ), sampler
    else:
        return DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=kwargs.get("collate_fn", collate_fn)
        ), None

def read_KG_corpus(
        params, 
        batch_size:int,
        local_rank:int,
        multi_gpu:bool=False,
        root_path:str="."
    ):
    
    _corpus_list = []
    for _, kg_path in params.kg_train_data_paths.items():
        with open(kg_path, "rb") as handle:
            C = pickle.load(handle)
            _corpus_list.extend(C)

    corpus_list = []
    for triplet_seq, label in tqdm(_corpus_list):
        corpus_list.append([triplet_seq, [1 if label == i else 0 for i in range(2)]])

    corpus_list = list(zip(*corpus_list))
    corpus_list = list(zip(*[
        corpus_list[0], 
        torch.tensor(corpus_list[1], dtype=torch.long)
    ]))

    dataset = CustomDataset(corpus_list, local_rank)
    if multi_gpu:
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn), sampler
    else:
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), None
