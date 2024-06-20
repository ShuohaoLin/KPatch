from time import time
from tqdm import tqdm

from modules import SD
from data import get_sd_dataset, read_KG_corpus
from utils import operate_json, cal_metrics, validate

import torch
from torch import nn
import torch.distributed as dist

from modules import KPatch

def valid(
    params,
    epoch_or_step:int,
    local_rank:int,
    model,
    MULTI_GPU,
    valid_dataloader,
    stop_flag,
    best_metric,
    patience_times
):
    def save_model():
        torch.save(
            model.module.state_dict() if MULTI_GPU else model.state_dict(), 
            params.sd_classifier_checkpoint_path
        )

    if epoch_or_step % params.valid_interval == 0 and local_rank in [0, None]:
        early_stop_metric = validate(model.module if MULTI_GPU else model, valid_dataloader, params)

        early_stop_f1 = early_stop_metric[-1]

        msg = "；".join([
            f"new F1：{early_stop_f1:.4f}",
            f"F1_pro：{early_stop_metric[0]:3f}",
            f"F1_con：{early_stop_metric[1]:3f}",
            f"previous best F1：{best_metric:.4f}",
        ])
        print(msg, end="；")

        if early_stop_f1 > best_metric:
            patience_times = 0
            best_metric = early_stop_f1
            save_model()
            msg = "Better! Save checkpoint..."
            print(msg)
        else:
            patience_times += 1
            msg = f"{patience_times} ({params.patience})"
            print(msg)
            if patience_times >= params.patience:
                print(f"Stop training.")
                stop_flag += 1
    if MULTI_GPU:
        dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
    if stop_flag == 1:
        print(f"Stop! Process {local_rank}...")
        return best_metric, patience_times, True
    return best_metric, patience_times, False

def train(params):
    MULTI_GPU = params.multi_gpu
    if MULTI_GPU:
        local_rank = params.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group(backend="nccl")
        except:
            pass
        model = SD(params).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        local_rank = 0
        device = torch.device(f"cuda:{params.device}") if params.device > -1 else torch.device("cpu")
        model = SD(params).to(device)

    train_dataloader, train_data_sampler = get_sd_dataset(params, params.batch_size, local_rank, multi_gpu=MULTI_GPU)
    valid_dataloader, _ = get_sd_dataset(params, params.batch_size, local_rank, "valid", multi_gpu=False)

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    best_metric = 0
    patience_times = 0
    training_step_count = 0
    s = time()
    for epoch in range(params.epoch):
        if MULTI_GPU:
            dist.barrier()
        model.train()
        if MULTI_GPU:
            train_data_sampler.set_epoch(epoch)
        total_loss, step_count = 0, 0

        stop_flag = torch.zeros(1).to(device)
        bar = tqdm(train_dataloader)
        for texts, labels in bar:
            optimizer.zero_grad()
            pred, feature = model(texts, get_bert_last_hidden=True)
            _L = labels.type(torch.LongTensor).to(pred.device)
            loss = loss_func(pred, _L)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            step_count += 1
            training_step_count += 1
            
            bar.desc = f"Rank: {local_rank}, Epoch: {epoch:03d}, Step: {training_step_count}；loss: {round(total_loss / step_count, 4)}"
        
        best_metric, patience_times, early_stop = valid(
            params, epoch, local_rank,
            model, MULTI_GPU,
            valid_dataloader,
            stop_flag,
            best_metric, patience_times
        )
        if early_stop:
            break
        
    params.train_SD_time_cost = f"{round(time() - s)}s"

def test(params):
    device = torch.device(f"cuda:{params.device}") if params.device > -1 else torch.device("cpu")

    model = SD(params).to(device)
    model.load_state_dict(torch.load(params.sd_classifier_checkpoint_path, map_location=device))
    model.eval()

    test_dataloader, _ = get_sd_dataset(params, params.batch_size, params.local_rank, "test", multi_gpu=False)
    
    bar = tqdm(test_dataloader)
    all_tweets, all_preds, all_gts = [], [], []
    s = time()
    for tweets, labels in bar:
        with torch.no_grad():
            pred = model(tweets)
        if isinstance(pred, tuple):
            pred = pred[0]
        
        if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels)
        labels:torch.Tensor
        all_gts.append(labels.type(torch.LongTensor).to(pred.device))
        all_preds.append(torch.argmax(pred, dim=1))
        all_tweets.extend(tweets)
        
    gts = torch.cat(all_gts, dim=0)
    preds = torch.cat(all_preds, dim=0)
    r = cal_metrics(params, gts, preds, True)

    msg = f"F1: {r[-1]:3f}; F1_pro: {r[0]:3f}; F1_con: {r[1]:3f}"
    print(msg)

    error_tweets = [{
        "tweet": all_tweets[i],
        "ground true": params.stances[gts[i]],
        "predition": params.stances[preds[i]],
    } for i in torch.where(gts.cpu() != preds.cpu())[0].cpu().tolist()]

    if "metric_result" not in params:
        params.metric_result = r
    params.tokenizer = None
    params.test_time_cost = f"{round(time() - s)}s"
    params.error_preds = {
        "len": len(error_tweets),
        "tweets": error_tweets
    }
    params.metric = r[-1]
    print(f"Finish! {params.output_reuslt_path}")
    operate_json(f"{params.output_reuslt_path}/{params.output_file_name}.json", vars(params))

def train_kg(params):
    MULTI_GPU = params.multi_gpu
    if MULTI_GPU:
        local_rank = params.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        local_rank = 0

    if MULTI_GPU:
        model = KPatch(params).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        device = torch.device(f"cuda:{params.device}") if params.device > -1 else torch.device("cpu")
        model = KPatch(params).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    train_dataloader, train_data_sampler = read_KG_corpus(
        params, params.batch_size, local_rank, multi_gpu=MULTI_GPU
    )
    loss_func = nn.CrossEntropyLoss()

    s = time()
    for epoch in range(params.epoch):
        if MULTI_GPU:
            dist.barrier()
        model.train()
        if MULTI_GPU:
            train_data_sampler.set_epoch(epoch)
        total_loss, count = 0, 0

        bar = tqdm(train_dataloader)
        for texts, labels in bar:
            optimizer.zero_grad()
            pred = model(texts)
            loss = loss_func(pred, torch.argmax(labels, dim=1).to(pred.device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
            bar.desc = f"Rank: {local_rank}, Epoch: {epoch:03d}, loss: {round(total_loss / count, 4)}"

        if (epoch % 2 == 0 or params.epoch == epoch + 1) and local_rank in [0, None]:
            _m = model.module if MULTI_GPU else model
            _m.kg_PEFT.save_pretrained(params.kg_checkpoint_path)
            
    params.train_KG_time_cost = f"{round(time() - s, 2)}s"
    params.KG_train_epoch = epoch
