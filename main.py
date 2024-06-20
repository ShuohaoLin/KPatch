import argparse, os
from time import time
from random import random

from configs import *
from utils import setup_seed, operate_json, root_path_transfer

from transformers import BertTokenizerFast

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--dataset', type=str, default="SemEval2016TaskA-HC")
parser.add_argument('--KG_name', nargs='+', type=str, choices=["SemEval2016TaskA_GPT_KG", "MiDe2022_GPT_KG"])
parser.add_argument('--kg_train_data_path', type=str)

parser.add_argument('--cuda', nargs='+', type=int, default=[0,1,2,3])
parser.add_argument("--multi_gpu", type=int, default=0, choices=[0, 1])
parser.add_argument("--local_rank", type=int)
parser.add_argument("--local-rank", type=int)
parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
parser.add_argument('--kg_checkpoint_path', type=str)
parser.add_argument('--sd_checkpoint_path', type=str)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=1)

parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--valid_interval', type=int, default=1)
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--PEFT_type', type=str, default="LoRA", choices=["LoRA", "P-Tuning v2"])
parser.add_argument('--KG_rank', type=int, default=128)
parser.add_argument('--num_virtual_tokens', type=int, default=128)
parser.add_argument('--KG_epoch', type=int, default=2)

parser.add_argument('--bert_path', type=str, default=BERT_LIST[5], choices=BERT_LIST)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--training', type=int, default=1)
parser.add_argument('--f1_score_type', type=str, default="avg", choices=["avg", "micro", "macro", "weighted"])

parser.add_argument('--checkpoint_id', default=None)
parser.add_argument('--same_time', type=int, default=1)
parser.add_argument('--output_reuslt_path', type=str, default="./results/")
parser.add_argument('--specific_output_reuslt_path', type=str, default="Ours")
parser.add_argument('--output_file_name', type=str, default=None)

params = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in params.cuda])
os.environ['WORLD_SIZE'] = str(len(params.cuda))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

setup_seed(params.random_seed)

encoder_model_name = params.bert_path.replace("/", "@").replace("~", "")

checkpoint_id = int(time()) if params.training == 1 else params.checkpoint_id
if params.same_time == 1:
    checkpoint_id = f"{checkpoint_id}_{round(random(), 6)}"

checkpoint_dir = os.path.join(params.checkpoint_path, params.dataset)
if params.training == 1 and not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
params.sd_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoints_SD_{checkpoint_id}")
if not os.path.exists(params.sd_checkpoint_path) and params.local_rank in [0, None]:
    os.makedirs(params.sd_checkpoint_path, exist_ok=True)

params.sd_classifier_checkpoint_path = os.path.join(params.sd_checkpoint_path, f"sd_classifier.pt")

if params.output_file_name is None:
    dst = params.dataset.split("-")[-1]
    params.output_file_name = f"{dst}-{params.random_seed}"

tn = str(int(time()))
if params.specific_output_reuslt_path == None:
    params.specific_output_reuslt_path = tn

params.output_reuslt_path = os.path.join(params.output_reuslt_path, params.dataset, params.specific_output_reuslt_path)
if not os.path.exists(params.output_reuslt_path) and params.local_rank in [0, None]:
    os.makedirs(params.output_reuslt_path)

if params.batch_size == -1:
    params.batch_size = BATCH_SIZE[params.bert_path]

if os.path.isdir(os.path.expanduser(params.bert_path)): 
    params.bert_path = os.path.expanduser(params.bert_path)
params.tokenizer = BertTokenizerFast.from_pretrained(params.bert_path)

params.stances = [_.lower() for _ in operate_json(os.path.join(f"./data/{params.dataset}/category2label.json"))]
params.stance_num = len(params.stances)
params.multi_gpu = params.multi_gpu == 1
params.full_KG_name = "#".join(sorted(params.KG_name))
params.kg_train_data_paths = {
    n: f"./data/{n}/KG_gen_text.pkl"
    for n in params.KG_name
}
params.main_dataset = params.dataset.split("-")[0]
if params.PEFT_type == "LoRA":
    path_by_PEFT = f"KG_rank{params.KG_rank}_{params.main_dataset}_{params.full_KG_name}_{encoder_model_name}"
elif params.PEFT_type == "P-Tuning v2":
    path_by_PEFT = f"PTuningV2_VTokenNum{params.num_virtual_tokens}_{params.main_dataset}_{params.full_KG_name}_{encoder_model_name}"
else:
    raise NotImplementedError

params.kg_checkpoint_path = os.path.join("KPatches", path_by_PEFT)

params.bert_path = root_path_transfer(params.bert_path)

if params.training == 1:
    params.KG_training = False
    
    if "tuning" in params.PEFT_type.lower(): 
        params.batch_size *= 0.8
        params.batch_size = int(params.batch_size)

    if os.path.exists(params.kg_checkpoint_path):
        print("KG has been compressed and will not be compressed again...")
    else:
        from train_utils import train_kg
        print("Start compressing KG...")
        params.KG_training = True
        
        ori_epoch = params.epoch
        ori_batch_size = params.batch_size
        ori_f1_type = params.f1_score_type
        ori_lr = params.lr
        params.epoch = params.KG_epoch
        if "tuning" in params.PEFT_type.lower():
            params.lr = 1e-3

        train_kg(params)
        params.KG_training = False
        params.epoch = ori_epoch
        params.batch_size = ori_batch_size
        params.lr = ori_lr

    from train_utils import train, test
    if "MiDe2022" in params.main_dataset and params.training == 1:
        params.batch_size = int(params.batch_size * 0.15)
        if params.batch_size % 2 != 0:
            params.batch_size += 1
    print("Start training on the target task...")
    train(params)
    params.training = 0
    if params.local_rank in [0, None]:
        test(params)
else:
    from train_utils import test
    test(params)