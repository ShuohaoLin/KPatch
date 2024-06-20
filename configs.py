from string import Template

PLM_MAX_SEQ_LEN = 512
DATASET_PATH_TEMPLATE = Template("${root_dir}/data/${dataset}/${status}")

BERT_LIST = [
   "~/BERT/bert-tiny",  # 0
   "~/BERT/bert-mini",  # 1
   "~/BERT/bert-small",  # 2
   "~/BERT/bert-medium",  # 3
   "~/BERT/bert-base-cased",  # 4
   "~/BERT/bert-base-uncased",  # 5
]

BATCH_SIZE = {bert: [
   800, 560, 400, 300, 
   64, 64
][bi] for bi, bert in enumerate(BERT_LIST)}

URL_REGEX = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"