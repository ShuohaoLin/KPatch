# KPatch

This is the repository for the paper [KPatch: Knowledge Patch to Pre-trained Language Model for Zero-Shot Stance Detection on Social Media](https://aclanthology.org/2024.lrec-main.871/) (LREC-COLING 2024).

## Requirements
- torch 1.13.0
- transformers 4.26.1
- peft 0.3.0

## Start
### Dataset
1. All data should be divided into three sets: training, validation and testing, and stored in the `corpus.pkl` of the `train`, `valid` and `test` folders of the corresponding subsets.
   1. For example, if the target dataset is `SemEval2016TaskA` and the subset used as the test set is `HC` (Hillary Clinton), then the folder name is `SemEval2016TaskA-HC`.
   2. The data format in `corpus.pkl` is a list, and each piece of data is also a list, storing `tweets`, `topics` and `stance index` respectively.
   3. The `category2label.json` in the corresponding subset stores a list of stance labels, and the index of the last element of each data in `corpus.pkl` corresponds to it.
2. Please download MiDe22-EN Dataset from [this repository](https://github.com/avaapm/mide22), and then organize it according to the format of the Sem16 dataset.
### Knowledge
> The `data/SemEval2016TaskA_GPT_KG/KG_gen_text.pkl` in the repository contains triplet information obtained from `WikiData`, which may have information bias and is for academic research only.

As described in the "Knowledge Searching Stage", please collect the knowledge graph and record each triplet in the form of a Python tuple in the file `KG_gen_text.pkl` under the folder `data/<corresponding dataset>_GPT_KG`. Each triplet tuple includes the triplet text (such as `Barack Obama [SEP] president [SEP] USA`) and whether the triplet is a positive example (1 if yes, 0 otherwise).

### Train and Test
#### LoRA Implementation
```bash
python main.py --device 0 --PEFT_type LoRA --KG_rank 64 --dataset SemEval2016TaskA-HC --KG_name SemEval2016TaskA_GPT_KG --specific_output_reuslt_path KPatch-LoRA-64 --f1_score_type avg
```
#### P-Tuning V2 Implementation
```bash
python main.py --device 0 --PEFT_type P-Tuning\ v2 --num_virtual_tokens 64 --dataset SemEval2016TaskA-HC --KG_name SemEval2016TaskA_GPT_KG --specific_output_reuslt_path KPatch-PTuningV2-64 --f1_score_type avg
```
> Note: You need to modify `peft/peft_model.py` of the `peft` package to let `PTuning V2` return the embedding of `[CLS]`.
```python
def _prefix_tuning_forward(
  self,
  input_ids=None,
  attention_mask=None,
  inputs_embeds=None,
  labels=None,
  output_attentions=None,
  output_hidden_states=None,
  return_dict=None,
  **kwargs,
):
  batch_size = input_ids.shape[0]
  past_key_values = self.get_prompt(batch_size)
  fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
  kwargs.update(
    {
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "inputs_embeds": inputs_embeds,
      "output_attentions": output_attentions,
      "output_hidden_states": output_hidden_states,
      "return_dict": return_dict,
      "past_key_values": past_key_values,
    }
  )
  if "past_key_values" in fwd_params:
    return self.base_model(labels=labels, **kwargs)
  else:
    transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
    fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
    if "past_key_values" not in fwd_params:
      raise ValueError("Model does not support past key values which are required for prefix tuning.")
    outputs = transformer_backbone_name(**kwargs)
    pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
    if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
      pooled_output = self.base_model.dropout(pooled_output)
    return pooled_output  # << modified
```

## Citation
```
@inproceedings{lin-etal-2024-kpatch-knowledge,
    title = "{KP}atch: Knowledge Patch to Pre-trained Language Model for Zero-Shot Stance Detection on Social Media",
    author = "Lin, Shuohao  and
      Chen, Wei  and
      Gao, Yunpeng  and
      Jiang, Zhishu  and
      Liao, Mengqi  and
      Zhang, Zhiyu  and
      Zhao, Shuyuan  and
      Wan, Huaiyu",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.871",
    pages = "9961--9973",
}
```