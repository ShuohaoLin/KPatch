from typing import List, Optional, Union, Tuple

from utils import param_init, tokenize_func

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType, PeftType
from peft import PeftModel

class CustomBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls_layer_name = ""  # for P-Tuning V2

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        return {
            "CLS_emb": pooled_output,
            "token_embs": outputs[0]
        }

class KPatch(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        
        if params.PEFT_type == "LoRA":
            peft_config = LoraConfig(
                inference_mode=False, 
                r=params.KG_rank, 
                lora_alpha=32, 
                lora_dropout=0.1
            )
        elif params.PEFT_type == "P-Tuning v2":
            peft_config = PrefixTuningConfig(
                peft_type=PeftType.P_TUNING,
                task_type=TaskType.SEQ_CLS,
                num_virtual_tokens=params.num_virtual_tokens,
                prefix_projection=False,
                inference_mode=False
            )
        else:
            raise NotImplementedError

        self.kg_PEFT = get_peft_model(CustomBert.from_pretrained(params.bert_path), peft_config)

        if params.PEFT_type == "LoRA":
            hidden_size = self.kg_PEFT.base_model.model.bert.config.hidden_size
        else:
            hidden_size = self.kg_PEFT.base_model.bert.config.hidden_size
        
        target_dim = 2
        self.classifier = nn.Linear(hidden_size, target_dim)
        param_init(self.classifier)
        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, texts:List[str]):
        text_tokenized = tokenize_func(self.params, texts, max_length=125)
        
        if self.params.PEFT_type == "LoRA":
            device = self.kg_PEFT.base_model.model.bert.embeddings.word_embeddings.weight.device
        else:
            device = self.kg_PEFT.base_model.bert.embeddings.word_embeddings.weight.device

        res = self.kg_PEFT(**{
            k: v.to(device) 
            for k, v in text_tokenized.items() 
        })
        return self.classifier(res["CLS_emb"] if self.params.PEFT_type == "LoRA" else res)

class SD(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

        model = CustomBert.from_pretrained(params.bert_path)
        self.model = PeftModel.from_pretrained(model, params.kg_checkpoint_path, is_trainable=False)
        for pname, parameter in self.model.named_parameters():
            if any([_ in pname for _ in ["lora", "prompt"]]): 
                parameter.requires_grad = False
                continue
            parameter.requires_grad = True
        self.classifier = nn.Linear(model.bert.config.hidden_size, params.stance_num)
        param_init(self.classifier)

    def forward(self, texts:List[str], get_bert_last_hidden:bool=False):
        text_tokenized = tokenize_func(self.params, texts)
        device = self.classifier.weight.device
        res = self.model(**{
            k: v.to(device) 
            for k, v in text_tokenized.items() 
        })
        _out = res['CLS_emb'] if self.params.PEFT_type == "LoRA" else res
        r = [self.classifier(_out)]
        if get_bert_last_hidden:
            r += [F.normalize(_out.unsqueeze(1), dim=-1)]
            return r
        return r[0]
