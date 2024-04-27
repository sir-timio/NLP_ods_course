from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
)

from .crf import CRF


class DebertaV2WithCRF(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        # do NOT use it with CRF or LSTM!
        self.post_init()
        
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # remove [CLS] token for CRF working properly
        logits = logits[:, 1:]
        attention_mask = attention_mask[:, 1:].type(torch.bool)
        labels = labels[:, 1:]
        is_pad = labels == -100
        labels = torch.where(is_pad, torch.tensor(12, device=labels.device, dtype=labels.dtype), labels)
        # labels.masked_fill_(is_pad, 12) # -> fail backward
        
        loss = None
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask, reduction="token_mean")
            tags = self.crf.decode(logits, mask=attention_mask)
        else:
            tags = self.crf.decode(logits, mask=attention_mask)
        
        # list[list[int]] -> padded tensor
        _, seq_length = attention_mask.shape
        padded_tags = [tag + [-100] * (seq_length - len(tag)) for tag in tags]
        tags = torch.tensor(padded_tags, dtype=torch.long, device=logits.device)
        
        # labels.masked_fill_(is_pad, -100)
        labels = torch.where(is_pad, torch.tensor(-100, device=labels.device, dtype=labels.dtype), labels)
        if not return_dict:
            output = (tags,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=tags, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )