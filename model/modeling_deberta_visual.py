from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.models.deberta.modeling_deberta import *
from transformers.models.bart.modeling_bart import (
    BartAttention,
    _make_causal_mask
)

from configuration_deberta_visual import DebertaWithVisualConfig

@dataclass
class WithVisualModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooled_visual_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class VisualCrossAttentionLayer(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.cross_attn = BartAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )

        self.cross_attn_LayerNorm = DebertaLayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, config.intermediate_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, self.hidden_size)
        self.final_LayerNorm = DebertaLayerNorm(self.hidden_size)

        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """Shapes of required inputs

            hidden_states: (batch, seqlen, hidden_size)
            visual_states: (batch, n_ctx_img_patches, hidden_size)
        """
        # cross attention
        residual = hidden_states

        visual_states = self.dropout(visual_states)

        # print (f"[in VisualCrossAttentionLayer.forward()] visual_states: {visual_states.size()}")
        
        hidden_states, attn_weights, _ = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=visual_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.cross_attn_LayerNorm(hidden_states + residual)

        # position-wise feedforward
        residual = hidden_states
        hidden_states = self.act_fn(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.final_LayerNorm(hidden_size + residual)

        if output_attentions:
            return (hidden_states, attn_weights)
        else:
            return hidden_states


class DebertaWithVisualEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.visual_LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.pooled_visual_LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.vlscore_LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # NOTE (Shih-Lun): below are added for visual inputs
        img_patches_per_side = config.img_patches_HW / (2 ** config.n_img_subsamp)

        if img_patches_per_side != int(img_patches_per_side):
            raise ValueError(f"Invalid n_img_subsamp -- {config.n_img_subsamp}")

        img_patches_per_side = int(img_patches_per_side)
        self.img_patch_position_embeddings = nn.ModuleDict({
            "H_emb": nn.Embedding(img_patches_per_side, config.hidden_size),
            "W_emb": nn.Embedding(img_patches_per_side, config.hidden_size)
        })

        self.img_id_embeddings = nn.Embedding(config.n_ctx_img, config.hidden_size)
        self.img_pred_id_embeddings = nn.Embedding(config.n_pred_img + 1, config.hidden_size)

        # shape for images -- (batch, n_ctx_img, H, W)
        self.register_buffer(
                "img_H_pos_ids", 
                torch.arange(img_patches_per_side).view((-1, 1)) \
                                                  .expand((1, 1, -1, img_patches_per_side))
            )
        self.register_buffer(
                "img_W_pos_ids", 
                torch.arange(img_patches_per_side).view((1, -1)) \
                                                  .expand((1, 1, img_patches_per_side, -1))
            )
        self.register_buffer(
                "img_ids", 
                torch.arange(config.n_ctx_img).view(-1, 1, 1) \
                                              .expand((1, -1, img_patches_per_side, img_patches_per_side))
            )

        self.visual_embed_proj = nn.Linear(config.visual_hidden_size, config.hidden_size)
        self.vlscore_proj = nn.Linear(config.n_ctx_img, config.hidden_size)

        n_img_subsamp = getattr(config, "n_img_subsamp", 0)
        if n_img_subsamp > 0:
            self.visual_subsamp_convs = nn.Sequential(*[
                nn.Conv2d(
                    config.visual_hidden_size,
                    config.visual_hidden_size,
                    kernel_size=3,
                    stride=2,
                    groups=16,
                    padding=1,
                ) for _ in range(n_img_subsamp)
            ])
        else:
            self.visual_subsamp_convs = None
        

    def forward(self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            mask=None,
            inputs_embeds=None,
            visual_inputs=None,
            img_pred_ids=None,
            vlscores=None,
        ):
        """Shapes for extra visual input tensors (Shih-Lun)

           -- visual_inputs: (batch, n_ctx_img, visual_hidden_size, H, W)
           -- img_pred_ids: (batch, n_ctx_img), e.g. (batch=2), [[0, 1, 0, 0, 2, 3], [1, 0, 2, 0, 0, 3]],
                            IMPORTANT !! -- the target img indices should be increasing ([*, 1, *, 2 *, 3, *]) 
           -- vlscores: (batch, seqlen, n_ctx_img)
        """

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batchsize = input_shape[0]
        seq_length = input_shape[1]
        n_ctx_img = visual_inputs.size(1)
        img_patches_per_side = self.img_patch_position_embeddings["W_emb"].num_embeddings

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)

        
        if self.visual_subsamp_convs is not None:
            visual_inputs = visual_inputs.view(-1, *visual_inputs.size()[2:])
            visual_inputs = self.visual_subsamp_convs(visual_inputs)
            visual_inputs = visual_inputs.view(batchsize, n_ctx_img, -1, img_patches_per_side, img_patches_per_side)

        assert visual_inputs.size(-1) == img_patches_per_side, "Something went wrong with visual feat subsampling"

        # put hidden size to last dimension
        visual_inputs = visual_inputs.permute(0, 1, 3, 4, 2)
        visual_inputs = self.visual_embed_proj(visual_inputs)

        visual_embeddings = visual_inputs + \
                            self.img_patch_position_embeddings["H_emb"](self.img_H_pos_ids) + \
                            self.img_patch_position_embeddings["W_emb"](self.img_W_pos_ids) + \
                            self.img_id_embeddings(self.img_ids)

        if img_pred_ids is not None:
            img_pred_ids = img_pred_ids.view(batchsize, n_ctx_img, 1, 1) \
                                       .expand(-1, -1, img_patches_per_side, img_patches_per_side)
            img_pred_ids_emb = self.img_pred_id_embeddings(img_pred_ids)
            visual_embeddings += img_pred_ids_emb

        pooled_visual_embeddings = visual_embeddings.mean(dim=(2, 3))

        visual_embeddings = self.visual_LayerNorm(visual_embeddings)
        pooled_visual_embeddings = self.pooled_visual_LayerNorm(pooled_visual_embeddings)

        # (batch, n_ctx_img, H, W, hidden_size) --> (batch, n_ctx_img * H * W, hidden_size)
        visual_embeddings = visual_embeddings.view(batchsize, -1, self.config.hidden_size)

        if vlscores is not None:
            vlscore_embeddings = self.vlscore_proj(vlscores)
        else:
            vlscore_embeddings = torch.zeros_like(embeddings)

        assert vlscore_embeddings.size() == embeddings.size()

        vlscore_embeddings = self.vlscore_LayerNorm(vlscore_embeddings)

        return {
            "text": embeddings,
            "visual": visual_embeddings,
            "pooled_visual": pooled_visual_embeddings,
            "vlscore": vlscore_embeddings,
        }

class DebertaWithVisualEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        # NOTE (Shih-Lun): settings for visual features
        assert config.num_visual_layers == len(config.visual_insert_layers)

        self.use_pooled_vis_feat = config.use_pooled_vis_feat
        self.use_patch_vis_feat = config.use_patch_vis_feat
        self.tie_visual_layers = config.tie_visual_layers
        self.visual_insert_layers = config.visual_insert_layers
        self.vlscore_insert_layers = set(config.vlscore_insert_layers)
        self.vlscore_dropout = StableDropout(config.hidden_dropout_prob)
        
        # NOTE (Shih-Lun): construct visual network parts
        if not self.tie_visual_layers:
            self.vis_layer = nn.ModuleList([
                VisualCrossAttentionLayer(config) for _ in range(config.num_visual_layers)
            ])
        else:
            _vis_layer = VisualCrossAttentionLayer(config)
            self.vis_layer = nn.ModuleList([_vis_layer] * config.num_visual_layers)

        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    def forward(
        self,
        hidden_states,
        visual_states,
        pooled_visual_states,
        vlscore_states,
        attention_mask,
        cross_attention_mask=None,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states

        rel_embeddings = self.get_rel_embedding()

        cur_vis_layer_idx = 0
        if self.use_pooled_vis_feat and self.use_patch_vis_feat:
            visual_states = torch.cat([pooled_visual_states, visual_states], dim=1)
        elif self.use_pooled_vis_feat and not self.use_patch_vis_feat:
            visual_states = pooled_visual_states
        elif not self.use_pooled_vis_feat and self.use_patch_vis_feat:
            visual_states = visual_states
        else:
            raise ValueError("at least one of use_pooled_vis_feat or use_patch_vis_feat should be true")

        # NOTE(Shih-Lun): fuse vlscore (e.g., clipscore) at the beginning
        if -1 in self.vlscore_insert_layers:
            next_kv += self.vlscore_dropout(vlscore_states)

        # NOTE(Shih-Lun): visual cross attention 
        # (visual_insert_layers[idx] < 0) -- fuse before all Deberta layers
        while cur_vis_layer_idx < len(self.vis_layer) and \
              self.visual_insert_layers[ cur_vis_layer_idx ] < 0:
            # print ("visual added after before zeroeth layer")

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = self.vis_layer[ cur_vis_layer_idx ](
                hidden_states=hidden_states,
                visual_states=visual_states,
                attention_mask=cross_attention_mask,
                output_attentions=output_attentions,
            )
            
            if output_attentions:
                hidden_states, att_m = hidden_states
                all_attentions = all_attentions + (att_m,)

            next_kv = hidden_states
            cur_vis_layer_idx += 1


        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                hidden_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

            # NOTE(Shih-Lun): fuse vlscore (e.g., clipscore) in the middle
            if i in self.vlscore_insert_layers:
                print ("vlscore added after layer #", i)
                next_kv += self.vlscore_dropout(vlscore_states)

            # NOTE(Shih-Lun) visual cross attention
            # (visual_insert_layers[idx] >= 0) -- fuse after some deberta layers
            while cur_vis_layer_idx < len(self.vis_layer) and \
                  self.visual_insert_layers[ cur_vis_layer_idx ] == i:
                # print ("visual added after layer #", i)

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                hidden_states = self.vis_layer[ cur_vis_layer_idx ](
                    hidden_states=next_kv,
                    visual_states=visual_states,
                    attention_mask=cross_attention_mask,
                    output_attentions=output_attentions,
                )
                
                if output_attentions:
                    hidden_states, att_m = hidden_states
                    all_attentions = all_attentions + (att_m,)

                next_kv = hidden_states
                cur_vis_layer_idx += 1

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            assert len(all_hidden_states) == len(self.vis_layer) + len(self.layer) + 1

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )

class DebertaWithVisualModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaWithVisualEmbeddings(config)
        self.encoder = DebertaWithVisualEncoder(config)
        self.do_causal_self_attn = getattr(config, "do_causal_self_attn", False)
        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def load_hf_pretrained_deberta(self, deberta_model_name):
        _hf_model = DebertaModel.from_pretrained(deberta_model_name)

        _load_out = self.load_state_dict(
                        _hf_model.state_dict(),
                        strict=False
                    )

        print (f"[in load_hf_pretrained_deberta()] pretrained model: {deberta_model_name} loaded", )
        print (f"[in load_hf_pretrained_deberta()] weights not in pretrained model: {_load_out.missing_keys}")
        print (f"[in load_hf_pretrained_deberta()] weights not in new model: {_load_out.unexpected_keys}")


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        visual_inputs: Optional[torch.Tensor] = None,
        img_pred_ids: Optional[torch.Tensor] = None,
        vlscores: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if visual_inputs is None:
            raise ValueError("You have to specify visual_inputs")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            if not self.do_causal_self_attn:
                attention_mask = torch.ones(input_shape, device=device)
            else:
                attention_mask = _make_causal_mask(input_shape, visual_inputs.dtype, 0) \
                                    .to(device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=None,
            inputs_embeds=inputs_embeds,
            visual_inputs=visual_inputs,
            img_pred_ids=img_pred_ids,
            vlscores=vlscores,
        )

        encoder_outputs = self.encoder(
            embedding_output["text"],
            embedding_output["visual"],
            embedding_output["pooled_visual"],
            embedding_output["vlscore"],
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return WithVisualModelOutput(
            last_hidden_state=sequence_output,
            pooled_visual_state=embedding_output["pooled_visual"],
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )

class DebertaForPhotobookListener(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 3)
        self.intermediate_size = getattr(config, "clf_head_hidden_size", 256)
        self.n_pred_img = config.n_pred_img
        self.act_fn = ACT2FN[config.hidden_act]

        self.deberta = DebertaWithVisualModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = nn.Linear(config.hidden_size * 2, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        visual_inputs: Optional[torch.Tensor] = None,
        vlscores: Optional[torch.Tensor] = None,
        img_pred_ids: Optional[torch.Tensor] = None,
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
        labels (`torch.LongTensor` of shape `(batch_size, seqlen, n_pred_img)`):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
            ignored (masked) labels should be set to -100 (pytorch default for CrossEntropyLoss())

            Shapes for extra input tensors (by Shih-Lun)

            -- token_type_ids: (batch, seqlen), used to distinguish a token is from self (idx 0) or partner (idx 1)
            -- visual_inputs:  (batch, n_ctx_img, visual_hidden_size, H, W)
            -- img_pred_ids:   (batch, n_ctx_img), e.g. (batch=2), [[0, 1, 0, 0, 2, 3], [1, 0, 2, 0, 0, 3]],
                                IMPORTANT !! -- the target img indices should be increasing ([*, 1, *, 2 *, 3, *]) 
            -- vlscores:       (batch, seqlen, n_ctx_img)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            visual_inputs=visual_inputs,
            vlscores=vlscores,
            img_pred_ids=img_pred_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # NOTE (Shih-Lun): make shapes of "sequence_output" & "target_img_pooled_state" to be 
        #                  (batch, seqlen, n_pred_img, hidden_size)
        sequence_output = outputs.last_hidden_state.unsqueeze(-2).expand(-1, -1, self.n_pred_img, -1)
        pooled_visual_state = outputs.pooled_visual_state
        
        # select pooled features of target imgs only
        target_img_pooled_state = pooled_visual_state[ 
                                        img_pred_ids.nonzero(as_tuple=True)
                                    ].view(sequence_output.size(0), 1, self.n_pred_img, -1) \
                                     .expand(-1, sequence_output.size(1), -1, -1)


        # three target images share the same MLP classifier
        clf_input = torch.cat([sequence_output, target_img_pooled_state], dim=-1)
        clf_input = self.dropout(clf_input)

        logits = self.act_fn(self.fc1(clf_input))
        logits = self.dropout(logits)
        logits = self.fc2(logits)

        print (f"[in Listener.forward()] clf_input: {clf_input.size()}, logits: {logits.size()}, labels: {labels.size()}")

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


# NOTE (Shih-Lun): below are unit tests
if __name__ == "__main__":
    import sys
    config_json = sys.argv[1] # e.g., "config/deberta_visual_base.json"
    hf_model_name = "microsoft/deberta-base"

    # test 01: huggingface model loading
    model = DebertaWithVisualModel(
                DebertaWithVisualConfig.from_json_file(
                    config_json
                )
            )    
    model.load_hf_pretrained_deberta(hf_model_name)
    print ("[test 01] load pretrained weights: Success !!")


    # test 02: embeddings module
    emb_module = DebertaWithVisualEmbeddings(
                    DebertaWithVisualConfig.from_json_file(
                        config_json
                    )
                )
    bs, seqlen, hidden_size = 4, 100, 768
    n_ctx_img, patchsize = 6, 16
    input_ids = torch.randint(0, 10, (bs, seqlen))
    token_type_ids = torch.randint(0, 1, (bs, seqlen)) # used to distinguish a token is from self (id = 0) or partner (id = 1)
    img_pred_ids = torch.LongTensor([
                        [0, 1, 0, 0, 2, 3], 
                        [1, 0, 2, 0, 0, 3],
                        [0, 0, 0, 1, 2, 3],
                        [1, 2, 3, 0, 0, 0],
                    ])
    visual_inputs = torch.randn(bs, n_ctx_img, hidden_size, patchsize, patchsize)
    vlscores = torch.randn(bs, seqlen, n_ctx_img)

    emb_outs = emb_module(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    visual_inputs=visual_inputs,
                    img_pred_ids=img_pred_ids,
                    vlscores=vlscores
                )

    print ("[test 02] embedding module:")
    for key in emb_outs:
        print (f"  {key:16}: {emb_outs[key].size()}")
    print ("")

    # test 03: transformer encoder module
    enc_outs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        visual_inputs=visual_inputs,
        img_pred_ids=img_pred_ids,
        vlscores=vlscores,
    )
    print ("[test 03] transformer encoder module:")
    print (f"  last_hidden: {enc_outs.last_hidden_state.size()}\n")

    # test 04: full photobook listener model
    model = DebertaForPhotobookListener(
                DebertaWithVisualConfig.from_json_file(
                    config_json
                )
            )
    model.deberta.load_hf_pretrained_deberta(hf_model_name)

    n_pred_img = 3
    labels = torch.randint(0, 3, (bs, seqlen, n_pred_img))
    model_outs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        visual_inputs=visual_inputs,
        img_pred_ids=img_pred_ids,
        vlscores=vlscores,
        labels=labels,
    )
    print ("[test 04] full PhotoBook listener model:")
    print (f"  logits: {model_outs.logits.size()}\n")
