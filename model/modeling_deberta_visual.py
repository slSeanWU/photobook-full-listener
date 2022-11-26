import torch.nn as nn
from transformers.models.deberta.modeling_deberta import *

from configuration_deberta_visual import DebertaWithVisualConfig

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

        if config.visual_hidden_size != config.hidden_size:
            self.visual_embed_proj = nn.Linear(config.visual_hidden_size, config.hidden_size, bias=False)
        else:
            self.visual_embed_proj = None

        self.vlscore_proj = nn.Linear(config.n_ctx_img, config.hidden_size, bias=False)

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
           -- img_pred_ids: (batch, n_ctx_img), e.g. (batch=2), [[0, 1, 0, 0, 2, 3], [1, 0, 2, 0, 0, 3]]
           -- vlscores: (batch, n_ctx_img)
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

        if self.visual_embed_proj is not None:
            visual_inputs = self.visual_embed_proj(visual_inputs)
        if self.visual_subsamp_convs is not None:
            visual_inputs = visual_inputs.view(-1, *visual_inputs.size()[2:])
            visual_inputs = self.visual_subsamp_convs(visual_inputs)
            visual_inputs = visual_inputs.view(batchsize, n_ctx_img, -1, img_patches_per_side, img_patches_per_side)

        assert visual_inputs.size(-1) == img_patches_per_side, "Something went wrong with visual feat subsampling"

        # put hidden size to last dimension
        visual_inputs = visual_inputs.permute(0, 1, 3, 4, 2)

        visual_embeddings = visual_inputs + \
                            self.img_patch_position_embeddings["H_emb"](self.img_H_pos_ids) + \
                            self.img_patch_position_embeddings["W_emb"](self.img_W_pos_ids) + \
                            self.img_id_embeddings(self.img_ids)

        if img_pred_ids is not None:
            img_pred_ids = img_pred_ids.view(batchsize, n_ctx_img, 1, 1) \
                                       .expand(-1, -1, img_patches_per_side, img_patches_per_side)
            img_pred_ids_emb = self.img_pred_id_embeddings(img_pred_ids)
            visual_embeddings += img_pred_ids_emb

        visual_embeddings = self.dropout(visual_embeddings)
        pooled_visual_embeddings = visual_embeddings.mean(dim=(2, 3))

        # (batch, n_ctx_img, H, W, hidden_size) --> (batch, n_ctx_img * H * W, hidden_size)
        visual_embeddings = visual_embeddings.view(batchsize, -1, self.config.hidden_size)

        if vlscores is not None:
            vlscore_embeddings = self.vlscore_proj(vlscores)
        else:
            vlscore_embeddings = torch.zeros_like(embeddings)

        assert vlscore_embeddings.size() == embeddings.size()

        vlscore_embeddings = self.dropout(vlscore_embeddings)

        return {
            "text": embeddings,
            "visual": visual_embeddings,
            "pooled_visual": pooled_visual_embeddings,
            "vlscore": vlscore_embeddings,
        }

class DebertaWithVisualModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaWithVisualEmbeddings(config)
        self.encoder = DebertaEncoder(config)
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

        self.load_state_dict(
            _hf_model.state_dict(),
            strict=False
        )

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
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
            visual_inputs=visual_inputs,
            img_pred_ids=img_pred_ids,
            vlscores=vlscores,
        )

        encoder_outputs = self.encoder(
            embedding_output,
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

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )

if __name__ == "__main__":
    # NOTE (Shih-Lun): unit tests
    import sys
    config_json = sys.argv[1]
    hf_model_name = "microsoft/deberta-base"

    # test 01: huggingface model loading
    model = DebertaWithVisualModel(
                DebertaWithVisualConfig.from_json_file(
                    config_json
                )
            )    
    model.load_hf_pretrained_deberta(hf_model_name)


    # test 02: embeddings module
    emb_module = DebertaWithVisualEmbeddings(
                    DebertaWithVisualConfig.from_json_file(
                        config_json
                    )
                )
    bs, seqlen, hidden_size = 4, 100, 768
    n_ctx_img, patchsize = 6, 16
    input_ids = torch.randint(0, 10, (bs, seqlen))
    img_pred_ids = torch.randint(0, 4, (bs, n_ctx_img))
    visual_inputs = torch.randn(bs, n_ctx_img, hidden_size, patchsize, patchsize)
    vlscores = torch.randn(bs, seqlen, n_ctx_img)

    emb_outs = emb_module(
                    input_ids=input_ids,
                    visual_inputs=visual_inputs,
                    img_pred_ids=img_pred_ids,
                    vlscores=vlscores
                )

    for key in emb_outs:
        print (f"{key:16}: {emb_outs[key].size()}")