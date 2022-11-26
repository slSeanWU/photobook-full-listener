from transformers.configuration_utils import PretrainedConfig

class DebertaWithVisualConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        img_patches_HW=16,
        n_img_subsamp=1,
        n_ctx_img=6,
        n_pred_img=3,
        num_visual_layers=2,
        visual_hidden_size=768,
        visual_insert_layers=[9, 9],
        vlscore_insert_layers=[9, 10],
        tie_visual_layers=True,
        use_pooled_vis_feat=True,
        use_patch_vis_feat=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act

        # NOTE (Shih-Lun): Configs below are added to bring visual feats in
        self.img_patches_HW = img_patches_HW
        self.n_img_subsamp = n_img_subsamp
        self.n_ctx_img = n_ctx_img
        self.n_pred_img = n_pred_img
        self.num_visual_layers = num_visual_layers
        self.visual_hidden_size = visual_hidden_size
        self.visual_insert_layers = visual_insert_layers
        self.vlscore_insert_layers = vlscore_insert_layers

        self.tie_visual_layers = tie_visual_layers
        self.use_pooled_vis_feat = use_pooled_vis_feat
        self.use_patch_vis_feat = use_patch_vis_feat