from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertModel


class ListenerModelBertAttCtxHist(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob):
        super().__init__()
        self.embedding_dim = embedding_dim  # BERT
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.attention_dim = att_dim

        # project images to hidden dimensions
        self.linear_separate = nn.Linear(self.img_dim, self.hidden_dim)

        # from BERT dimensions to hidden dimensions, as we receive embeddings from BERT
        self.lin_emb2hid = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.lin_emb2HIST = nn.Linear(self.embedding_dim, self.hidden_dim)  # here we also have history

        # highlighted image projected to hidden
        self.lin_img = nn.Linear(self.img_dim, self.hidden_dim)

        # Multimodal (bert representation; visual context)
        self.lin_mm = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # attention linear layers
        self.att_linear_1 = nn.Linear(self.hidden_dim, self.attention_dim)
        self.att_linear_2 = nn.Linear(self.attention_dim, 1)

        # BERT text embedding
        _bertmodel = BertModel.from_pretrained("bert-base-uncased")
        self.bert_emb = deepcopy(_bertmodel.embeddings)
        del _bertmodel

        # Output classification MLP
        self.out_fc1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.out_actfn = nn.GELU()
        self.out_fc2 = nn.Linear(self.hidden_dim, 2) # ["common", "different"], no dense learning signal

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()  # initialize layers

    def init_weights(self):

        for ll in [self.linear_separate, self.lin_emb2hid, self.lin_emb2HIST,
                   self.lin_img, self.lin_mm,
                   self.att_linear_1, self.att_linear_2,
                   self.out_fc1, self.out_fc2]:

            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

        for p in self.bert_emb.parameters():
            p.requires_grad = False


    def forward(
            self, separate_images, img_pred,
            prev_hist, prev_hist_len, masks, 
            representations=None, input_text=None,
        ):
        """
        @param separate_images: image feature vectors for all 6 images in the context separately
        @param img_pred: the highlighted image to predict common/different
        @param prev_hist: contains padded histories for 6 images separately (if exists for a given image)
        @param prev_hist_len: contains length of histories for 6 images separately (if exists for a given image)
        @param masks: attention mask for pad tokens
        @param representations: utterance converted into BERT representations
        @param input_text: dialogue text (per round) in the form BERT token ids
        """
        assert input_text is not None or representations is not None

        if input_text is not None:
            assert representations is None, "got input_text, representations should be None"
            batch_size = input_text.shape[0]
            representations = self.bert_emb(
                                input_ids=input_text
                            )
        else:
            batch_size = representations.shape[0]  # effective batch size
            # utterance representations are processed
            representations = self.dropout(representations)

        device = input_text.device if input_text is not None else representations.device

        input_reps = self.relu(self.lin_emb2hid(representations))   # (batch, seqlen, hidden)

        # highlighted img is processed
        img_pred = self.dropout(img_pred)
        projected_img = self.relu(self.lin_img(img_pred))           # (batch, hidden)
        repeated_img = projected_img.unsqueeze(1).repeat(1, input_reps.shape[1], 1)     # (batch, seqlen, hidden)

        # multimodal utterance representations
        mm_reps = self.relu(self.lin_mm(torch.cat((input_reps, repeated_img), dim=2)))

        # attention over the multimodal utterance representations (tokens and visual context interact)
        outputs_att = self.att_linear_2(self.tanh(self.att_linear_1(mm_reps)))

        # mask pads so that no attention is paid to them (with -inf)
        masks = masks.unsqueeze(2)      # (batch, max_seqlen, 1) 
        outputs_att = outputs_att.masked_fill_(masks, float('-inf'))

        # final attention weights
        att_weights = self.softmax(outputs_att)

        # encoder context representation
        attended_hids = (mm_reps * att_weights).sum(dim=1)

        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)

        # this is where we add history to candidates

        for b in range(batch_size):

            batch_prev_hist = prev_hist[b]
            batch_prev_hist_len = prev_hist_len[b]

            for s in range(len(batch_prev_hist)):

                if batch_prev_hist_len[s] > 0:
                    # if there is history for a candidate image
                    if input_text is not None:
                        ref_chain = batch_prev_hist[s][:batch_prev_hist_len[s]]
                        ref_chain = ref_chain.unsqueeze(0)
                        hist_rep = self.bert_emb(input_ids=ref_chain) # (1, seqlen, 768) 
                        hist_rep = hist_rep.squeeze(0)
                    else:
                        hist_rep = torch.stack(batch_prev_hist[s]).to(device)

                    # take the average history vector
                    hist_avg = self.dropout(hist_rep.sum(dim=0)/hist_rep.shape[0])

                    # process the history representation and add to image representations
                    separate_images[b][s] += self.relu(self.lin_emb2HIST(hist_avg))

        # some candidates are now multimodal with the addition of history
        separate_images = self.relu(separate_images)
        separate_images = F.normalize(separate_images, p=2, dim=2)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance
        # dot = torch.bmm(separate_images, attended_hids.view(batch_size, self.hidden_dim,1))

        # return dot

        # average-pool key encoder 6-image context
        separate_images = separate_images.mean(dim=1)

        # output classifier
        out = self.out_fc1(torch.cat([separate_images, attended_hids], dim=1))
        out = self.out_actfn(out)
        out = self.dropout(out)
        out = self.out_fc2(out)

        return out


if __name__ == "__main__":
    embed_dim = 768
    hidden_dim = 384
    img_dim = 512
    att_dim = 1024
    
    model = ListenerModelBertAttCtxHist(
        embedding_dim=embed_dim,
        hidden_dim=hidden_dim,
        img_dim=img_dim,
        att_dim=att_dim,
        dropout_prob=0.1
    )

    bsize, seqlen = 16, 50
    n_img = 6

    representations = torch.randn(bsize, seqlen, embed_dim)
    img_pred = torch.randn(bsize, img_dim)
    input_text = torch.randint(0, 1000, (bsize, seqlen))
    separate_images = torch.randn(bsize, n_img, img_dim)
    visual_context = torch.randn(bsize, n_img * img_dim)
    one_prev_hist = [[1,2,3,0,0],[0]*5,[4,5,0,0,0],[6,7,8,9,10],[0]*5, [0]*5]
    one_prev_hist_len = [3,0,2,5,0,0]
    prev_hist = [ one_prev_hist[:] for _ in range(bsize)]
    prev_hist_len = [one_prev_hist_len for _ in range(bsize)]
    #prev_hist = [ [[] for _ in range(n_img)] for _ in range(bsize)]
    masks = torch.randint(0, 2, (bsize, seqlen, 1))
    masks = masks == 1
    device = "cpu"

    model_out = model(
        None,
        separate_images,
        img_pred,
        prev_hist,
        prev_hist_len,
        masks,
        device,
        input_text=input_text
    )
    print (model_out.size())
