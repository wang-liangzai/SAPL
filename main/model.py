import math
import copy
import numpy
import torch
import timm
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from collections import Counter
from pretrained_model.PyTorch_Pretrained_ViT.pytorch_pretrained_vit.model import ViT
from GeneratePrompt import SoftPromptInitializer
import torch.nn as nn
from TRAR.trar import DynRT_ED
from TRAR.cls_layer import cls_layer_both, cls_layer_img, cls_layer_txt
import pickle
import tqdm
import os
from torch.utils.data import DataLoader
n_tokens = 10
roberta_initialize_from_vocab = True

# 使用示例，用来高斯分布来初始化软提示


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertPooler_i2t(nn.Module):
    def __init__(self):
        super(BertPooler_i2t, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPooler(nn.Module):
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]   
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        del s, u
        return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCoAttention(nn.Module):
    def __init__(self):
        super(BertCoAttention, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        # s2_attention_mask  b*1*1*49
        mixed_query_layer = self.query(s1_hidden_states)  # b*75*768
        mixed_key_layer = self.key(s2_hidden_states)  # b*49*768
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b*12*75*64
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b*12*49*64
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b*12*49*64

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # b*12*75*49
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # b*12*75*49
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask
        # attention_scores b*12*75*49
        # Normalize the attention scores to probabilities.
        # b*12*75*49
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # aa = attention_probs.cpu().numpy()
        # aa = numpy.sum(aa,axis=1,keepdims=True)
        # aa = numpy.sum(aa,axis=2,keepdims=True)
        # aa = numpy.linalg.norm(aa, ord=None, axis=-2, keepdims=True)
        # with open('/data/qiaoyang/ms_data/aa.pkl', 'wb') as f:
        #     pickle.dump(aa, f)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)      # (32,12,197,64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()      # (32,197,12,64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)     # (32,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer b*75*768
        del attention_scores, attention_probs
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttention(nn.Module):
    def __init__(self):
        super(BertCrossAttention, self).__init__()
        self.bertCoAttn = BertCoAttention()
        self.output = BertSelfOutput()

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self):
        super(BertCrossAttentionLayer, self).__init__()
        self.bertCorssAttn = BertCrossAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.bertCorssAttn(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        # b*75*768
        intermediate_output = self.intermediate(attention_output)
        # b*75*3072
        layer_output = self.output(intermediate_output, attention_output)
        # b*75*3072
        del intermediate_output, attention_output
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(3)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        return s1_hidden_states

# robert = RobertaModel.from_pretrained("/home/mmn/wgj/DynRT-main/DynRT/roberta-base")
# tokenizers = RobertaTokenizer.from_pretrained("/home/mmn/wgj/DynRT-main/DynRT/roberta-base")
class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                    n_tokens,
                                                                                    random_range,
                                                                                    initialize_from_vocab=True))
        self.learned_embedding_1 = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab=True))
        self.learned_embedding_2 = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab=False))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        else:
            # return softprompt
            return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)  # 这里是随机初始化
        # 初始化是用的一句话


    def forward(self, bert_indices_add):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # 原始文本
        if self.training:

            input_embedding = self.wte(bert_indices_add[:int(bert_indices_add.shape[0]/3), :]).cuda()

            # 使用语言模型的初始化方式，初始化一个字面意思的文本
            input_embedding_1 = self.wte(bert_indices_add[int(bert_indices_add.shape[0]/3):2 * int(bert_indices_add.shape[0]/3), :-(self.n_tokens)]).cuda()   # 1*26 十个提示向量，剩下的是文本向量

            learned_embedding_1 = self.learned_embedding_1.repeat(input_embedding_1.size(0), 1, 1).cuda()  # 重复第0个维度n次，其它为1的就是不重复
            prompt_embedding_1 = torch.cat([learned_embedding_1, input_embedding_1], 1)
            # 使用随机初始化的方式，初始化一个反意的表达
            input_embedding_2 = self.wte(bert_indices_add[2 * int(bert_indices_add.shape[0]/3):3 * int(bert_indices_add.shape[0]/3), :-(self.n_tokens)]).cuda()

            learned_embedding_2 = self.learned_embedding_2.repeat(input_embedding_2.size(0), 1, 1).cuda()
            prompt_embedding_2 = torch.cat([learned_embedding_2, input_embedding_2], 1)

            return torch.cat([input_embedding, prompt_embedding_1, prompt_embedding_2], dim=0)
        else:
            input_embedding = self.wte(bert_indices_add[:, :]).cuda()
            return input_embedding

def OpenJson(fname):
    with open(fname, "r") as f:
            return json.load(f)

class DMSD(nn.Module):
    def __init__(self, feature_dim=128):
        super(DMSD, self).__init__()
        self.roberta = RobertaModel.from_pretrained('../pretrained_model/roberta-base', return_dict=True)
        self.text_image_attention = BertCrossEncoder()
        self.dropout = nn.Dropout(0.1)
        self.cls_pooler = BertPooler()
        self.classifier = nn.Linear(768, 2)
        self.vit = ViT('B_16', pretrained=True)
        self.DynRT_vit = timm.create_model("vit_base_patch32_224", pretrained=True)

        self.projection = nn.Sequential(nn.Linear(768, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, feature_dim))
        self.mmd_linear = nn.Sequential(nn.Linear(768, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, feature_dim))


        self.roberta_text_linear = torch.nn.Linear(768, 512).to('cuda:0')
        self.classifier_text = torch.nn.Linear(512, 2).to('cuda:0')
        self.classifier_image = torch.nn.Linear(768, 2).to('cuda:0')
        ''''''
        fname = './config/DynRT.json'
        json = OpenJson(fname)
        self.opt = json["opt"]
        self.trar = DynRT(self.opt)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.opt['modelopt']["output_size"], 2)
        )
        self.sigm = torch.nn.Sigmoid()
        prompt_emb = SoftEmbedding(self.roberta.get_input_embeddings(),
                                   n_tokens=n_tokens,
                                   initialize_from_vocab=roberta_initialize_from_vocab)
        self.roberta.set_input_embeddings(prompt_emb)

    def vit_forward(self, x):
        x = self.DynRT_vit.patch_embed(x)  # pixel_embed ->patch_embed
        cls_token = self.DynRT_vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        batch = x.shape[0]
        x = torch.cat((cls_token[:batch, :, :], x), dim=1)
        # x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.DynRT_vit.pos_drop(x + self.DynRT_vit.pos_embed)
        x = self.DynRT_vit.blocks(x)
        x = self.DynRT_vit.norm(x)
        return x[:, 1:]


    def forward(self, input_mask, bert_mask_add, bert_indices_add, img_trans, mode="train"):
        outputs = self.roberta(input_ids=bert_indices_add, attention_mask=bert_mask_add, output_attentions=True)
        pooler_output = outputs.last_hidden_state   # pooler_output(batchsize,77,768)
        mean, visual = self.vit(img_trans)      # viasual(batchsoze,197,768)
        # 利用额外的vit进行编码
        # DynRT_image_feature = self.vit_forward(img_trans)

        text_mask = input_mask.to(dtype=torch.float32)  # text_mask(batchsize,77) # 如果输入动态路由，则需要进行维度扩展
        text_mask = (1.0 - text_mask) * -10000.0
        text_extended_mask = text_mask.unsqueeze(1).unsqueeze(2)   # (batchsize,1,1,77)
        # 利用动态路由机制
        # DynRT_text_mask = input_mask.to(dtype=torch.bool)
        # DynRT_extended_mask = DynRT_text_mask.unsqueeze(1).unsqueeze(2)

        cross_attn = self.text_image_attention(visual, pooler_output, text_extended_mask)  #(batchsize,197,768)
        image_extended_mask = torch.zeros([visual.shape[0], 1, 1, visual.shape[1]], device=visual.device)
        cross_attn_1 = self.text_image_attention(pooler_output, visual, image_extended_mask)  #(batchsize,197,768)
        # 做Poole
        cls = self.cls_pooler(cross_attn)  #(batchsize,768)，只取cls
        cls = cls + self.cls_pooler(cross_attn_1)
        pooled_output = (self.dropout(cls))
        ''''''#新增的动态路由的内容
        lang_feat = torch.mean(pooler_output, dim=1)
        img_feat = torch.mean(visual, dim=1)
        text_feature = self.roberta_text_linear(lang_feat)
        text_feature = self.classifier_text(text_feature)
        image_feature = self.classifier_image(img_feat)
        text_score = nn.functional.softmax(text_feature, dim=-1)
        image_score = nn.functional.softmax(image_feature, dim=-1)



        if mode == "train":
            output = self.classifier(pooled_output)
            output = nn.functional.softmax(output, dim=-1)
            ''''''# 新增加的内容
            # out = self.classifier(out1)
            # result = self.sigm(out)
            # result = result + text_score + image_score
            feature = F.normalize(self.projection(cls), dim=1)
            mmd_feature = F.normalize(self.mmd_linear(cls))
            mmd_feature = F.softmax(mmd_feature, dim=-1)
            del pooled_output

            # (out1, lang_emb, img_emb) = self.trar(DynRT_image_feature[:int(DynRT_image_feature.size(0)/3),:,:], pooler_output[:int(DynRT_image_feature.size(0)/3),:,:], DynRT_extended_mask[:int(DynRT_image_feature.size(0)/3),:,:,:])

            # out = self.classifier(out1)
            # result = self.sigm(out)

            return output + text_score + image_score, feature, mmd_feature
            # return output, feature, mmd_feature
        else:
            output = self.classifier(pooled_output)
            output = nn.functional.softmax(output, dim=-1)
            ''''''# 新增加的内容
            # out = self.classifier(out1)
            # result = self.sigm(out)
            # result = result + text_score + image_score

            # (out1, lang_emb, img_emb) = self.trar(DynRT_image_feature,
            #                                       pooler_output,
            #                                       DynRT_extended_mask)
            #
            # out = self.classifier(out1)
            # result = self.sigm(out)

            del pooled_output
            return output + text_score + image_score


class DynRT(nn.Module):
    def __init__(self, opt):
        super(DynRT, self).__init__()
        assert ("input1" in opt['modelopt'])
        assert ("input2" in opt['modelopt'])
        assert ("input3" in opt['modelopt'])
        self.input1 = opt['modelopt']["input1"]
        self.input2 = opt['modelopt']["input2"]
        self.input3 = opt['modelopt']["input3"]


        self.backbone = DynRT_ED(opt)
        self.att = torch.nn.Linear(512, 1, bias=False)
        self.classifier_fuse = torch.nn.Linear(512, 2)
        self.text_projection = torch.nn.Linear(768, 512)
        self.visual_projection = torch.nn.Linear(768, 512)
        if opt['modelopt']["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt['modelopt']["hidden_size"], opt['modelopt']["output_size"])
        elif opt['modelopt']["classifier"] == 'text':
            self.cls_layer = cls_layer_txt(opt['modelopt']["hidden_size"], opt['modelopt']["output_size"])
        elif opt['modelopt']["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt['modelopt']["hidden_size"], opt['modelopt']["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0], 1, 1, img_feat.shape[1]], dtype=torch.bool,
                                    device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = torch.mean(lang_feat, dim=1)
        img_feat = torch.mean(img_feat, dim=1)

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat

if __name__ == '__main__':
    data = torch.rand(2, 128)
    data2 = torch.rand(2, 128)
    print(torch.cat([data, data2], dim=1).shape)
    