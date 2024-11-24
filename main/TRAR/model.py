import torch.nn as nn
import torch
from TRAR.trar import DynRT_ED
from TRAR.cls_layer import cls_layer_both, cls_layer_img, cls_layer_txt
from transformers import RobertaConfig
from transformers.models.bert.modeling_bert import BertLayer
import copy

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)  # 只返回最后一层的隐藏状态
        return all_encoder_layers, all_encoder_attentions

class DynRT(nn.Module):
    def __init__(self, opt):
        super(DynRT, self).__init__()
        assert ("input1" in opt)
        assert ("input2" in opt)
        assert ("input3" in opt)
        self.input1 = opt["input1"]
        self.input2 = opt["input2"]
        self.input3 = opt["input3"]


        self.backbone = DynRT_ED(opt)

        self.config = RobertaConfig.from_pretrained("/home/mmn/wgj/DynRT-main/DynRT/roberta-base")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.att = torch.nn.Linear(512, 1, bias=False)
        self.classifier_fuse = torch.nn.Linear(512, 2)
        self.text_projection = torch.nn.Linear(768, 512)
        self.visual_projection = torch.nn.Linear(768, 512)
        self.trans = MultimodalEncoder(self.config, layer_number=1)
        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"], opt["output_size"])
        elif opt["classifier"] == 'text':
            self.cls_layer = cls_layer_txt(opt["hidden_size"], opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"], opt["output_size"])
        self.fuse = nn.Parameter(torch.randn(32, 1, 512))  # 初始化为随机值，可以根据需要调整，第二个维度是可以扩充的维度


    def forward(self, img_feat, lang_feat, inputs):
        img_feat_mask = torch.zeros([img_feat.shape[0], 1, 1, img_feat.shape[1]], dtype=torch.bool, device=img_feat.device)
        # (bs, 1, 1, grid_num) image：32*49*768，text：32*100*768

        # 在此处可以放入图像数据和文本数据的交互内容mmsd2.0
        # 计算交互模块
        text_embeds = self.text_projection(lang_feat)
        image_embeds = self.visual_projection(img_feat)
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)  # 在len维度上拼接
        input_embeds = torch.cat((self.fuse[:input_embeds.shape[0], :, :], input_embeds), dim=1)
        attention_mask = torch.cat(
            (torch.ones(lang_feat.shape[0], 50).to(lang_feat.device), inputs[self.input3]), dim=-1)  # 创建一个掩码矩阵，这里前面那用一个图像形状大小的矩阵，拼接后面文本的矩阵， 这里的初始未49然后根据可学习向量进行扩充
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fuse_hiddens, all_attentions = self.trans(input_embeds, extended_attention_mask,
                                                  output_all_encoded_layers=False)  # 对图片特征和文本特征注意力，在这里确定
        fuse_hiddens = fuse_hiddens[-1][:, 1:, :]  # 确保是最后一层，因为增加了一个长度的可学习向量，所以这里删除第一个
        new_text_features = fuse_hiddens[:, 49:, :]  # 要50以后的文本特征内容
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['text'].device), inputs['text'].to(
                torch.int).argmax(dim=-1)
        ]  # 第一个部分是选定保留每一个样本，第二部分是锁定每个样本最后一个单词出现的位置，因为在代码里面文本最后一个是cls
        # 这里需要确定一下inputs里面，哪部分是文本的ids
        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)  # 每个图片样本第一个位置的特征，因为在代码里面图片第一个是cls
        text_weight = self.att(new_text_feature)  # 对每一个样本计算一个数字作为权重 32*512 -》 32*1
        image_weight = self.att(new_image_feature)
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1), dim=-1)  # 拼接起来，让概率分布到0-1
        tw, iw = att.split([1, 1], dim=-1)  # 计算的注意力分数
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature
        logits_fuse = self.classifier_fuse(fuse_feature)
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)

        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            inputs[self.input3].unsqueeze(1).unsqueeze(2),
            img_feat_mask
        )
        lang_feat = torch.mean(lang_feat, dim=1)
        img_feat = torch.mean(img_feat, dim=1)

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat, fuse_score
    # def forward(self, img_feat, lang_feat, lang_feat_mask):
    #     img_feat_mask = torch.zeros([img_feat.shape[0], 1, 1, img_feat.shape[1]], dtype=torch.bool,
    #                                 device=img_feat.device)
    #     # (bs, 1, 1, grid_num)
    #     lang_feat, img_feat = self.backbone(
    #         lang_feat,
    #         img_feat,
    #         lang_feat_mask,
    #         img_feat_mask
    #     )
    #
    #     lang_feat = torch.mean(lang_feat, dim=1)
    #     img_feat = torch.mean(img_feat, dim=1)
    #
    #     proj_feat = self.cls_layer(lang_feat, img_feat)
    #
    #     return proj_feat, lang_feat, img_feat
