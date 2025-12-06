import os
from os.path import exists
from torch.nn.functional import log_softmax, pad
from transformers import BertTokenizer, BertModel
import logging
import transformers
from transformers.models.bert.configuration_bert import BertConfig 
from transformers.models.bert.modeling_bert import BertEmbeddings 
###
from heinsen_routing import EfficientVectorRouting as Routing
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
import torch.nn as nn
import torch
import math
from torch.nn.functional import normalize
##
from transformers import XLNetTokenizer
from transformers import AutoTokenizer, LongT5Model

import random
from itertools import combinations
import torch
import torch.nn.functional as F
from rc_util import _get_pretrained_transformer3





class RoutingHead(nn.Module):
    """Route [n pos, d_depth, d_inp] to [n_out, d_out] ([n_out] if d_out is 1)."""

    def __init__(self, transformer_config, kwds_by_routing):
        super().__init__()
        d_depth, d_inp = (transformer_config['d_depth'], kwds_by_routing[0]['d_inp'])
        self.normalize = nn.LayerNorm(d_inp, elementwise_affine=False)
        self.W = nn.Parameter(torch.ones(d_depth, d_inp))
        self.B = nn.Parameter(torch.zeros(d_depth, d_inp))
        self.route = nn.Sequential(*[Routing(**kwds) for kwds in kwds_by_routing])
        for name, param in self.route.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_normal_(param)


    def forward(self, x):
        x = self.normalize(x)      # [..., n pos, d_depth, d_inp]
        x = x * self.W + self.B    # [..., n pos, d_depth, d_inp]
        x = x.flatten(-3,-2)       # [..., n_inp, d_inp]
        x = self.route(x)          # [..., n_out, d_out]
        return x.squeeze(-1)       # if d_out is 1, remove it


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.gelu = nn.GELU()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = self.gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x




class Relation_Classifier_Model(nn.Module):
    """

    """
    def __init__(
        self,
        args

    ):
        super().__init__()

        vocab_size, d_model = args.vocab_size,args.embed_size

        self.embed_size=args.embed_size
        ####
        args.feed_y0=False
        model_name=args.model_name
        self.pretrained_name=model_name
        #self.tokenizer = get_tokenizer(model_name)
        self.d_model = d_model
        dim=0
        heads={'head_1':512,'head_2':args.h2_embed_size,'head_3':args.h3_embed_size,'head_conditional':args.embed_size}
        for h in heads.keys():
            if h in args.heads:
     
                dim=dim+heads[h]

 
        input_sizes=[dim,dim,dim,args.n_class]
        self.classifier=MLP(input_sizes)
        for name, param in self.classifier.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    if args.data_type!='wikidata':
                        nn.init.xavier_normal_(param)

        self.head=8
        self.args=args
        self.modality=args.model_name#'roberta-large' if model_name=='roberta-large'  else 'bert-base-uncased'

        tokenizer_special_dic='semeval_2012_re' if args.data_type=='semeval_2012' else 're'
        data_selected=args.data_type
        self.transformer_config, self.transformer,self.tokenizer =  _get_pretrained_transformer3(data_selected,self.modality,tokenizer_special_dic=tokenizer_special_dic) 


        self.transformer.resize_token_embeddings(len(self.tokenizer))
        #50633
        #self.transformer.resize_token_embeddings(50633)

        ###***head_conditional***
        if 'head_conditional' in args.heads:

            # encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=self.head,batch_first=True)
            # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
            # for name, param in self.transformer_encoder.named_parameters():
            #     if 'weight' in name and param.data.dim() == 2:
            #         nn.init.xavier_normal_(param)
                    #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

            ##########
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=self.head,batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            ##
            for name, param in self.transformer_decoder.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_normal_(param)
                    #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

            ln_tokenizer = len(self.tokenizer) if model_name=='roberta-large' else len(self.tokenizer)
            self.generator=Generator(self.embed_size,ln_tokenizer)
            
       
            for name, param in self.generator.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    if args.data_type!='wikidata':
                        nn.init.xavier_normal_(param)

                    else:
                        nn.init.xavier_normal_(param)


        self.device=args.device
        
        ###
        config = BertConfig(
                    num_hidden_layers=2, 
                    hidden_act="relu", 
                    num_attention_heads=2,
                    hidden_dropout_prob=0.5, 
                    vocab_size=args.vocab_size ,
                    hidden_size=args.embed_size
                        )

        ###########################################################
        if 'head_1' in args.heads:
            self.nn_linear2= nn.Linear(2,512,bias=False)
            for name, param in self.nn_linear2.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_normal_(param)



        #####
        d_emb = args.embed_size
        n_classes= args.h1_embed_size#args.n_class
        d_hid, n_cls = (d_emb, n_classes)
        n_hid=128
        if 'head_1' in args.heads:
            self.head_1 = RoutingHead(self.transformer_config, kwds_by_routing=[
                { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
            ])
            self.m = nn.Softmax(dim=-1)


        

        #####
        d_emb = args.embed_size
        n_classes= args.h2_embed_size
        d_hid, n_cls = (d_emb, n_classes)
        n_hid=256
        if 'head_2' in args.heads:
            self.head_2 = RoutingHead(self.transformer_config, kwds_by_routing=[
                { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
            ])
        # for p in self.head_.parameters():
        #     p.requires_grad = False
        # for p in self.head_.route[-1].parameters():
        #     p.requires_grad = True
        #####
        Test=True
        if Test:
            d_emb = args.embed_size
            n_classes= args.h3_embed_size
            d_hid, n_cls = (d_emb, n_classes)
            n_hid=256
        else:
            d_emb = args.embed_size
            n_classes= args.h2_embed_size
            d_hid, n_cls = (d_emb, n_classes)
            n_hid=512




        if 'head_3' in args.heads:
            self.head_3 = RoutingHead(self.transformer_config, kwds_by_routing=[
                { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
            ])
     


    def body(self, batch,args,head='head_2',eval_epxeriment_data=None,eval=False):
        #sentence_masked_idxs
        if args.abstract=='mask':
            if head=='head_1':
                sentence_flagged_idxs=batch['sentence_masked_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentence_masked_flagged_tokens_masks']  
            elif head=='head_2':
                sentence_flagged_idxs=batch['ents_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['ents_flagged_tokens_masks'] 
            elif head=='head_3':
                sentence_flagged_idxs=batch['sentence_masked_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentence_masked_flagged_tokens_masks']  
        elif args.abstract=='abstract':
            if head=='head_1':
                sentence_flagged_idxs=batch['sentecne_ents_abstracted_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentecne_ents_abstracted_flagged_tokens_masks']  
            elif head=='head_2':
                sentence_flagged_idxs=batch['abstracted_ents_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['abstracted_ents_flagged_tokens_masks']
            elif head=='head_3':
                sentence_flagged_idxs=batch['sentecne_ents_abstracted_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['sentecne_ents_abstracted_flagged_tokens_masks'] 
        elif args.abstract=='flagged_ents':
            if head=='head_1':
                sentence_flagged_idxs=batch['sentence_ents_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentence_ents_flagged_tokens_masks']  
            elif head=='head_2':
                sentence_flagged_idxs=batch['ents_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['ents_flagged_tokens_masks'] 
            elif head=='head_3':
                sentence_flagged_idxs=batch['sentence_ents_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['sentence_ents_flagged_tokens_masks'] 
        else:

            if head=='head_1':
                sentence_flagged_idxs=batch['sentecne_entabs_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['sentecne_entabs_flagged_tokens_masks'] 
            elif head=='head_2':
                sentence_flagged_idxs=batch['EntsAbst_flagged_tokens']
                sentence_flagged_masks_idxs=batch['EntsAbst_flagged_tokens_masks'] 
            elif head=='head_3':
                sentence_flagged_idxs=batch['sentecne_entabs_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentecne_entabs_flagged_tokens_masks']  

        input_ids=sentence_flagged_idxs  # x is a dict computed by tokenizer
        masks=sentence_flagged_masks_idxs
        tmp=['paper2_WikidataPretraining','paper2_EVALutionPretraining','paper2_lexicalZeroshotTraining','paper2_RE_Trained_lexicalTraining','six']
            # exit()
        if args.experiment_no in tmp :
            if args.abstract=='flagged_ents':


                input_ids=batch['ents_flagged_tokens']
                masks=batch['ents_flagged_tokens_masks']
            elif args.abstract=='mix':
                input_ids=batch['EntsAbst_flagged_plus_rel_tokens']
                masks=batch['EntsAbst_flagged_plus_rel_tokens_masks']
        x=self.transformer(input_ids=input_ids,attention_mask=masks).hidden_states
        # x_=[]
        # for xi in range(len(x)):
        #         if xi%2!=0:
        #             x_.append(x[xi])
        # x=x_

        if head=='head_3' and eval and (args.data_type=='retacred'  or  args.data_type=='conll' ):
            L=sentence_flagged_idxs
            self.get_head_tail_rep(L,batch,x,eval_epxeriment_data)
        x = torch.stack(x, dim=-2)  # [chunks in batch, n pos, d_depth, d_emb]
        return x

    def get_head_tail_rep(self,L,batch,x,eval_epxeriment_data=False,wordanalogy=False):

        e1_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
        e2_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])
        y=batch['y']

        sentence_flagged_idxs=L

        Batch_h_t=[]
 

        population=[b for b in range(L.shape[0])]
        selected=random.sample(population, int(len(population)/10))
        for b in range(L.shape[0]):
            if e1_se[0] in L.tolist()[b]:

                e1_start=L.tolist()[b].index(e1_se[0])
            else: 
                continue 
            if e1_se[1] in L.tolist()[b]:

                e1_end=L.tolist()[b].index(e1_se[1])
            else:
                e1_end=-1
            if e2_se[0] in L.tolist()[b]:
                e2_start=L.tolist()[b].index(e2_se[0])
            else:
                e2_start=-1
            if e2_se[1] in L.tolist()[b]:
                e2_end=L.tolist()[b].index(e2_se[1])
            else:
                e2_end=-1
            y=batch['y'][b].cpu().detach().item()
            item_id=batch['ids'][b]
     
            temp_eval_d={}
            if wordanalogy==False:
                for h in range(len(x)):
                    temp=x[h][b,e1_start:e1_end+1,:] 
                    head=temp.cpu().detach().numpy().tolist()
                    #print(temp.shape)
                    temp=x[h][b,e2_start:e2_end+1,:] 
                    tail=temp.cpu().detach().numpy().tolist()
                    temp_eval_d[h]={'head':head,'tail':tail}
                if eval_epxeriment_data!=None:
                    if y in eval_epxeriment_data.keys():
                        eval_epxeriment_data[y][item_id]=temp_eval_d
                    else:
                        eval_epxeriment_data[y]={}
                        eval_epxeriment_data[y][item_id]=temp_eval_d
            else:
                temp=[t[b,e1_start:e1_end+1,:]  for t in x]
                temp=torch.stack(temp,0)
                head=temp.unsqueeze(0)#.cpu().detach()
                temp=[t[b,e2_start:e2_end+1,:]  for t in x]
                temp=torch.stack(temp,0)
                tail=temp.unsqueeze(0)#.cpu().detach()
      
                temp_eval_d={'head':head,'tail':tail}
                
                Batch_h_t.append(temp_eval_d)

        if wordanalogy==True:
            H=[]
            T=[]
            for b in range(len(Batch_h_t)):
                head=Batch_h_t[b]['head']
                tail=Batch_h_t[b]['tail']
                H.append(torch.sum(head,-2))
                T.append(torch.sum(tail,-2))
 
            H=torch.concat(H,0)
            T=torch.concat(T,0)
            Batch_h_t={'head':H,'tail':T}
      
        return Batch_h_t


    def forward_head(self, batch,args,head='head_2',eval_epxeriment_data=None,eval=False,abstract_test=False):
        if head=='head_2':
            x = self.body(batch,args,eval=eval)
            h = self.head_2(x)
            return h 
        elif head=='head_3':
            x = self.body(batch,args,head='head_3',eval_epxeriment_data=eval_epxeriment_data,eval=eval)
            h = self.head_3(x)
            return h 
        elif head=='head_1':
            x = self.body(batch,args,head='head_1',eval=eval)
            h = self.head_1(x)
            x = normalize(h, p=1.0, dim = -1)
            t = self.m(x)
            y0=batch['y0']
            t=t.to(dtype=torch.float)
            x=self.nn_linear2(t)
            return x,h

    def forward(self, batch,args,eval_epxeriment_data=None,eval=False,abstract_test=False,return_offset=False):
        flg_n=-1 
        if args.abstract=='mask':
            sentence_flagged_idxs=batch['sentence_masked_flagged_tokens']
            sentence_flagged_masks_idxs=batch['sentence_masked_flagged_tokens_masks']
            ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
            Len_Target=batch['Len_Target']
        elif args.abstract=='abstract':
            sentence_flagged_idxs=batch['sentecne_ents_abstracted_flagged_tokens']
            sentence_flagged_masks_idxs=batch['sentecne_ents_abstracted_flagged_tokens_masks']
            ents_flagged_plus_rel_idxs=batch['abstracted_ents_flagged_plus_rel_tokens']
            Len_Target=batch['abstract_Len_Target']
        elif args.abstract=='flagged_ents':
            sentence_flagged_idxs=batch['sentence_ents_flagged_tokens']
            sentence_flagged_masks_idxs=batch['sentence_ents_flagged_tokens_masks']
            ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
            Len_Target=batch['Len_Target']
        else:
            sentence_flagged_idxs=batch['sentecne_entabs_flagged_tokens']
            sentence_flagged_masks_idxs=batch['sentecne_entabs_flagged_tokens_masks']
            ents_flagged_plus_rel_idxs=batch['EntsAbst_flagged_plus_rel_tokens']
            Len_Target=batch['ent_abstract_Len_Target']
            n=Len_Target[0]
            temp=ents_flagged_plus_rel_idxs[0,n-3]
        if 'head_conditional' in args.heads:
  


            Tgt_Sentence_idxs=ents_flagged_plus_rel_idxs
            x=self.transformer(input_ids=sentence_flagged_idxs,attention_mask=sentence_flagged_masks_idxs).hidden_states






            if (args.data_type=='retacred'  or  args.data_type=='conll') and eval_epxeriment_data!=None:
                L=sentence_flagged_idxs
                self.get_head_tail_rep(L,batch,x,eval_epxeriment_data)

              
            x1=x[flg_n]

        if 'head_1' in args.heads or 'head_2' in args.heads or 'head_3' in args.heads:
            if args.feed_y0==False:
                e,h=None,None
                h2,h3=None,None
                if 'head_1' in args.heads:
                    temp=True if eval==True else False
                    e,h = self.forward_head(batch,args,head='head_1',eval=temp,abstract_test=False)
                if 'head_2' in args.heads:
                    h2=self.forward_head(batch,args,head='head_2',eval=eval,abstract_test=False)
                if 'head_3' in args.heads:

                    h3=self.forward_head(batch,args,head='head_3',eval_epxeriment_data=eval_epxeriment_data,eval=eval,abstract_test=False)
                temp=['head_3','head_2','head_1']
                hs=[t for th,t in zip(temp,(h3,h2,e)) if th in args.heads]
                e=torch.cat(hs,-1)
            else:
                y0=batch['y0']
                e=self.y_emb(y0)

                h=None
        else:
            h=None
        if 'head_conditional' in args.heads:
            mask=torch.ones_like(ents_flagged_plus_rel_idxs).to(ents_flagged_plus_rel_idxs.device)
            ####
            for i in range(Len_Target.shape[0]):
                if Len_Target[i].item()<39:
                    row=Len_Target[i].item()
                    mask[i,row-3:]=0
            #####
            if args.data_type=='wikidata' :
                x2 = self.transformer(ents_flagged_plus_rel_idxs,attention_mask=mask)['hidden_states'][flg_n]
            else:
                x2 = self.transformer(ents_flagged_plus_rel_idxs,attention_mask=mask)['hidden_states'][flg_n]
            # x1=self.PositionalEncoding(x1)
            # x2=self.PositionalEncoding(x2)
            if args.data_type=='wikidata' :
                t=sentence_flagged_masks_idxs
                temp_mask=t.unsqueeze(-1).repeat(1*self.head,1,t.shape[1]).to(self.device)
                z1=x1#self.transformer_encoder(x1,mask=temp_mask.bool())

            else:
                z1=x1#self.transformer_encoder(x1,mask=sentence_flagged_masks_idxs)
            ####
            tgtmask=Tgt_Sentence_idxs[:,:]!=0
            tgtmask=tgtmask.unsqueeze(-1).repeat(1*self.head,1,tgtmask.shape[1]).to(self.device)
            mask=nn.Transformer.generate_square_subsequent_mask(x2.shape[1])
            mask[mask == float("0")] = 1.
            mask[mask == float("-Inf")] = 0
            mask=mask.unsqueeze(0).repeat(tgtmask.shape[0],1,1).to(self.device)
            tgtmask=torch.mul(tgtmask, mask)
            if args.data_type=='wikidata':
                z2=x2#self.transformer_encoder(x2,mask=tgtmask,is_causal=True)

            else:
                z2=self.transformer_encoder(x2,mask=tgtmask,is_causal=True)
            memory = z1
            tgt =z2[:,:-1] 
            tgt_mask=Tgt_Sentence_idxs[:,:-1]!=0
            tgt_mask=tgt_mask.unsqueeze(-1).repeat(1*self.head,1,tgt_mask.shape[1])
            mask=nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
            mask[mask == float("0")] = 1.
            mask[mask == float("-Inf")] = 0
            mask=mask.unsqueeze(0).repeat(tgt_mask.shape[0],1,1).to(self.device)
            tgt_mask=torch.mul(tgt_mask, mask)
            out = self.transformer_decoder(tgt, memory,tgt_mask=tgt_mask,tgt_is_causal=True)

            if abstract_test:
                return out
            rel_n=3
            #print('out[i,Len_Target[i].item()-rel_n,:]',tgt[0,Len_Target[i].item()-rel_n])
            for i in range(Len_Target.shape[0]):
                if Len_Target[i].item()>39:
                    if i==0:
                        r=torch.zeros(1,self.d_model).to(out.device)
                    else:
                        temp=torch.zeros(1,self.d_model).to(r.device)
                        r=torch.cat((r,temp),0)
                elif i==0:
                    r=out[i,Len_Target[i].item()-rel_n,:].unsqueeze(0).to(out.device)
                else:
                    temp=out[i,Len_Target[i].item()-rel_n,:].unsqueeze(0).to(r.device)
                    r=torch.cat((r,temp),0)
            x=r
            if return_offset:
                return r

        if 'head_1' in args.heads or 'head_2' in args.heads or 'head_3' in args.heads:
            if 'head_conditional' in args.heads:
                x=torch.cat((x,e),-1)
            else:
                out=None
                x=e
        else:
            x=x
        relations = self.classifier(x)
        return h,out,relations
