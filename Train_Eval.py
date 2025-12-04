import os
from os.path import exists
from torch.nn.functional import log_softmax, pad
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import warnings
import argparse
from rc_util import *
from RC_Model import *
# from Analogy_Model import *
# from Additional_Experiments import do_table_6
# from Multichoice_Model import run_mc
import json
import pandas as pd
import csv
#######
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt##
###########
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from transformers import BertTokenizer, BertModel
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
import logging
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
from sklearn.metrics import f1_score
import copy
logger = logging.getLogger(__name__)
from Abstraction import *
###

# from Additional_Experiments import LLMs_RC
# from Additional_Experiments import solve_analogies





def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):


        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm,fl=False):
        x = self.generator(x)
        #print('x',x.shape)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        #print('loss,norm',loss)
   
        if fl:
            loss.backward()
            
        else:
            
            loss.backward(retain_graph=True)

        if self.opt is not None and fl:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data * norm


confusion_matrix={}
def get_original_label(y,relations,batch,i):
    sentence_flagged_idxs=batch['sentence_flagged_idxs'][i,:]
    abstracted_ents_flagged_idxs=batch['abstracted_ents_flagged_idxs'][i,:]


    sentence=tokenizer.convert_ids_to_tokens(sentence_flagged_idxs)
    abstract=tokenizer.convert_ids_to_tokens(abstracted_ents_flagged_idxs)
    #############
    temp=y[i].item()
    temp=Relation_counter_tacerd[temp]
    temp=str(temp).upper()


    temp=id2kbid[temp]

    y_t=tacerd_rel[temp]


    temp=relations[i].item()
    temp=Relation_counter_tacerd[temp]
    temp=str(temp).upper()

    temp=id2kbid[temp]
    r_t=tacerd_rel[temp]
    t=str(y_t)+'-'+str(r_t)
    if t in confusion_matrix:
        confusion_matrix[t]=confusion_matrix[t]+1
    else:
        confusion_matrix[t]=1
    print('y',y_t)
    print('r',r_t)
    print('sentence',sentence)
    print('abstract',abstract)
    print('confusion_matrix',confusion_matrix)
    print('****')


@torch.no_grad()    
def rc_eval(args,model,abstract,batch_data,device,idx2word='none'):

    torch.manual_seed(0)
    file='essential_files/'+args.data_type+'rel_dic.json'
    with open(file) as f:
        rel_dic = json.load(f)['rel_dic']
        rel_dic_rev = {y: x for x, y in rel_dic.items()}
    ####
# #
#     file_name='essential_files/id2prop.json'
#     id2kbid= json.load(open(file_name))['id2prop']
#     kbid2id= {y: x for x, y in id2kbid.items()}
    ####
    with open('essential_files/prop_wiki_all.json') as f:
        properties = json.load(f)['prop_wiki_all']
    id2kbid={}
    for n,p in enumerate(properties.keys()):
        p=p.lower()
        #n=str(n)
        id2kbid[n]=p
    #
    kbid2id= {y: x for x, y in id2kbid.items()}

    # print('kbid2id',kbid2id)


    subclass_f1={}



    # with open('essential_files/properties-with-labels.json') as f:
    #     properties = json.load(f)

    with open('essential_files/prop_wiki_all.json') as f:
        properties = json.load(f)['prop_wiki_all']

        properties_p_to_id = {y: x for x, y in properties.items()}



    file='essential_files/Relation_counter_'+str(args.data_type)+'.json'

    with open(file) as f:
        Relation_counter_p_int = json.load(f)
        Relation_counter_int_p = {y: x for x, y in Relation_counter_p_int.items()}


    #####################
    with open('essential_files/prop_wiki_all.json') as f:
        properties = json.load(f)['prop_wiki_all']

    ########
    id2kbid={}
    kbid_as_special_tokens_rev={}
    kbid_as_special_tokens={}
    for n,p in enumerate(properties.keys()):
        p=p.lower()
        id2kbid[p]=n
        kbid_as_special_tokens_rev[n]=p
        kbid_as_special_tokens[p]=n




    ################




    PREDICTIONS=[]
    Y=[]
    PREDICTIONS_h=[]
    Y_0=[]
    n_epochs=1

    not_correct_dic={}

    predictions_not_corect={}
    model.eval()

    F1_l=[]
    F1_h_l=[]
    for di,data in enumerate(batch_data):
        print('di',di)
        # if di==0:
        #     continue
        
        c_1=0
        c_total=0
        eval_epxeriment_data={}

        for i_d_,x in enumerate(tqdm(data)):  

            examples=x



            y=examples['y']
            y0=examples['y0']

            ########
            batch={}
            for key in examples.keys():
                batch[key]=examples[key].to(device) if key!='ids' else examples[key]
            ######################################
            ids=batch['ids']


            if (di!=1 and args.data_type=='retacred') or  True:
                eval_epxeriment_data=None
                #continue

            h,predictions,relations = model(batch,args,eval_epxeriment_data=eval_epxeriment_data,eval=True)

            relations=relations.argmax(axis=-1)

            h=h.argmax(axis=-1) if h!= None else 0
            ###
            y = y.to(device) 
            ###
            Y_Labels=[]
            for y_ in y.tolist():
                if args.data_type not in ['wikidata','semeval_2012']:
                    p=Relation_counter_int_p[y_]
                    p=p
                    p_id=kbid2id[p]
                    p_id=str(int(p_id))
                    rel_l=rel_dic_rev[p_id]
                    Y_Labels.append(rel_l)
                else:

                    p=Relation_counter_int_p[y_]
           
                    rel_l=p
                    Y_Labels.append(rel_l)

         
            correct = relations == y
            print('correct',correct)


            for ci, c in enumerate(correct):
                yl=Y_Labels[ci]
                if yl in subclass_f1.keys():
                    t=1 if relations[ci].item()==y[ci].item() else 0
                    if t==1:
                        t1,t2=1,1
                    else:
                        t1,t2=1,0
                    subclass_f1[yl]['relation'].append(t1)
                    subclass_f1[yl]['Y'].append(t2)
                else:
                    t=1 if relations[ci].item()==y[ci].item() else 0
                    if t==1:
                        t1,t2=1,1
                    else:
                        t1,t2=1,0
                    subclass_f1[yl]={'relation':[],'Y':[]}
                    subclass_f1[yl]['relation'].append(t1)
                    subclass_f1[yl]['Y'].append(t2)
            # print('correct',correct)
            # print('relations',relations)
            # print('y',y)
            # print('++++')
            check=True
            if check:

                for i in range(y.shape[0]):
                    c_total=c_total+1
                    if y[i].item()!=relations[i].item():
                        c_1=c_1+1
                        id=ids[i]
                        h_i=h[i].item() if h!=0 else 0
                        y0_i=y0[i].item()
                        t=h_i==y0_i
                 
                        if id not in predictions_not_corect.keys():

                            ground_truth=y[i].item()
                            predicted_truth=relations[i].item()
                            ground_truth_p=Relation_counter_int_p[ground_truth] if ground_truth in Relation_counter_int_p else ground_truth
                            ground_truth_label=properties[ground_truth_p.upper()] if ground_truth_p.upper() in properties.keys() else ground_truth
                            predicted_truth_p=Relation_counter_int_p[predicted_truth] if predicted_truth in Relation_counter_int_p else predicted_truth
                            predicted_truth_label=properties[predicted_truth_p.upper()] if predicted_truth_p.upper() in properties.keys() else predicted_truth
                            
                    
                            item={'ground_truth_label':ground_truth_label,\
                            'predicted_truth_label':predicted_truth_label,'id':id,\
                            'ground_truth_p':ground_truth_p,'predicted_truth_p':predicted_truth_p}
                            predictions_not_corect[id]=item

            if 'head_1' in args.heads:
                PREDICTIONS_h.extend(h.flatten().tolist())
                Y_0.extend(y0.flatten().tolist())
            PREDICTIONS.extend(relations.flatten().tolist())
            Y.extend(y.flatten().tolist())
        ##
        # h_data={'evaluation_data/predictions_not_corect':predictions_not_corect}
        # file='files/predictions_not_corect'+'_'+args.data_type+''+str(di)+'.json'
        # with open(file, 'w') as fp:
        #     json.dump(predictions_not_corect, fp)
        if di ==1 and args.data_type=='conll' :
            print('eval_epxeriment_data',eval_epxeriment_data.keys())
            cal_similarity(eval_epxeriment_data)
            print('NEGATIVE******************************************')
            cal_similarity(eval_epxeriment_data,negative=True)
        f1_macro=f1_score(Y, PREDICTIONS, average='macro')
        f1_micor=f1_score(Y, PREDICTIONS, average='micro')
        print('f1_macro',f1_macro)
        print('f1_micor',f1_micor)

        if 'head_1' in args.heads:
            f1_macro_h=f1_score(Y_0, PREDICTIONS_h, average='macro')
            f1_micor_h=f1_score(Y_0, PREDICTIONS_h, average='micro')
            print('f1_macro_h',f1_macro_h)
            print('f1_micor_h',f1_micor_h)


        print('subclass F1')
        dic_r_f1_micro={}
        dic_r_f1_macro={}

        for c in subclass_f1.keys():
            #print('class',c)
            f1_macro_h=f1_score(subclass_f1[c]['Y'], subclass_f1[c]['relation'], average='macro')
            f1_micor_h=f1_score(subclass_f1[c]['Y'], subclass_f1[c]['relation'], average='micro')
            # print('f1_macro_h',f1_macro_h)
            # print('f1_micor_h',f1_micor_h)
            dic_r_f1_micro[c]=f1_micor_h
            dic_r_f1_macro[c]=f1_macro_h

        dic_r_f1_micro_sorted=dict(sorted(dic_r_f1_micro.items(), key=lambda item: item[1]))
        dic_r_f1_macro_sorted=dict(sorted(dic_r_f1_macro.items(), key=lambda item: item[1]))
        # for r in dic_r_f1_micro_sorted.keys():
        #     f1_micor_h=dic_r_f1_micro_sorted[r]
        #     f1_macro_h=dic_r_f1_macro_sorted[r]
        #     print('r',r)
        #     print('f1_micor_h',f1_micor_h)
        #     print('f1_macro_h',f1_macro_h)
        #     print('####')
        for r in dic_r_f1_macro_sorted.keys():
            f1_macro_h=dic_r_f1_macro_sorted[r]
            f1_micor_h=dic_r_f1_micro_sorted[r]
            print('r',r)
            print('f1_macro_h',f1_macro_h)
            print('f1_micor_h',f1_micor_h)
            print('####')


    

    return f1_micor#F1_l[-1]




def georoc_train_eval(data_name,mode='train',exp_args=None,DATA=None,model_for_fine_tune=None):
    torch.manual_seed(64)
    if exp_args==None:

        model_name='bert-base-uncased'
        #model_name='roberta-large'
        
        #args.both_ab_not_ab='both'
        args=get_settings(mode,model_name,data_name)
        train_loader,dev_loader_b=rc_data(args)
        print('test',len(train_loader),len(dev_loader_b))
    else:
        args=exp_args
        train_loader,dev_loader_b=DATA['train'],DATA['dev']
        print('test',len(train_loader),len(dev_loader_b))
    
    
    abstract=args.abstract 

   
    model_to_train=args.model_to_train


    if model_to_train=='rc':
        model = Relation_Classifier_Model(
            args 
        ).to(args.device)
    elif model_to_train=='wordanalogy_re_model':
        model = Analogy_RE_Model(
           args
        ).to(args.device)
    ############
    Loss_Trend_pre=[]

    i_skip_pre=0
    start_epoch=1

    if args.load_from_checkpoint:

        if args.experiment_no in ['one','two','three','four','five']:
            args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.data_type)+'.t7'
            print('args.save_dir',args.save_dir)
        #args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_analogykb.t7'
        print('#args.experiment_no',args.experiment_no)
        #checkpoint = torch.load(args.save_dir,weights_only=False)#,map_location=torch.device('mps'))
        if args.device==torch.device('mps'):
            checkpoint = torch.load(args.save_dir,weights_only=False,map_location=torch.device('mps'))
            print()
        else:
            checkpoint = torch.load(args.save_dir,weights_only=False)#,map_location=torch.device('mps'))
        ##########
        if args.experiment_no in ['one','two','three','four','five',]:
          
            model.load_state_dict(checkpoint['state_dict'])
            learning_rate=checkpoint['learning_rate']
            args.lr=learning_rate


        Loss_Trend_pre=[]#checkpoint['Loss_Trend'] 

    print('mode',mode)
    if model_for_fine_tune!=None:
        pass
        #model=model_for_fine_tune

    ###########
    if mode=='train':
        v=args.vocab_size

        

        criterion = nn.CrossEntropyLoss()
        lr =args.lr
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=lr
        )
        if args.load_from_checkpoint and args.mode!='eval':
            pass
           #optimizer.load_state_dict(checkpoint['optimizer'])
        if 'head_conditional' in args.heads:
            criterion2 = LabelSmoothing(size=v, padding_idx=0, smoothing=0.0)
            SimpleLossCompute_=SimpleLossCompute(model.generator, criterion2, optimizer)
        else:
            SimpleLossCompute_=None

        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=12)
  

    
    ##
    if args.data_parallel:
        model = nn.DataParallel(model)# torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        model=model.to(device)

    Loss_Trend=Loss_Trend_pre+[] if args.load_from_checkpoint else [] 
    
    flag=0
    if mode =='eval':
        if args.data_type=='wordanalogy':
            args.train=False
            f1=rc_eval_wordanalogy(args,model,args.abstract,dev_loader_b,args.device)
            f1=0
        else:
            print()
            f1=rc_eval(args,model,abstract,dev_loader_b,args.device)
            print('end eval')
    F1=0

    if mode=='train':
        data=train_loader
        n_epochs=args.epcoh_n
        #initialize_relation_emb(model,optimizer,scheduler)
        for epoch in range(start_epoch,n_epochs):
                print('epoch',epoch)
                model.train()
                if args.model_to_train!='rc':
                    if epoch <= args.only_train_classifier:
                        model.set_only_classifier_train(True)
                    else:
                        model.set_only_classifier_train(False)
                epoch_loss = 0
                #model.set_only_head_train()
                
                train_model(model,optimizer,scheduler,data,dev_loader_b,args,SimpleLossCompute_,criterion,Loss_Trend,epoch,F1)
        
    return model

def train_model(model,optimizer,scheduler,data,dev_loader_b,args,SimpleLossCompute_,criterion,Loss_Trend,epoch,F1):
    device=args.device
    model_to_train=args.model_to_train

    for i_d,x in enumerate(tqdm(data)):
        examples=x

        #break



        batch={}
        for key in examples.keys():
            batch[key]=examples[key].to(device) if key!='ids' else examples[key]

        if args.data_type!='wordanalogy':
            y0=batch['y0']
            y=batch['y']
            if args.abstract=='flagged_ents':
                ents_flagged_idxs=batch['ents_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
                Len_Target =batch['Len_Target']

            elif args.abstract=='abstract':
                ents_flagged_idxs=batch['abstracted_ents_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['abstracted_ents_flagged_plus_rel_tokens'] 
                Len_Target =batch['abstract_Len_Target']
            elif args.abstract=='mix':
                ents_flagged_idxs=batch['EntsAbst_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['EntsAbst_flagged_plus_rel_tokens'] 
                Len_Target =batch['ent_abstract_Len_Target']
            else:
                ents_flagged_idxs=batch['EntsAbst_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['EntsAbst_flagged_plus_rel_tokens'] 
                Len_Target =batch['ent_abstract_Len_Target']
        
        ######################################
        if args.data_type=='wordanalogy':
            y0=batch['y0']

            if 'baseline'  in args.wordanalogy_model:
                label= batch['r']
                h,loss = model(batch,args)
                l=0
                loss=loss
            elif 'sentence' in args.wordanalogy_model:
                label= batch['r']
                label_= batch['r1']
                h,h_,l= model(batch,args)
                loss=criterion(h, label) if args.fin_tune==False else 0
                
                loss_=0#criterion(h_, label_) if args.fin_tune==False else 0

                loss=loss_+loss+l if args.fin_tune==False else l

            elif 'route' in args.wordanalogy_model:
                label= batch['r']

                h,l= model(batch,args)
                loss=criterion(h, label) if args.fin_tune==False else 0
                loss=loss+l if args.fin_tune==False else l
            else:
                label= torch.where(y0 == -1, 0, y0)
                # print('y0',y0)
                # print('label',label)
                h= model(batch,args)
                loss=criterion(h, label)
                #print('loss',loss) 

            loss.backward()
            optimizer.step()
            model.zero_grad()
            continue
  

        else:
            h,predictions,relations = model(batch,args)
            y = y.to(device) 
            loss_relation = criterion(relations, y) 
            print('loss_relation',loss_relation)
            loss_relation_h =criterion(h, y0) if 'head_1' in args.heads  else 0
            loss=loss_relation_h+loss_relation
            h=h.argmax(axis=1) if 'head_1' in args.heads  else h
     
            ###############
            correct = relations.argmax(axis=1) == y

            temp=False if args.abstract=='mask' else True

            # print('loss',loss)
            # print('relations.argmax(axis=1)',relations.argmax(axis=1))
            # print('y',y)
          
            loss.backward(retain_graph=True)
        ############

        labels =ents_flagged_plus_rel_idxs[:,1:,] if model_to_train=='rc' else ents_flagged_idxs[:,1:,]
        Len_Target=batch['Len_Target']
        if 'head_conditional' in args.heads:
            loss_deocoder = SimpleLossCompute_(predictions, labels,len(torch.unique(labels)))
            #print('loss_deocoder',loss_deocoder)
        ##############
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        ##
        optimizer.step()
        ##
        if (i_d+1)%args.scheduler_step==0:
            scheduler.step()
        model.zero_grad()
    scheduler.step()
    # if args.experiment_no !='paper2_EVALutionPretraining':
    #     args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.data_type)+'.t7'
    # else:
    args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'.t7'
    print('will save in ',args.save_dir)


    temp='None'
    state = {
            'Train_Args':args,
            'i_d':i_d,
            'epoch': epoch,
            'learning_rate':optimizer.param_groups[0]["lr"],
            'state_dict': deepcopy(model.state_dict()),
            'classifier_head':temp,
            'optimizer': optimizer.state_dict(),
    }
    savepath=args.save_dir

    if args.save :
        print('saving.....')
        print('savepath',savepath)
       
        torch.save(state,savepath)
    if args.data_type!='wordanalogy':
        f1=rc_eval(args,model,args.abstract,dev_loader_b,device)

    else:
        f1=rc_eval_wordanalogy(args,model,args.abstract,dev_loader_b,device)
   

    







