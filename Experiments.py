import os
from os.path import exists
from torch.nn.functional import log_softmax, pad
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import warnings
import argparse
from rc_util import *
from RC_Model import *
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

from Train_Eval import *
from rc_util import _get_pretrained_transformer3





def get_settings(mode,model_name,data_name):

    #tokenizer = AutoTokenizer.from_pretrained("./tokenizer/") if model_name=='roberta-large' else BertTokenizer.from_pretrained("./tokenizer/")

    tokenizer_special_dic='semeval_2012_re' if data_name=='semeval_2012' else 're' 
    if 'wordanalogy' in data_name :
        tokenizer_special_dic='default'

    _, _,tokenizer =  _get_pretrained_transformer3(model_name,tokenizer_special_dic=tokenizer_special_dic) 

    Train_Args= {
    "batch_size": 128,
    "data_type": 'wikidata',
    "dev_record_file": "dev.npz",
    "test_record_file": "test.npz",
    "dev_rev_record_file": "dev_rev.npz",
    "test_rev_record_file": "test_rev.npz",
    "lr": 1e-5,
    "name": "baseline",
    "save_dir": "./models_saved",
    "seed": 224,
    "train_record_file": "train.npz",
    "local_rank":-1,
    "data_parallel":False,
    'abstract':'flagged_ents',
    'mode':'eval',
    'model_to_train':'rc',#unsupervise_re ,,rc
    'load_from_checkpoint':False,
    'vocab_size':len(tokenizer),
    'data_parallel':False,
    'checkpoint_path':"models_saved",
    'epoch':1,
    'scheduler_step':1200,
    'f1_score':'micro',
    'print_eval':False,
    'num_workers':0,
    'embed_size':1024,
    'n_class':352,
    'save':True,
    'both_ab_not_ab':'',
    'heads':['head_3',],
    'model_name':model_name,
    'experiment':False,
    'h1_embed_size':2,
    'h2_embed_size':512,
    'h3_embed_size':512,
    'train':True,
    'wordanalogy_model':'baseline',
    'train_on_all':False,
    'filter_type':None,
    'abcd_attention':False,
    'epcoh_n':5,
    'fin_tune':False,
    'experiment_no':'four',
    'person_person':'all',
    'only_train_classifier':0,
    'similarity_measure':'offset',
    'wordanalogy_pretrain':False
    }


    t_1024=['t5-large','bert-large-uncased','roberta-large','opt','prophetnet']
    t_768=['gpt2','bert_base_uncased','roberta-base','flaxopt']
    t_512=['t5-small',]

    torch.manual_seed(0)
    
    args=Args_dic(Train_Args)


    if model_name in t_1024:
        args.embed_size=1024 
    elif model_name in t_768:
        args.embed_size=768
    elif model_name in t_512:
        args.embed_size=512

    args.mode=mode
    if mode=='eval':
        args.load_from_checkpoint=True
   
    
 
    args.data_type=data_name
    #temp=['BLESS','EVALution','CogALexV']
    if data_name=='tacred':
        args.f1_score='micro'

    elif args.data_type=='tacred':
        args.n_class=42
        args.h3_embed_size=42
        #args.combin_two=True
    elif args.data_type=='retacred':
        args.n_class=41
        args.h3_embed_size=41

    elif args.data_type=='wikidata':
        args.n_class=352
        args.h3_embed_size=352

    elif args.data_type=='conll':
        args.n_class=5
        args.h3_embed_size=5




    if args.abstract!='learn_abstract':
        args.save_dir=args.checkpoint_path+'/'+args.abstract+'/model_'+str(args.abstract)+'_'+str(args.data_type)+'.t7'

    #model_to_train='unsupervise_re'
    else:
        args.save_dir=args.checkpoint_path+'/model_'+str(args.abstract)+'_'+str(args.data_type)+'_learn_abstract.t7'

    # set n_gpu
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print('device',device)
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    if args.device==torch.device('cpu'):
        args.batch_size=2

    return args


 




def experiment_run(data_name,experiment,mode='train',model_to_train='rc',backend_model_name='bert-base-uncased'):
    print('test',experiment)
    model_name=backend_model_name#'bert-base-uncased'
    #model_name='roberta-large'
    ## default args
    print('args',model_name)

    args=get_settings(mode,model_name,data_name)
    args.model_name=backend_model_name

    print('#experiment',experiment)


    # according to  the paper: Experiment one
    # Experiment One: Comparative
    # Performance on Different Information
    # Granularity
    if str(experiment)=='one':
        abstracs_flag=['flagged_ents','abstract','mask']
       #abstracs_flag=['mix',]
        args.heads=list(('head_3',),)
        args.batch_size=24
        args.h3_embed_size=512
        args.epcoh_n=2
        if args.device==torch.device('mps'):
                args.batch_size=2
        print('args.heads',args.heads)
        train_loader,dev_loader_b=rc_data(args)
        DATA={'train':train_loader,'dev':dev_loader_b}
        args.experiment_no='one'


        args.load_from_checkpoint=False
        for a in abstracs_flag:
            args.abstract=a
            args.save=False
            args.experiment=True
            print('a',a)
            print('mode',mode)
            georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)
    # According to the paper: Experiment two
    #     Entity Types as Manual
    #     Label for Re-Representation
    elif str(experiment)=='two':

        abstracs_flag=['flagged_ents','abstract','mask']
        args.heads=list(('head_2',),)
        args.batch_size=24
        args.h2_embed_size=512
        args.epcoh_n=5
        if args.device==torch.device('mps'):
                args.batch_size=2
        print('args.heads',args.heads)
        args.experiment_no='two'
        train_loader,dev_loader_b=rc_data(args)
        DATA={'train':train_loader,'dev':dev_loader_b}


        args.load_from_checkpoint=False
        for a in abstracs_flag:
            args.abstract=a
            args.save=False
            args.experiment=True
            print('a',a)
            print('mode',mode)
            georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)
    #
    # Experiment three: 
    # Experiment Three: Performance on
    # Varying Number of Entity Types
    # this is only for retacred dataset
    elif str(experiment)=='three':
        heads=[('head_conditional',),]

        for h in heads:
            subset_flag=['person-person*','person-person',]
            #abstracs_flag=['mix',]
            args.heads=h
            args.experiment_no='three'
            args.batch_size=24
            args.h3_embed_size=512
            args.epcoh_n=3
            if args.device==torch.device('mps'):
                    args.batch_size=2
            print('args.heads',args.heads)
            train_loader,dev_loader_b=rc_data(args)
            DATA={'train':train_loader,'dev':dev_loader_b}

        

            args.load_from_checkpoint=False
            for a in subset_flag:
                args.person_person=a
                args.save=False
                args.experiment=True
                print('a',a)
                print('mode',mode)
                georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)
    #Experiment Four: Comparison with State
    # of the Art
    elif str(experiment)=='four':

        expert_head=('head_1','head_3','head_2','head_conditional')
        h3=('head_3',)

        h=h3
        print('h',h)

        args=get_settings(mode,model_name,data_name)
        args.heads=list(h)
        args.h3_embed_size=512
        args.abstract='mix'
        args.experiment_no='four'
        args.epcoh_n=5
        args.model_to_train=model_to_train#
        print('model_to_train',model_to_train)


        if 'head_1' in args.heads and 'head_conditional' in args.heads:
            args.batch_size=12
            args.load_from_checkpoint=True if mode=='train' else True

        else:
            args.batch_size=24
            args.load_from_checkpoint=False if mode=='train' else False
        args.save=True
        args.experiment=True
        args.load_from_checkpoint=True if mode!='train' else False
        if args.device==torch.device('mps'):
            args.batch_size=2
        print('batch_size',args.batch_size)
            #model_name='roberta-large'

     
        
        train_loader,dev_loader_b=rc_data(args)
        print('len',len(train_loader))
        print('len',len(dev_loader_b))
        DATA={'train':train_loader,'dev':dev_loader_b}

        georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)



    elif str(experiment)=='five':

        expert_head=('head_1','head_3','head_2','head_conditional')
        h3=('head_conditional',)

        h=h3
        print('h',h)

        args=get_settings(mode,model_name,data_name)
        args.heads=list(h)
        args.h3_embed_size=512
        args.abstract='mix'

        args.experiment_no='five'
        args.epcoh_n=7
        args.model_to_train=model_to_train#
        print('model_to_train',model_to_train)


        if 'head_1' in args.heads and 'head_conditional' in args.heads:
            args.batch_size=12
            args.load_from_checkpoint=True if mode=='train' else True

        else:
            args.batch_size=64
            args.load_from_checkpoint=False if mode=='train' else False
        args.save=True

        args.experiment=True
        args.load_from_checkpoint=False
        #args.load_from_checkpoint=True if mode!='train' else True
        if args.device==torch.device('mps') or args.device==torch.device('cpu'):
            args.batch_size=1
        print('batch_size',args.batch_size)
            #model_name='roberta-large'

     
        
        train_loader,dev_loader_b=rc_data(args)
        print('len',len(train_loader))
        print('len',len(dev_loader_b))
        DATA={'train':train_loader,'dev':dev_loader_b}

        georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)


    elif str(experiment)=='six':

        expert_head=('head_1','head_3','head_2','head_conditional')
        h3=('head_3',)

        h=h3
        print('h',h)

        args=get_settings(mode,model_name,data_name)
        args.heads=list(h)
        args.h3_embed_size=512
        args.abstract='mix'
        args.experiment_no='six'
        args.epcoh_n=10
        args.model_to_train=model_to_train#
        print('model_to_train',model_to_train)


        if 'head_1' in args.heads and 'head_conditional' in args.heads:
            args.batch_size=12
            args.load_from_checkpoint=True if mode=='train' else True

        else:
            args.batch_size=128
            args.load_from_checkpoint=False if mode=='train' else False
        args.save=True
        args.experiment=True
        args.load_from_checkpoint=True if mode!='train' else False
        if args.device==torch.device('mps') or args.device==torch.device('cpu'):
            args.batch_size=2
        print('batch_size',args.batch_size)
            #model_name='roberta-large'

     
        
        train_loader,dev_loader_b=rc_data(args)
        print('len',len(train_loader))
        print('len',len(dev_loader_b))
        DATA={'train':train_loader,'dev':dev_loader_b}

        georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)



