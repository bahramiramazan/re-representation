import torch 

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from ujson import load as json_load
import os
import re
import ujson as json
from collections import Counter
from args import get_setup_args
from codecs import open
import pandas as pd
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
from string import whitespace
import requests
import time
import copy
from Wikidata_Abstraction import *
from transformers import XLNetTokenizer
from transformers import AutoTokenizer, LongT5Model
from datasets import load_from_disk
from rc_util import _get_pretrained_transformer3

####

#
# file_name='essential_files/id2prop.json'
# id2kbid= json.load(open(file_name))['id2prop']

with open('essential_files/prop_wiki_all.json') as f:
    properties = json.load(f)['prop_wiki_all']

########
id2kbid={}
kbid_as_special_tokens_rev={}
kbid_as_special_tokens={}
for n,p in enumerate(properties.keys()):
    p=p.lower()
    n=str(n)
    id2kbid[n]=p
    kbid_as_special_tokens_rev[n]=p
    kbid_as_special_tokens[p]=n

SRE_Analogy={'wikidata':[],'retacred':[],'conll':[],'semeval':[]}


###
# with open('essential_files/kbid_as_special_tokens.json') as f:
#     kbid_as_special_tokens = json.load(f)


# kbid_as_special_tokens_rev={}
# for j , k in enumerate(kbid_as_special_tokens.keys()):
#     j=str(j)
#     kbid_as_special_tokens_rev[j]=k


rel_dic={}
word_analogy_types_dic={}
dic_keys_re_common=['sentence_ents_flagged_tokens','sentence_tokens','sentence_masked_flagged_tokens',\
'sentecne_ents_abstracted_flagged_tokens',\
'sentecne_entabs_flagged_tokens',\
'ents_flagged_tokens',\
'ents_flagged_plus_rel_tokens',\
'EntsAbst_flagged_tokens',\
'EntsAbst_flagged_plus_rel_tokens',\
'abstracted_ents_flagged_tokens',\
'abstracted_ents_flagged_plus_rel_tokens',\
'item_id','y','y0','subset']


dic_keys_re_semeval=['sentence_ents_flagged_tokens','sentence_tokens','sentence_masked_flagged_tokens',\
'ents_flagged_tokens',\
'ents_flagged_plus_rel_tokens',\
'item_id','y','y0','subset']



dic_keys_word_analogy=['r_label','a','b','c',\
'd','ab','cd','abcd','not_positive_not_negaative',
'item_id','y','y0','subset','r','r1','similarity']


with open('essential_files/instances_heirarchy.json') as f:
    instances_heirarchy = json.load(f)
    instances_heirarchy=instances_heirarchy['h']

word_analogy_dic={}
CNT_var={'count':0,'instance_of_not_found':set(),'subclass_of_not_found':set()}
q_dic_train={'train':set(),'dev':set(),'dev_total':set(),'uniqe_dev_edge':set(),'instance_of':set()}



def prepare_data(data_selected,tokenizer,entities_context,data,data_type,abstraced,related_entities=False):
    with open('essential_files/properties-with-labels.json') as f:
        properties = json.load(f)
    data_list=[]
    unique_instances=set()
    p_n_dic={'p':0,'neg':0}
    has_superclass_no_superclass_count={'has_no_superclass':0,'has_superclass':0}
    type_dic={}
    for doc_record_id,indx in enumerate(tqdm(range(len(data)))):
        if data_selected=='wikidata' :
            indx=str(indx)
        d=data[indx]

        if data_selected=='conll':
            entities=d['entities']
            tokens=d['tokens']
            E=d['relations']
            orig_id=d['orig_id']

            print('E',E)
            for e in E:
                head=e['head']
                tail=e['tail']
                type_=e['type']
                head_e=entities[head]
                tail_e=entities[tail]
                head_start,head_end=head_e['start'],head_e['end']
                tail_start,tail_end=tail_e['start'],tail_e['end']

                head_type=head_e['type']
                tail_type=tail_e['type']
                ##
                y0=0
                relation=type_
                if relation in rel_dic.keys():
                    id_=rel_dic[relation]
                    kbID=id2kbid[id_].lower()
                else:
                    rel_dic[relation]=str(len(rel_dic.keys()))
                    id_=rel_dic[relation]
                    kbID=id2kbid[id_].lower()

                n1=int(head_start)#-1 if len(head)!=1 else int(head[0])
                n2=int(head_end)+1 #if len(head)!=1 else int(head[-1])+1
                
                m1=int(tail_start)#if len(tail)!=1 else int(tail[0])
          
                m2=int(tail_end)+1# if len(tail)!=1 else int(tail[-1])+1
                ####
                sentence_without_flags=" ".join(tokens)
                ent1=tokens[n1:n2]
                ent2=tokens[m1:m2]
                e1=" ".join(ent1)
                e2=" ".join(ent2)
                ##
                analogy_item={'a':e1,'b':e2,'r':type_,'type':data_type,'kbID':relation}
                SRE_Analogy['conll'].append(analogy_item)
                e2=e2 
                label_head_e=e1.split(' ')
                label_tail_e=e2.split(' ')

                flags=[['[e11]'],['[e12]'],['[e21]'],['[e22]']]
                first='head'
                sentence_text=' '.join(tokens)
                head_label_of_indexed=head_type
                tail_label_of_indexed=tail_type
                temp_=kbID if (data_type!='dev' and data_type!='test') else 'MASK'
                ents_flagged_tokens=flags[0]+label_head_e+flags[1]+flags[2]+label_tail_e+flags[3]
                ents_flagged_plus_rel_tokens=ents_flagged_tokens.copy()+['['+str(temp_).lower()+']']
                sentence_flagged_tokens=tokens[:n1]+flags[0]+label_head_e+flags[1]+tokens[n2:m1]+flags[2]+label_tail_e+flags[3]+tokens[m2:]
                sentence_masked_tokens=tokens[:n1]+flags[0]+['[MASK]']+flags[1]+tokens[n2:m1]+flags[2]+['[MASK]']+flags[3]+tokens[m2:]
                sentecne_entabs_flagged_tokens=tokens[:n1]+flags[0]+label_head_e+['*']+[head_label_of_indexed]+['#']+flags[1]+tokens[n2:m1]+flags[2]+label_tail_e+label_tail_e+['@']+[tail_label_of_indexed]+['&']+flags[3]+tokens[m2:]

                flags=[['[e11]'],['[e12]'],['[e21]'],['[e22]']]

                flags_=[['[e11_]'],['[e12_]'],['[e21_]'],['[e22_]']]

                abstracted_ents_flagged_tokens=flags[0]+[head_label_of_indexed]+flags[1]+flags[2]+[tail_label_of_indexed]+flags[3]
                
                abstracted_ents_flagged_tokens_=flags_[0]+[head_label_of_indexed]+flags_[1]+flags_[2]+[tail_label_of_indexed]+flags_[3]

                abstracted_ents_flagged_plus_rel_tokens=ents_flagged_tokens.copy()+['['+str(temp_).lower()+']']

                abstracted_sentence_flagged_tokens=tokens[:n1]+flags[0]+[head_label_of_indexed]+flags[1]+tokens[n2:m1]+flags[2]+[tail_label_of_indexed]+flags[3]+tokens[m2:]
                #print('abstracted_sentence_flagged_tokens',abstracted_sentence_flagged_tokens)
                sentence_without_flags_tokens=tokens
                ##
                EntsAbst_flagged_tokens=flags[0]+label_head_e+['*']+[head_label_of_indexed]+['#']+flags[1]+flags[2]+label_tail_e+['@']+[tail_label_of_indexed]+['&']+flags[3]
                EntsAbst_flagged_plus_rel_tokens=ents_flagged_tokens.copy()+['['+str(temp_).lower()+']']
             
                item_id=str(doc_record_id)+'_'+str(orig_id)+'_'+str(0)
                y=kbID
                for f in flags:
                    f=f[0]
                    if f not in abstracted_sentence_flagged_tokens:
                        print('abstracted_sentence_flagged_tokens',abstracted_sentence_flagged_tokens)
                        exit()

                subset=1
                item={'sentence_ents_flagged_tokens':sentence_flagged_tokens,
                'sentence_tokens':tokens,
                'sentence_masked_flagged_tokens':sentence_masked_tokens,
                'sentecne_entabs_flagged_tokens':sentecne_entabs_flagged_tokens,
                'sentecne_ents_abstracted_flagged_tokens':abstracted_sentence_flagged_tokens,
                'ents_flagged_tokens':ents_flagged_tokens,
                'ents_flagged_plus_rel_tokens':ents_flagged_plus_rel_tokens,
                'EntsAbst_flagged_tokens':EntsAbst_flagged_tokens,
                'EntsAbst_flagged_plus_rel_tokens':EntsAbst_flagged_plus_rel_tokens,
                'abstracted_ents_flagged_tokens':abstracted_ents_flagged_tokens,
                'abstracted_ents_flagged_plus_rel_tokens':abstracted_ents_flagged_plus_rel_tokens,
                'item_id':item_id,'y':y,'y0':y0 ,'eval_d':d,'subset':subset}
                data_list.append(item)
        if data_selected=='tacred' or data_selected=='retacred':
            subj_start, subj_end=d['subj_start'],d['subj_end']
            obj_start, obj_end=d['obj_start'],d['obj_end']
            relation,docid,id=d['relation'],d['docid'],d['id']
            if data_type=='dev' and False :
                if id in patch_dic.keys():
                    relation=patch_dic[id]['relation']
                    print('relations',relation)
                    exit()
            tokens=d['token']
            subj_type=d['subj_type'].split(' ')
            obj_type=d['obj_type'].split(' ')
            first='head' if subj_start>obj_start else 'tail'
            temp=subj_type+obj_type 
            temp='-'.join(temp)
            if temp not in type_dic.keys():
                type_dic[temp]=1
            else:
                type_dic[temp]=type_dic[temp]+1
            if temp!='PERSON-PERSON':
                subset=1
                #continue
            else:
                subset=0
                #continue
            token_anonymous=[]
            token_flagged=[]
            token_only_ents_flagged=[]
            token_masked=[]
            subj_type_temp=['*']+subj_type+['@']
            object_type_temp=['#']+obj_type+['^']

            subj_type_temp_=subj_type.copy()
            object_type_temp_=obj_type.copy()
            e1=[]
            e2=[]
            token_simple_flagged=[]
            for ti,t in enumerate(tokens):
                temp1=False
                temp2=False
        
                if ti ==subj_start:
                    token_flagged.append('[e11]')
                    token_flagged.append(t)
                    e1.append(t)
                    token_only_ents_flagged.append('[e11]')
                    token_only_ents_flagged.append(t)
                    #token_masked.append('[e11]')
                    token_masked.append('[MASK]')
                    #token_masked.append('[e21]')
                    token_anonymous.append('[e11]')
                    for t2 in subj_type:
                        token_anonymous.append(t2)
                    token_anonymous.append('[e12]')
                    if subj_start==subj_end:
                        token_flagged.extend(subj_type_temp)
                        token_flagged.append('[e12]')
                        token_only_ents_flagged.append('[e12]')

                elif subj_start<=ti and ti<= subj_end:
                    token_flagged.append(t)
                    e1.append(t)
               
                    if ti==subj_end :
                        token_flagged.extend(subj_type_temp)
                        token_flagged.append('[e12]')
                        #token_only_ents_flagged.extend(subj_type_temp_)
                        token_only_ents_flagged.append('[e12]')
                elif ti ==obj_start:
                    token_flagged.append('[e21]')
                    token_flagged.append(t)
                    token_anonymous.append('[e21]')
                    token_only_ents_flagged.append('[e21]')
                    token_only_ents_flagged.append(t)
                    e2.append(t)
                    token_masked.append('[MASK]')
                    for t2 in obj_type:
                        token_anonymous.append(t2)
                    token_anonymous.append('[e22]')
                    if obj_start==obj_end:
                        token_flagged.extend(object_type_temp)
                        token_flagged.append('[e22]')
                        token_only_ents_flagged.append('[e22]')

                elif obj_start<=ti and ti <=obj_end:
                    token_flagged.append(t)
                    e2.append(t)
                    token_only_ents_flagged.append(t)
                    if ti==obj_end :
                        token_flagged.extend(object_type_temp)
                        token_flagged.append('[e22]')
                        token_only_ents_flagged.append('[e22]')
                else:
                    token_flagged.append(t)
                    token_anonymous.append(t)
                    token_masked.append(t)
                    token_only_ents_flagged.append(t)
            sentence_orig=' '.join(tokens)
            sentence_anonymous=' '.join(token_anonymous)
            sentence_flagged=' '.join(token_only_ents_flagged)
            sentence_masked=' '.join(token_masked)
            sbj_label=tokens[subj_start:subj_end+1]
            obj_label=tokens[obj_start:obj_end+1]
            sentecne_entabs_flagged_tokens=token_flagged
            abstracted_sentence_flagged_tokens=token_anonymous
            sentence_without_flags_tokens=tokens
            sentence_masked_tokens=token_masked
            sentence_flagged_tokens=token_only_ents_flagged
            e1=' '.join(e1)
            e2=' '.join(e2)
            if data_selected=='retacred':
                analogy_item={'a':e1,'b':e2,'r':relation,'type':data_type,'kbID':'retacred'}
                SRE_Analogy['retacred'].append(analogy_item)
            y0=0
            if relation=='no_relation':
                y0=1

            else:
                y0=0
            if relation in rel_dic.keys():
                id_=rel_dic[relation]
                kbID=id2kbid[id_].lower()
            else:
                rel_dic[relation]=str(len(rel_dic.keys()))
                id_=rel_dic[relation]
                kbID=id2kbid[id_].lower()

            kbID_place_holder=copy.deepcopy(kbID) if (data_type!='dev' and data_type!='test') else 'MASK'
            EntsAbst_flagged_tokens=['[e11]']+sbj_label+ subj_type_temp+['[e12]'] +['[e21]']+ obj_label +object_type_temp+['[e22]']#if first=='head' else  obj_replacement+ ' '+  sub_replacement
            EntsAbst_flagged_plus_rel_tokens=['[e11]']+sbj_label+subj_type_temp+['[e12]'] +['[e21]']+ obj_label +object_type_temp+['[e22]']+[' ['+str(kbID_place_holder).lower()+']' ]#if first=='head' else  obj_replacement+ ' '+ sub_replacement+' ['+str(kbID).lower()+']'
            #####################
            ents_flagged_tokens=['[e11]']+sbj_label+['[e12]'] +['[e21]']+ obj_label +['[e22]']#if first=='head' else  obj_replacement+ ' '+  sub_replacement
            ents_flagged_plus_rel_tokens=['[e11]']+sbj_label+['[e12]'] +['[e21]']+ obj_label +['[e22]']+[' ['+str(kbID_place_holder).lower()+']' ]#if first=='head' else  obj_replacement+ ' '+ sub_replacement+' ['+str(kbID).lower()+']'
            abstracted_ents_flagged_tokens_=['[e11_]']+subj_type_temp_+['[e12_]'] +['[e21_]']+ object_type_temp_ +['[e22_]']
            abstracted_ents_flagged_tokens=['[e11]']+subj_type_temp_+['[e12]'] +['[e21]']+ object_type_temp_ +['[e22]'] 
            abstracted_ents_flagged_plus_rel_tokens=['[e11]']+subj_type_temp_+['[e12]'] +['[e21]']+ object_type_temp_ +['[e22]'] +[' ['+str(kbID_place_holder).lower()+']' ]#
            token_flagged_all_ents=[]
            token_abstracted_flagged_all_ents=[]
            ents_neut_flagged_shuffled_tokens=[]
            all_ent_in_sent_tokens=[]
            instances_flagged_tokens=[]
            entity_description_tokens=[]
            item_id=str(doc_record_id)+'_'+str(id)+'_'+str(0)
            y=kbID
            item={'sentence_ents_flagged_tokens':sentence_flagged_tokens,
            'sentence_tokens':tokens,
            'sentence_masked_flagged_tokens':sentence_masked_tokens,
            'sentecne_entabs_flagged_tokens':sentecne_entabs_flagged_tokens,
            'sentecne_ents_abstracted_flagged_tokens':abstracted_sentence_flagged_tokens,
            'ents_flagged_tokens':ents_flagged_tokens,
            'ents_flagged_plus_rel_tokens':ents_flagged_plus_rel_tokens,
            'EntsAbst_flagged_tokens':EntsAbst_flagged_tokens,
            'EntsAbst_flagged_plus_rel_tokens':EntsAbst_flagged_plus_rel_tokens,
            'abstracted_ents_flagged_tokens':abstracted_ents_flagged_tokens,
            'abstracted_ents_flagged_plus_rel_tokens':abstracted_ents_flagged_plus_rel_tokens,
            'item_id':item_id,'y':y,'y0':y0 ,'eval_d':d,'subset':subset}
            data_list.append(item)
 
     
 
        elif data_selected=='wikidata':
            
            tokens=d['tokens']

            sent=" ".join(tokens)
            edges=d['edges']
            for e in edges:
                # print('+++++++')
                right=list(e['right'])
                #print(right)
                if len(right)==0:
                    continue
                if len(right)==1:
                    right_tokens=tokens[right[0]]
                else:
                    right_tokens=tokens[right[0]:right[-1]]
                left=list(e['left'])
                #print('left',left)
                if len(left)==0:
                    continue
                if len(left)==1:
                    left_tokens=tokens[left[0]]

                else:
                    left_tokens=tokens[left[0]:left[-1]]
                
                kbID=e['kbID']
            vertexSet=d['vertexSet']
            rel_label=properties[kbID] if kbID in properties.keys() else 'empty'

            token_flagged_all_ents,token_abstracted_flagged_all_ents,abstracted_q_to_labels=\
            anonymous_flag_all_ents(entities_context,instances_heirarchy,tokens,vertexSet,has_superclass_no_superclass_count)

            vertexDic={}
            for v in vertexSet:
                pos=tuple(v['tokenpositions'])
                vertexDic[pos]=v['kbID']
            edges=d['edges']
            tokens=d['tokens']
            sent=" ".join(tokens)
            ents_found={}
            ENTS=[]
            ENTS_set=set()
            ents_dic={}
            ents_dic_={}
            for v in vertexSet:
                name=v['lexicalInput']
                if name in ENTS_set:
                    continue
                text=name 
                kbID=v['kbID']
                e_tokens=name.split(' ')
                ENTS.append((text,e_tokens))
                ENTS_set.add(text)
            for ei,e in enumerate(edges):
                head=e['left']
                tail=e['right']
                kbID=e['kbID']
                head_id=vertexDic[tuple(head)]
                tail_id=vertexDic[tuple(tail)]
                head_context='None'
                tail_context='None'
                head_aliases='None'
                tail_aliases='None'
                label_head='None'
                label_tail='None'
                instances_head=[]
                instances_tail=[]
                if len(head)<1 or len(tail)<1:
                    continue
                if head_id in entities_context:
                    head_context=entities_context[head_id]['desc']
                    head_aliases=entities_context[head_id]['aliases']
                    label_head=entities_context[head_id]['label']
                    instances_head=entities_context[head_id]['instances']
                    head_aliases=" , ".join(str(x) for x in head_aliases)
                    head_context_split=head_context.split()
                    if len(head_context_split)>20:
                        head_context=' '.join(head_context_split[:20])

                if tail_id in entities_context:
                    tail_context=entities_context[tail_id]['desc']
                    tail_aliases=entities_context[tail_id]['aliases']
                    label_tail=entities_context[tail_id]['label']
                    instances_tail=entities_context[tail_id]['instances']
                    tail_aliases=" , ".join(str(x) for x in tail_aliases)
                    tail_context_split=tail_context.split()
                    if len(tail_context_split)>20:
                        tail_context=' '.join(tail_context_split[:20])
                abstracted_head=abstracted_q_to_labels[head_id]
                abstracted_tail=abstracted_q_to_labels[tail_id]

                if rel_label=='instance of' or rel_label=='subclass of' :
                    abstracted_head=label_head
                    abstracted_tail=label_tail
                if kbID=='P0':
                    y0=1
                    p_n_dic['neg']= p_n_dic['neg']+1
                else:
                    y0=0
                    p_n_dic['p']= p_n_dic['p']+1
                rel_label=properties[kbID] if kbID in properties.keys() else 'empty'

                for ins in instances_head:
                    kbID_h_ins=ins['kbID'] if 'kbID' in ins.keys() else 'em'
                    unique_instances.add(kbID_h_ins)
                for ins in instances_tail:
                    kbID_t_ins=ins['kbID'] if 'kbID' in ins.keys() else 'em'
                    unique_instances.add(kbID_t_ins)
                
                n1=int(head[0])
                n2=int(head[-1])+1 

                m1=int(tail[0])

                m2=int(tail[-1])+1#

                first='head' #if m1>= n2 else 'tail'

                head_tail_pairs=[]
                abstracted_sentence_tokens_list=[]
                instances_head_labels=[abstracted_head]
                instances_tail_labels=[abstracted_tail]
                sentence_text=' '.join(tokens)



                for t1 in instances_head_labels:
                    head_label_of_indexed=t1
                    for t2 in instances_tail_labels:
                        tail_label_of_indexed=t2
                        tokens_=None
                        flags=[['[e11]'],['[e12]'],['[e21]'],['[e22]']]
                        flags_=[['[e11_]'],['[e12_]'],['[e21_]'],['[e22_]']]
                        if first=='head':
                            ents_flagged_tokens=flags[0]+[head_label_of_indexed]+flags[1]+flags[2]+[tail_label_of_indexed]+flags[3]
                            ents_flagged_tokens_=flags_[0]+[head_label_of_indexed]+flags_[1]+flags_[2]+[tail_label_of_indexed]+flags_[3]
                            ents_flagged_plus_rel_tokens=ents_flagged_tokens+['['+str(kbID).lower()+']']
                            tokens_=tokens[:n1]+flags[0]+[head_label_of_indexed]+flags[1]+tokens[n2:m1]+flags[2]+[tail_label_of_indexed]+flags[3]+tokens[m2:]
                        abstracted_sentence=' '.join(tokens_)
                        item={'ents_flagged_tokens_':ents_flagged_tokens_,'tokens':tokens_,'ents_flagged_tokens':ents_flagged_tokens,'ents_flagged_plus_rel_tokens':ents_flagged_plus_rel_tokens}
                  
                        abstracted_sentence_tokens_list.append(item)
                #######################################
                instances_head_labels=['[e11]']+instances_head_labels+['[e12]']
                instances_tail_labels=['[e21]']+instances_tail_labels+['[e22]']
                instances_flagged_tokens=[]
                if first=='head':
                    instances_flagged_tokens=instances_head_labels+instances_tail_labels
                else:
                    instances_flagged_tokens=instances_tail_labels+instances_head_labels
                ####################################
                if len(head)==0 or len(tail)==0:
                    continue
                n1=int(head[0])#-1 if len(head)!=1 else int(head[0])
                n2=int(head[-1])+1 #if len(head)!=1 else int(head[-1])+1
                
                m1=int(tail[0])#if len(tail)!=1 else int(tail[0])
          
                m2=int(tail[-1])+1# if len(tail)!=1 else int(tail[-1])+1
                ####
                sentence_without_flags=" ".join(tokens)
                ent1=tokens[n1:n2]
                ent2=tokens[m1:m2]
                e1=" ".join(ent1)
                e2=" ".join(ent2)
                if sentence_text=='Sanphebagar is a municipality of Achham District in the Seti Zone of western Nepal .':
                    print('sentence_text',sentence_text)

                    print('e1',e1)
                    print('e2',e2)
                    print('rel_label',rel_label)
                    print('++++')
                    #exit()
                ########

                ##
                analogy_item={'a':e1,'b':e2,'r':rel_label,'type':data_type,'kbID':kbID,'e1_abstract':abstracted_head,'e2_abstract':abstracted_tail}
                SRE_Analogy['wikidata'].append(analogy_item)
                ###############
                place_holder_ner=19
                place_holder_pos='other'
                label_head_e=label_head.split(' ')
                label_tail_e=label_tail.split(' ')

                flags=[['[e11]'],['[e12]'],['[e21]'],['[e22]']]
                if first=='head':
                    EntsAbst_flagged_tokens=flags[0]+label_head_e+['*']+[abstracted_head]+['#']+flags[1]+flags[2]+label_tail_e+['@']+[abstracted_tail]+['&']+flags[3]
                    EntsAbst_flagged_tokens=ents_flagged_tokens+['['+str(kbID).lower()+']']
                    ents_flagged_tokens=flags[0]+label_head_e+flags[1]+['is','to']+flags[2]+label_tail_e+flags[3]
                    ents_flagged_plus_rel_tokens=ents_flagged_tokens+['['+str(kbID).lower()+']']
                    sentecne_entabs_flagged_tokens=tokens[:n1]+flags[0]+label_head_e+['*']+[abstracted_head]+['#']+flags[1]+tokens[n2:m1]+flags[2]+label_tail_e+['@']+[abstracted_head]+['&']+flags[3]+tokens[m2:]
                    sentence_flagged_tokens=tokens[:n1]+flags[0]+label_head_e+flags[1]+tokens[n2:m1]+flags[2]+label_tail_e+flags[3]+tokens[m2:]
                    sentence_masked_tokens=tokens[:n1]+flags[0]+['[MASK]']+flags[1]+tokens[n2:m1]+flags[2]+['[MASK]']+flags[3]+tokens[m2:]

                L=[label_head_e,label_tail_e]
                random.shuffle(L)
                ents_neut_flagged_shuffled_tokens=['[es]']+L[0]+['[en]']+['[es]']+L[1]+['[en]']
                place_holder_ner=19
                place_holder_pos='other'
                ####
                place_holder_rel_pos='rel'
                place_holder_ner=20
                ents_flagged_plus_rel_tokens=ents_flagged_tokens+['['+str(e['kbID'])+']']
                all_ent_in_sent_tokens=[]
                all_ent_in_sent_ner_ids=[]
                separator=['[entsep]']
                separator_pos=['other']
                separator_ner_id=[19]
                for ent_ in ENTS:
                    text,e_tokens=ent_
                    all_ent_in_sent_tokens.extend(e_tokens)
                    all_ent_in_sent_tokens.extend(separator)
                ####
                sentence_without_flags_tokens=tokens

                ###
                sent=head_context#" ".join(head_context)
                head_entity_description_tokens=[],[]#word_tokenize(sent)
                head_entity_description_tokens=head_context.split(' ')
                sent=tail_context#" ".join(tail_context)
                tail_entity_description_tokens=[],[]#word_tokenize(sent)
                tail_entity_description_tokens=tail_context.split(' ')

                if first=='head':
                    entity_description_tokens=['[e11]']+head_entity_description_tokens+['[e12]']+['[e21]']+tail_entity_description_tokens+['[e22]']
                    head_entity_description_tokens=['[e11]']+head_entity_description_tokens+['[e12]']
                    tail_entity_description_tokens=['[e12]']+['[e21]']+tail_entity_description_tokens+['[e22]']
            
                else:
                    entity_description_tokens=['[e21]']+tail_entity_description_tokens+['[e22]']+['[e11]']+head_entity_description_tokens+['[e12]']
                y=e['kbID']
                for itemi,item in enumerate(abstracted_sentence_tokens_list):
                    subset=1
                    item_id=str(doc_record_id)+'_'+str(ei)+'_'+str(itemi)
                    abstracted_sentence_flagged_tokens=item['tokens']
                    abstracted_ents_flagged_tokens=item['ents_flagged_tokens']
                    abstracted_ents_flagged_plus_rel_tokens=item['ents_flagged_plus_rel_tokens']
                    abstracted_ents_flagged_tokens_=item['ents_flagged_tokens_']
                    item={'sentence_ents_flagged_tokens':sentence_flagged_tokens,
                    'sentence_tokens':tokens,
                    'sentence_masked_flagged_tokens':sentence_masked_tokens,
                    'sentecne_entabs_flagged_tokens':sentecne_entabs_flagged_tokens,
                    'sentecne_ents_abstracted_flagged_tokens':abstracted_sentence_flagged_tokens,
                    'ents_flagged_tokens':ents_flagged_tokens,
                    'ents_flagged_plus_rel_tokens':ents_flagged_plus_rel_tokens,
                    'EntsAbst_flagged_tokens':ents_flagged_tokens,
                    'EntsAbst_flagged_plus_rel_tokens':ents_flagged_plus_rel_tokens,
                    'abstracted_ents_flagged_tokens':abstracted_ents_flagged_tokens,
                    'abstracted_ents_flagged_plus_rel_tokens':abstracted_ents_flagged_plus_rel_tokens,
                    'item_id':item_id,'y':y,'y0':y0,'eval_d':d ,'subset':subset}
                    data_list.append(item)
                    # print('item',item['ents_flagged_tokens'])
                    # exit()

    #############################


    ########################
    file='essential_files/'+str(data_selected)+'rel_dic.json'
    h_data={'rel_dic':rel_dic}
    with open(file, 'w') as fp:
        json.dump(h_data, fp)
    print('rel_dic',rel_dic)

    return data_list
#####


def process_file(tokenizer,filename, data_type,Relation_counter,data_selected):
    # temp=['BLESS','EVALution','CogALexV']
    # temp=['BLESS','EVALution','CogALexV','ROOT09','KandH_plus_N','semeval_2012']
    def bert_tokenize(tokens):
        text=' '.join(tokens)
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        return tokenized_text
    if data_selected=='conll':
        datasets = load_from_disk('unprocessed_data/conll04')
        temp='train' if data_type=='train' else 'test'
        data=datasets[temp]
   
    else:
        with open(filename) as data_file:
            data = json.load(data_file)
    if data_selected=='wikidata':
        entities_context = json.load(open('unprocessed_data/wikipedia_distant//entities_context.json'))
    elif  data_selected=='nyt':
        entities_context = json.load(open('unprocessed_data/NYT//dataset_context_en_all.json'))

    else:
        entities_context={}

    data_prepared=prepare_data(data_selected,tokenizer,entities_context,data,data_type,'abstracted')
    examples = []
    eval_examples = {}
    total = 0
    i=0
    rel_frequency={}
    rel_data_dic={}
    balance=False
    keys_list=[]
    if data_selected not in ['none']:
        keys_list=dic_keys_re_common 


    for index,d in enumerate(tqdm(data_prepared)):
        i=i+1
        temp_dic={}
        for k in keys_list:
            temp=d[k]
            if k=='y' and data_selected!='wordanalogy':
                if str(d[k]) not in Relation_counter:
                    Relation_counter[str(d[k])]=len(Relation_counter)
                temp=Relation_counter[str(d[k])]
            elif k not in ['item_id','y0','y','not_positive_not_negaative','subset','r','r1','similarity']:
                temp=bert_tokenize(temp)
            else:
                temp=temp
            temp_dic[k]=temp
        examples.append(temp_dic)
        eval_examples[str(d['item_id'])] = {"supports": d['eval_d'],}
        total=total+1
    return examples, eval_examples





def build_features(tokenizer,args, examples, data_type, out_file, is_test=False):
    Sentence_limit = 150 if args.data!='wordanalogy' else 150
    ent_limit = 75
    char_limit =75# args.char_limit
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model.eval()
    temp=['none',]

    def idx_sen(tokenized_text):
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)
        if len(tokenized_text)!=len(indexed_tokens):
            print('not equal')
            exit()
        return indexed_tokens,segments_ids

    def store_idx_in_numpy(sentence_tokens,sentence_idx,segments_ids,limit):

        sentence_tokens=copy.deepcopy(sentence_tokens)
        sentence_idx=copy.deepcopy(sentence_idx)
        segments_ids=copy.deepcopy(segments_ids)
        temp_idx,temp_segment_idx=idx_sen(sentence_tokens)
        n=len(temp_idx)
        if n>limit:
            n=limit
        sentence_idx[:n]=temp_idx[:n]
        segments_ids[:n]=temp_segment_idx[:n]
        return sentence_idx,segments_ids,n
 

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["abstract_tokens"]) > abstract_limit or \
                   len(ex["title_tokens"]) > title_limit 
        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    Len_Target=[]
    abstract_Len_Target=[]
    ##
    if args.data not in ['none',]:
        L_s=[\
        'sentence_ents_flagged_tokens',
        'sentence_ents_flagged_tokens_masks',
        'sentence_tokens',
        'sentence_tokens_masks',

        "sentence_masked_flagged_tokens" ,
        "sentence_masked_flagged_tokens_masks" ,

        'sentecne_entabs_flagged_tokens',
        'sentecne_entabs_flagged_tokens_masks'

        "sentecne_ents_abstracted_flagged_tokens",
        "sentecne_ents_abstracted_flagged_tokens_masks",

        ]
        L1=[\
            'ents_flagged_tokens',\
            'ents_flagged_tokens_masks',\

            'ents_flagged_plus_rel_tokens',\
            'ents_flagged_plus_rel_tokens_masks',\

            'EntsAbst_flagged_tokens',\
            'EntsAbst_flagged_tokens_masks',\

            'EntsAbst_flagged_plus_rel_tokens',\
            'EntsAbst_flagged_plus_rel_tokens_masks',\

            'abstracted_ents_flagged_tokens',\
            'abstracted_ents_flagged_tokens_masks',\
            "abstracted_ents_flagged_plus_rel_tokens","abstracted_ents_flagged_plus_rel_tokens_masks"  ]



    L2=[\
    'y',\
    'item_id','y0','not_positive_not_negaative','subset','r','r1','similarity']
 
    Len_Target=[]
    abstract_Len_Target=[]
    ent_abstract_Len_Target=[]
    List_dic={}
    for k in examples[0].keys():
        List_dic[k]=[]
        k_mask=str(k)+'_masks'
        List_dic[k_mask]=[]

    for n, example in enumerate(tqdm(examples)):
      
        #############
        d={}

        for k in example.keys():
            if k not in L1 and k not in L2:
                d[k] = {'idx':np.zeros([Sentence_limit+1], dtype=np.int32),'segment':np.zeros([Sentence_limit+1], dtype=np.int32)}
            elif k in L1:
                d[k] = {'idx':np.zeros([ent_limit+1], dtype=np.int32),'segment':np.zeros([ent_limit+1], dtype=np.int32)}
            elif k in L2:
                d[k]=[] 
            if k =='sentence_flagged_tokens':
                t=example[k]
                if len(example[k])>Sentence_limit:
                    t=t[:Sentence_limit]
                else:
                    n=len(t)
                    m=Sentence_limit-n
                    temp=['[PAD]' for i in range(m)]
                    t=t+temp
        for k in example.keys():
            if k in L2:
                List_dic[k].append(example[k])
            else:
                sentence_tokens=example[k]
                sentence_idx=d[k]['idx']
                segments_ids=d[k]['segment']
                limit=Sentence_limit if k in L_s else ent_limit
                _idx,_segment_idx,n=store_idx_in_numpy(sentence_tokens,sentence_idx,segments_ids,limit)
                k_mask=str(k)+'_masks'
                List_dic[k].append(_idx)
                List_dic[k_mask].append(_segment_idx)
                if k=='ents_flagged_plus_rel_tokens':
                    tgt_len=n
                    Len_Target.append(tgt_len)
                elif k=='abstracted_ents_flagged_plus_rel_tokens':
                    abstract_tgt_len=n
                    abstract_Len_Target.append(abstract_tgt_len)
                elif k=='EntsAbst_flagged_plus_rel_tokens':
                    ent_abstract_tgt_len=n
                    ent_abstract_Len_Target.append(ent_abstract_tgt_len)



    if args.data not in ['none'] :

        np.savez(out_file,
        sentence_ents_flagged_tokens=np.array(List_dic['sentence_ents_flagged_tokens']),
        sentence_tokens=np.array(List_dic['sentence_tokens']),
        sentence_masked_flagged_tokens=np.array(List_dic['sentence_masked_flagged_tokens']),
        sentecne_ents_abstracted_flagged_tokens=np.array(List_dic['sentecne_ents_abstracted_flagged_tokens']),
        sentecne_entabs_flagged_tokens=np.array(List_dic['sentecne_entabs_flagged_tokens']),

        ents_flagged_tokens=np.array(List_dic['ents_flagged_tokens']),
        ents_flagged_plus_rel_tokens=np.array(List_dic['ents_flagged_plus_rel_tokens']),
        EntsAbst_flagged_tokens=np.array(List_dic['EntsAbst_flagged_tokens']),
        EntsAbst_flagged_plus_rel_tokens=np.array(List_dic['EntsAbst_flagged_plus_rel_tokens']),
        abstracted_ents_flagged_tokens=np.array(List_dic['abstracted_ents_flagged_tokens']),
        abstracted_ents_flagged_plus_rel_tokens=np.array(List_dic['abstracted_ents_flagged_plus_rel_tokens']),

        ###
        sentence_ents_flagged_tokens_masks=np.array(List_dic['sentence_ents_flagged_tokens_masks']),
        sentence_tokens_masks=np.array(List_dic['sentence_tokens_masks']),
        sentence_masked_flagged_tokens_masks=np.array(List_dic['sentence_masked_flagged_tokens_masks']),
        sentecne_ents_abstracted_flagged_tokens_masks=np.array(List_dic['sentecne_ents_abstracted_flagged_tokens_masks']),
        sentecne_entabs_flagged_tokens_masks=np.array(List_dic['sentecne_entabs_flagged_tokens_masks']),

        ents_flagged_tokens_masks=np.array(List_dic['ents_flagged_tokens_masks']),
        ents_flagged_plus_rel_tokens_masks=np.array(List_dic['ents_flagged_plus_rel_tokens_masks']),
        EntsAbst_flagged_tokens_masks=np.array(List_dic['EntsAbst_flagged_tokens_masks']),
        EntsAbst_flagged_plus_rel_tokens_masks=np.array(List_dic['EntsAbst_flagged_plus_rel_tokens_masks']),
        abstracted_ents_flagged_tokens_masks=np.array(List_dic['abstracted_ents_flagged_tokens_masks']),
        abstracted_ents_flagged_plus_rel_tokens_masks=np.array(List_dic['abstracted_ents_flagged_plus_rel_tokens_masks']),
        y=np.array(List_dic['y']),
        y0=np.array(List_dic['y0']),
        subset=np.array(List_dic['subset']),
        id=np.array(List_dic['item_id']),
        Len_Target=np.array(Len_Target),
        abstract_Len_Target=np.array(abstract_Len_Target),
        ent_abstract_Len_Target=np.array(ent_abstract_Len_Target)
        )


    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    print('out_file',out_file)
    return meta



def pre_process(data_name):

    args = get_setup_args(data_name)
    data_selected=args.data
    temp=['conll',]



    with open('essential_files/kbid_as_special_tokens.json') as f:
        kbid_as_special_tokens = json.load(f)
 
    if data_selected=='wikidata':
        train_file ="unprocessed_data/wikipedia_distant/distant_re_data_train.json"
        test_file ="unprocessed_data/wikipedia_distant/distant_re_data_test.json"
        b1={'name':'dev','path':test_file,'record':args.dev_record_file}
        eval_data=[b1]
    elif data_selected=='nyt':
        train_file ="unprocessed_data/NYT/dataset_triples_train.json"
        dev_file ="unprocessed_data/NYT/dataset_triples_test.json"
    elif data_selected=='tacred':
        train_file ="unprocessed_data//tacred/train.json"
        dev_file ="unprocessed_data//tacred/dev.json"
        test_file ="unprocessed_data//tacred/test.json"

        dev_rev_file ="unprocessed_data//tacred/dev_rev.json"
        test_rev_file ="unprocessed_data//tacred/test_rev.json"
        

        b1={'name':'dev','path':dev_file,'record':args.dev_record_file}
        b2={'name':'dev_rev','path':dev_rev_file,'record':args.dev_rev_record_file}
        b3={'name':'test','path':test_file,'record':args.test_record_file}
        b4={'name':'test_rev','path':test_rev_file,'record':args.test_rev_record_file}


        eval_data=[b1,b2,b3,b4]



    elif data_selected=='retacred':
        train_file ="unprocessed_data//retacred/train.json"
        dev_file ="unprocessed_data//retacred/dev.json"
        test_file ="unprocessed_data//retacred/test.json"

        

        b1={'name':'dev','path':dev_file,'record':args.dev_record_file}
        b2={'name':'test','path':test_file,'record':args.test_record_file}
        eval_data=[b1,b2]
    
    elif data_selected in temp:


        train_file ="unprocessed_data//fewrel/train.json"
        dev_file ="unprocessed_data//fewrel/dev.json"
        dev_file ="unprocessed_data//fewrel/dev.json"
        b1={'name':'dev','path':dev_file,'record':args.dev_record_file}
        eval_data=[b1]

    ####
    #args.tokenizer_name='presaved'
    #args.tokenizer_name='t5_large'
    #args.tokenizer_name='bert_base_uncased'
    #args.tokenizer_name='roberta-large'
    # args.tokenizer_name='roberta-base'
    re=True
    #args.tokenizer_name='bert-large-uncased'
    if args.tokenizer_name=='bert-base-uncased' or args.tokenizer_name=='bert-large-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.tokenizer_name=='bert-large-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    elif args.tokenizer_name=='roberta-large':
        config = { 'name': 'roberta-large', 'revision': '5069d8a', 'd_depth': 25, 'chunk_len': 512, }
        tokenizer = AutoTokenizer.from_pretrained(config['name'], revision=config['revision'])
        
    elif args.tokenizer_name=='roberta-base':
        config = { 'name': 'roberta-base',  'd_depth': 25, 'chunk_len': 512, }
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
      
    ##############################
    elif args.tokenizer_name=='t5-small':
        from transformers import T5Model
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")


    elif args.tokenizer_name=='t5-large':
        from transformers import T5Model
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")

    elif args.tokenizer_name=='gpt2':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    elif args.tokenizer_name=='opt':
        from transformers import OPTModel
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    elif args.tokenizer_name=='gpt':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
    elif args.tokenizer_name=='prophetnet':
        from transformers import ProphetNetEncoder
        tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")





    ####
    modality=args.tokenizer_name
    
    tokenizer_special_dic='semeval_2012_re' if data_selected=='semeval_2012' else 're' 
    if 'wordanalogy' in data_selected :
        tokenizer_special_dic='default'


    _, _,tokenizer =  _get_pretrained_transformer3(data_selected,modality,tokenizer_special_dic=tokenizer_special_dic) 







    train_file= train_file
    Relation_counter={}
    train_examples, eval_examples = process_file(tokenizer,train_file, "train",Relation_counter,data_selected)
    file='evaluation_data/'+str(data_selected)+'train_eval.json'
    eval_examples={'data':eval_examples}
    with open(file, "w") as outfile: 
        json.dump(eval_data, outfile)

    build_features(tokenizer,args, train_examples, "train", args.train_record_file )
    del train_examples

    for b in eval_data:
        name=b['name']
        record=b['record']
        path=b['path']
        dev_examples, eval_examples = process_file(tokenizer,path, name,Relation_counter,data_selected)

        file='evaluation_data/'+str(data_selected)+'dev_eval.json'
        eval_examples={'data':eval_examples}
        with open(file, "w") as outfile: 
            json.dump(eval_examples, outfile)
        dev_meta = build_features(tokenizer,args, dev_examples, "dev", record)


    file="essential_files/Relation_counter_"+str(data_selected)+".json"
  
    with open(file, "w") as outfile: 
        json.dump(Relation_counter, outfile)

