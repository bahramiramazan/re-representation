


import json
from Wikidata_Abstraction import *
from tqdm import tqdm
import random
###
with open('essential_files/properties-with-labels.json') as f:
    properties = json.load(f)
with open('essential_files/prop_wiki_all.json') as f:
    properties = json.load(f)


with open('essential_files/instances_heirarchy.json') as f:
    instances_heirarchy = json.load(f)
    instances_heirarchy=instances_heirarchy['h']
p_n_dic={'p':0,'neg':0}
unique_instances=set()
has_superclass_no_superclass_count={'has_no_superclass':0,'has_superclass':0}
def get_id_data(doc_record_id,d,data_dic,data_predictions,entities_context):
            tokens=d['tokens']
            sent=" ".join(tokens)
            edges=d['edges']

            temp=False
            for e in edges:
                right=list(e['right'])
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
                temp=True

            if temp==False:
            	return
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
            ##
            for v in vertexSet:
                name=v['lexicalInput']
                if name in ENTS_set:
                    continue
                text=name 
                kbID=v['kbID']
                ##
                e_tokens=name.split(' ')
         
                ENTS.append((text,e_tokens))
                ENTS_set.add(text)
            for ei,e in enumerate(edges):
                head=e['left']
                tail=e['right']
                kbID=e['kbID']
                head_id=vertexDic[tuple(head)]

                tail_id=vertexDic[tuple(tail)]
                #####
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

                #abstracted_head

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
                

                n1=int(head[0])#-1 if len(head)!=1 else int(head[0])
                n2=int(head[-1])+1 #if len(head)!=1 else int(head[-1])+1

                m1=int(tail[0])#if len(tail)!=1 else int(tail[0])

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
                ##
                place_holder_ner=19
                place_holder_pos='other'
                label_head_e=label_head.split(' ')
                label_tail_e=label_tail.split(' ')

                flags=[['[e11]'],['[e12]'],['[e21]'],['[e22]']]


                if first=='head':
                    EntsAbst_flagged_tokens=flags[0]+label_head_e+['*']+[abstracted_head]+['#']+flags[1]+flags[2]+label_tail_e+['@']+[abstracted_tail]+['&']+flags[3]
                    EntsAbst_flagged_tokens=ents_flagged_tokens+['['+str(kbID).lower()+']']
                    ###
                    ents_flagged_tokens=flags[0]+label_head_e+flags[1]+flags[2]+label_tail_e+flags[3]
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
                #all_ent_in_sent_pos=[]
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
                    if item_id in data_predictions.keys():
                    	item['predictions']=data_predictions[item_id]
                    	data_dic[item_id]=item

def correct_predictions():
	#####
	data_dic={}
	file='predictions_not_corect_wikidata0.json'
	with open(file) as f:
	    data_predictions = json.load(f)#['evaluation_data/predictions_not_corect']
	entities_context = json.load(open('unprocessed_data/wikipedia_distant//entities_context.json'))
	test_file ="unprocessed_data/wikipedia_distant/distant_re_data_test.json"
	with open(test_file) as data_file:
		data_records = json.load(data_file)

	for doc_record_id,indx in enumerate(tqdm(range(len(data_records)))):
		indx=str(indx)
		d=data_records[indx]
		get_id_data(doc_record_id,d,data_dic,data_predictions,entities_context)
	print('data_dic',len(data_dic.keys()),len(data_predictions.keys()))
	categorized_data={}
	for k in data_dic.keys():

		item=data_dic[k]
		predictions=item['predictions']
		ents_rel=item['abstracted_ents_flagged_plus_rel_tokens']
		sentence_ents_flagged_tokens=item['sentence_ents_flagged_tokens']
		d=predictions
		d['ents']=ents_rel
		d['sen']=' '.join(sentence_ents_flagged_tokens)
		ground_truth_p=d['ground_truth_p']
		predicted_truth_p=d['predicted_truth_p']

		new_key=ground_truth_p+'-'+predicted_truth_p

		if new_key in categorized_data.keys():
			categorized_data[new_key].append(d)

		else:
			categorized_data[new_key]=[]
			categorized_data[new_key].append(d)
	count=0
	total=0
	categorized_data_l_=[(k,len(categorized_data[k])) for k in categorized_data.keys()]
	categorized_data_l_2={}
	for t in categorized_data_l_:
		k,l=t[0],t[1]
		categorized_data_l_2[k]=l
	categorized_data_l_2=sorted(categorized_data_l_2.items(), key=lambda x: x[1], reverse=True)
	
	categorized_data_l_2
	j=0
	total=0
	count=0

	for k,t in categorized_data_l_2:

		temp=False

		if 'P0'   in k.split('-')[0]:
			#pass
			temp=True
			#continue
		#print(k)
		if temp==False:
			continue

		data=categorized_data[k]
		if len(data)<1:
			continue

		count=count+1
		total=total+len(data)


		print('*****************************###########################',len(data))
		L=[i for i in range(len(data))]
		import random 
		kl=10 if len(L)>10 else 1
		L=random.sample(population=L, k=kl)
		for di,d in enumerate(data):
			if di not in L:
				continue

			print('k',k)
			for key in d.keys():

				print('key',key,d[key])
			print('*****************************')
	print('count',count)
	print('total',total)




if __name__ == "__main__":
	correct_predictions()