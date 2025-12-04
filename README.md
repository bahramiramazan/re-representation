# Re-Representation

Re-Representation in Sentential Relation Extraction with Sequence Routing Algorithm



# Citation
If you use our work kindly consider citing the paper  [Re-Representation](https://aclanthology.org/2025.icnlsp-1.31/)

```
@inproceedings{bahrami-yahyapour-2025-representation,
    title = "Re-Representation in Sentential Relation Extraction with Sequence Routing Algorithm",
    author = "Bahrami, Ramazan  and
      Yahyapour, Ramin",
    editor = "Abbas, Mourad  and
      Yousef, Tariq  and
      Galke, Lukas",
    booktitle = "Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP-2025)",
    month = aug,
    year = "2025",
    address = "Southern Denmark University, Odense, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.icnlsp-1.31/",
    pages = "315--327"
}
```

# Dataset
Download the datasets and place it in their respective folders at unprocessed_data folder.
The datasets can be downloaded from the below link

[Wikidata](https://drive.google.com/file/d/1mmKLh6a78GVNizBoCGhs5ZMYJX2g-DIU/view?usp=sharing)

[conll04](https://huggingface.co/datasets/DFKI-SLT/conll04)

[tacred](https://catalog.ldc.upenn.edu/LDC2018T24)

obtain retaced by applying the patch from the the following link:
[retacred](https://github.com/gstoica27/Re-TACRED)



# Setup
```requirements
pip install git+https://github.com/glassroom/heinsen_routing
pip install transformers
pip3 install torch torchvision
pip installl tqdm
```

# Running
### Train

The parameters are in the config file. 
Download the datasets, embeddings etc. and place in the expected locations.

> preprocess the data: \

```
 python main.py  --task preprocess  --data dataname --tokenizer_name modelname
```
dataname= conll, wikidata, tacred, retacred\
tokenizer_name= bert-large-uncased, roberta-large\

Example: To preprocess conll with roberta-large\
```
 python main.py  --task preprocess  --data wikidata --tokenizer_name roberta-large
```

> to train for an experiment   \
experimentNo=one, two, three, four, five according to the paper
```
 python main.py  --task train  --data wikidata  --experiment experimentNo  --model_to_train wordanalogy_re_model --tokenizer_name roberta-large
```
> to eval  \
```
 python main.py  --task eval  --data wikidata  --experiment experimentNo  --model_to_train wordanalogy_re_model --tokenizer_name roberta-large
```

