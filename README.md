# Re-Representation

Re-Representation in Sentential Relation Extraction with Sequence Routing Algorithm



<!-- 
| Model | Tacred | Tacredrev | Retacred | Conll04 | Wikidata |
|-------|--------|-----------|----------|---------|----------|
| **[Entity Marker ](https://aclanthology.org/2022.aacl-short.21/)** | 74.6 | 83.2 | 91.1 | – | – |
| **[Curriculum Learning ](https://arxiv.org/abs/2107.09332)** | 75.2 | – | **91.4** | – | – |
| **[REBEL](https://aclanthology.org/2021.findings-emnlp.204/)** | – | – | 90.4 | 76.5 | – |
| **[KGpool](https://aclanthology.org/2021.findings-acl.48/)** | – | – | – | – | **88.6** |
| **[RAG4RE](https://arxiv.org/abs/2404.13397)** | **86.6** | **88.3** | 73.3 | – | – |
| ours: <hr>|
| **bert H₃** | **84.8 (47.8)** | **85.3 (49.7)** | **89.4 (74.0)** | **99.7 (99.8)** | **84.5 (32.0)** |
| **bert H₁,H₂,H₃,Decoder** | **87.4 (48.3)** | **88.7 (50.9)** | **88.7 (68.5)** | **100. (100.)** | – |
| **RoBERTa H₃** | **87.1 (61.1)** | **88.8 (64.2)** | **92.2 (80.1)** | **100. (100.)** | **85.6 (32.9)** |

 -->

<table>
  <tr>
    <th>Model</th>
    <th>Tacred</th>
    <th>Tacredrev</th>
    <th>Retacred</th>
    <th>Conll04</th>
    <th>Wikidata</th>
  </tr>

  <tr><td colspan="6" align="center"><b>Baseline Models</b></td></tr>

  <tr><td><b>
    <a href="https://aclanthology.org/2022.aacl-short.21/">Entity Marker</a>
  </b></td><td>74.6</td><td>83.2</td><td>91.1</td><td>–</td><td>–</td></tr>
  <tr><td><b>
  <a href="https://arxiv.org/abs/2107.09332">Curriculum Learning</a>
</b></td><td>75.2</td><td>–</td><td><b>91.4</b></td><td>–</td><td>–</td></tr>
  <tr><td><b>
    
   
  <a href="https://aclanthology.org/2021.findings-emnlp.204/">REBEL</a>
</b></td><td>–</td><td>–</td><td>90.4</td><td>76.5</td><td>–</td></tr>
  <tr><td><b>
  <a href="https://aclanthology.org/2021.findings-acl.48//">KGpool</a>
</b></td><td>–</td><td>–</td><td>–</td><td>–</td><td><b>88.6</b></td></tr>
  <tr><td><b>
    
 
  <a href="https://arxiv.org/abs/2404.13397">RAG4RE</a>
</b></td><td><b>86.6</b></td><td><b>88.3</b></td><td>73.3</td><td>–</td><td>–</td></tr>

  <tr><td colspan="6" align="center"><b>Our Models</b></td></tr>

  <tr><td><b>Ours bert H₃</b></td><td><b>84.8 (47.8)</b></td><td><b>85.3 (49.7)</b></td><td><b>89.4 (74.0)</b></td><td><b>99.7 (99.8)</b></td><td><b>84.5 (32.0)</b></td></tr>
  <tr><td><b>Ours bert H₁,H₂,H₃,Decoder</b></td><td><b>87.4 (48.3)</b></td><td><b>88.7 (50.9)</b></td><td><b>88.7 (68.5)</b></td><td><b>100. (100.)</b></td><td>–</td></tr>
  <tr><td><b>Ours RoBERTa H₃</b></td><td><b>87.1 (61.1)</b></td><td><b>88.8 (64.2)</b></td><td><b>92.2 (80.1)</b></td><td><b>100. (100.)</b></td><td><b>85.6 (32.9)</b></td></tr>

</table>




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

+ [Wikidata](https://drive.google.com/file/d/1mmKLh6a78GVNizBoCGhs5ZMYJX2g-DIU/view?usp=sharing)
+ [conll04](https://huggingface.co/datasets/DFKI-SLT/conll04)
+ [tacred](https://catalog.ldc.upenn.edu/LDC2018T24)
Tacred Extensions: 
obtain retacred and tacredrev/tacrev by follwing the instruction in the provided links.
+ [retacred](https://arxiv.org/abs/2104.08398) and the [instrcution](https://github.com/gstoica27/Re-TACRED). 
+ [tacrev](https://arxiv.org/abs/2004.14855) and the  [instrcution](https://github.com/DFKI-NLP/tacrev). 
# Setup
```requirements
pip install git+https://github.com/glassroom/heinsen_routing
pip install transformers
pip3 install torch torchvision
pip installl tqdm
```

# Running
### Train and Eval

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

