# QASPA: Question Answering Using Vector Symbolic Algebras

This repo provides the source code & data of my paper: [QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering](https://arxiv.org/abs/2104.06378) (NAACL 2021).
```bib
@inproceedings{laube-eliasmith-2024-qavsa,
    title = "{QAVSA}: Question Answering using Vector Symbolic Algebras",
    author = "Laube, Ryan  and
      Eliasmith, Chris",
    editor = "Zhao, Chen  and
      Mosbach, Marius  and
      Atanasova, Pepa  and
      Goldfarb-Tarrent, Seraphina  and
      Hase, Peter  and
      Hosseini, Arian  and
      Elbayad, Maha  and
      Pezzelle, Sandro  and
      Mozes, Maximilian",
    booktitle = "Proceedings of the 9th Workshop on Representation Learning for NLP (RepL4NLP-2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.repl4nlp-1.14",
    pages = "191--202",
}
```


## Usage
### 0. Dependencies
Run the following commands to create a conda environment (assuming CUDA11.8):
```bash
conda create -n qaspa python=3.11.7
conda activate qaspa
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.37.0
pip install nltk spacy==2.1.6
python -m spacy download en

conda install nengo nengo-spa
conda install optuna
```

### 1. Download data
For question answering datasets CommonsenseQA, OpenBookQA and the ConceptNet knowledge graph, follow preprocessing described in https://github.com/michiyasunaga/qagnn.

To skip preprocessing, download preprocessed CSQA and OBQA datasets and knowledge graphs by
```
./download_preprocessed_data.sh
```

You can download all the preprocessed data from [here](https://nlp.stanford.edu/projects/myasu/DRAGON/data_preprocessed.zip). This includes the UMLS biomedical knowledge graph and MedQA dataset.

The resulting file structure will look like:

```plain
.
├── README.md
├── data/
    ├── cpnet/                 (prerocessed ConceptNet)
    ├── csqa/
        ├── train_rand_split.jsonl
        ├── dev_rand_split.jsonl
        ├── test_rand_split_no_answers.jsonl
        ├── statement/             (converted statements)
        ├── grounded/              (grounded entities)
        ├── graphs/                (extracted subgraphs)
        ├── ...
    ├── obqa/
    ├── medqa_usmle/
    └── ddb/
```

### 1. Generate graph node/relation embeddings and total graph embeddings


### 2. Hyperparameter Tune QASPA
Hyperparameter tuning is done with Optuna and experiment tracking is done with Weights & Biases (W&B). Follow the W&B quickstart to create an account and log in: https://docs.wandb.ai/quickstart/.

Run 

### 3. Train QASPA
As configured in these scripts, the model needs three types of input files
* `--{train,dev,test}_statements`: preprocessed question statements in jsonl format. This is mainly loaded by `load_input_tensors` function in `utils/data_utils.py`.
* `--{train,dev,test}_adj`: information of the KG subgraph extracted for each question. This is mainly loaded by `load_sparse_adj_data_with_contextnode` function in `utils/data_utils.py`.
* `--{train,dev,test}_sp`: VSA KG subgraph vector (.npy). this is loaded by `load_graph_sp` function in `utils/data_utils.py`.

To train model (for 5 seeds), run

```
run_qaspa_{csqa,obqa,medqa}_seed_runs.bat
```


**Note**: The models were trained and tested with HuggingFace transformers==4.37.0.


## Use your own dataset
- Convert your dataset to  `{train,dev,test}.statement.jsonl` in .jsonl format (see `data/csqa/statement/train.statement.jsonl`)
- Create a directory in `data/{yourdataset}/` to store the .jsonl files
- Modify `preprocess.py` and perform subgraph extraction for your data
- Modify `utils/parser_utils.py` to support your own dataset


## Acknowledgment
This repo is built upon the following work:
```
QA-GNN: Question Answering using Language Models and Knowledge Graphs. Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, Jure Leskovec. NAACL 2021
https://github.com/michiyasunaga/qagnn
```
