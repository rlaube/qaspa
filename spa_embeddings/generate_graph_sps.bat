@echo off

@REM COMMENT OUT DATASETS YOU DO NOT WANT TO RUN
@REM REQUIRED INPUT FILES/Variables
@REM emb-dir: directory that knowledge graph (KG) entity (vocabulary) embeddings are contained
@REM concept-file, relation-file (.npy): within emb-dir, embeddings for every node and edge in the KG
@REM outdir: directory of output graph vectors (to be used by qaspa.py to train model)

call activate qaspa

@REM Create CSQA graph vectors
set dataset=csqa
set mc-options=5
@REM should be 'hrr', 'tvtb', or 'vtb'
set algebra=hrr
@REM should be Null or 0 (or another seed)
set permute-vector-seed=Null
@REM Use pruned version of graph (to 200 nodes). Uses graph_ids_pruned.npz instead of graph_ids.npz
set pruned=True
@REM normalize should be 'l2' or 'unitary' or 'None' depending on normalization method
set normalize-rel=l2
set normalize-concept=l2

set emb-dir="../data/cpnet/"
set concept-file="encoder_embs/bert-large-uncased_concept_emb.npy"
set relation-file="encoder_embs/bert-large-uncased_rel_emb.npy"
@REM Can also use random relation embeddings
@REM set relation-file="random_relation_emb.npy"
set outdir="../data/%dataset%/spa/bert-concept_bert-rel/l2-norm/pruned/%algebra%/"

@REM partitions=# of concurrent script runs
set split=train
set partitions=4
start "%split%" cmd /c run_split_graph_sp.bat

set split=dev
set partitions=1
start "%split%" cmd /c run_split_graph_sp.bat

set split=test
set partitions=1
start "%split%" cmd /c run_split_graph_sp.bat

@REM @REM Create OBQA graph vectors using CPNet KG
@REM set dataset=obqa
@REM set mc-options=4
@REM set algebra=hrr
@REM set permute-vector-seed=Null
@REM set pruned=True
@REM set normalize-rel=l2
@REM set normalize-concept=l2

@REM set emb-dir="../data/cpnet/"
@REM set concept-file="encoder_embs/bert-large-uncased_concept_emb.npy"
@REM set relation-file="encoder_embs/bert-large-uncased_rel_emb.npy"
@REM @REM Can also use random relation embeddings
@REM @REM set relation-file="random_relation_emb.npy"
@REM set outdir="../data/%dataset%/spa/bert-concept_bert-rel/l2-norm/pruned/%algebra%/"

@REM @REM partitions=# of concurrent script runs
@REM set split=train
@REM set partitions=4
@REM start "%split%" cmd /c run_split_graph_sp.bat

@REM set split=dev
@REM set partitions=1
@REM start "%split%" cmd /c run_split_graph_sp.bat

@REM set split=test
@REM set partitions=1
@REM start "%split%" cmd /c run_split_graph_sp.bat


@REM @REM Create MedQA graph vectors using UMLS KG
@REM set dataset=medqa
@REM set algebra=hrr
@REM set permute-vector-seed=Null
@REM set mc-options=4
@REM set pruned=True
@REM set normalize-rel=l2
@REM set normalize-concept=l2
@REM set emb-dir="../data/umls/"
@REM set concept-file="encoder_embs/biolinkbert_umls_concept_embs.npy"
@REM set relation-file="encoder_embs/biolinkbert_umls_relation_embs.npy"
@REM @REM Can also use random relation embeddings
@REM @REM set relation-file="random_relation_emb.npy"
@REM set outdir="../data/%dataset%/biolinkbert-concept_biolinkbert-rel/l2-norm/pruned/%algebra%/"

@REM set split=train
@REM set partitions=4
@REM start "%split%" cmd /c run_split_graph_sp.bat

@REM set split=dev
@REM set partitions=1
@REM start "%split%" cmd /c run_split_graph_sp.bat

@REM set split=test
@REM set partitions=1
@REM start "%split%" cmd /c run_split_graph_sp.bat

@REM call conda deactivate




@REM set emb-dir="cpnet/"
@REM set concept-file="tzw_concept_emb.npy"
@REM set relation-file="encoder_embs/bert-large-uncased_rel_emb.npy"
@REM set outdir="%dataset%/tzw/bert-rel/%algebra%/"

@REM @REM cpnet-ver is either pruned_cpnet, full_cpnet, or cpnet_with_loops
@REM set encoder=roberta-large
@REM set cpnet-ver=cpnet-with-loops
@REM @REM make sure concept and relation files match cpnet-ver
@REM set concept-file="encoder_embs/%encoder%_with_loops_concept_emb.npy"
@REM set relation-file="encoder_embs/%encoder%_with_loops_rel_emb.npy"
@REM set outdir="%dataset%/%encoder%/%cpnet_ver%/%algebra%/"