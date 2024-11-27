SETLOCAL ENABLEDELAYEDEXPANSION
@echo off

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "datestamp=%YYYY%%MM%%DD%" & set "timestamp=%HH%%Min%%Sec%"
set "dt=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"
echo %dt%

set CUDA_VISIBLE_DEVICES=0
set dataset="csqa"


echo "***** hyperparameters *****"
echo "dataset: %dataset%"
echo "enc_name: %model%"
echo "batch_size: %bs%"
echo "learning_rate: elr %elr% dlr %dlr%"
echo "fc sp: layer %k%"
echo "******************************"
   
@REM save_model=0 doesn't save, save_model=1 saves on best dev acc, save_model=2 saves every epoch
set wandb=True
set save_model=0
set save_dir_pref=runs
mkdir %save_dir_pref%

call conda activate qaspa

@REM paper params (qavsa-seed-runs_2024-05-13_14-17-34)
set k=5
set elr=0.00001766729212078751
set dlr=0.03713776900239807
set bs=64
set mbs=8
set dropoutf=0.1
set dropoutspa=0.2
set n_epochs=15
set max_epochs_before_stop=6
set warmup_steps=200
set unfreeze_epoch=3
set lr_schedule=warmup_cosine_restarts
set cycles=1
set algebra=hrr
set fp16=True
set normalize_graphs=False
set sent_trans=False
set qa_context=False
set sp_layer_norm=True
set normalize_embeddings=True
set encoder_only=False
set seed=0


@REM @REM from group: qavsa_full-skip-test_2024-06-22_19-55-57
set skip_type=0
set skip_place=0

@REM tests: context enc = {BERT, Roberta}, concept emb = {BERT, Roberta}, norm = {l2, unitary}
set study_name=qavsa_PLM_normalization_tests_2024-10-04_14-35-13

set model="bert-large-uncased"
set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy
set run_name=qavsa_bert-QA_bert-enc_l2-norm

python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
  --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 1 ^
  --skip_type %skip_type% --skip_placement %skip_place% ^
  --hyperparameter_tuning False ^
  --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
  --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
  --train_adj data/%dataset%/graph/train.graph.adj.pk ^
  --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
  --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
  --train_sp %sp_dir%train_graph_sp.npy ^
  --dev_sp   %sp_dir%dev_graph_sp.npy ^
  --test_sp  %sp_dir%test_graph_sp.npy ^
  --cpnet_emb_path %emb_dir% ^
  --train_statements  data/%dataset%/statement/train.statement.jsonl ^
  --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
  --test_statements  data/%dataset%/statement/test.statement.jsonl ^
  --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
  --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

set model="bert-large-uncased"
set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy
set run_name=qavsa_bert-QA_bert-enc_unitary

python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
  --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 1 ^
  --skip_type %skip_type% --skip_placement %skip_place% ^
  --hyperparameter_tuning False ^
  --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
  --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
  --train_adj data/%dataset%/graph/train.graph.adj.pk ^
  --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
  --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
  --train_sp %sp_dir%train_graph_sp.npy ^
  --dev_sp   %sp_dir%dev_graph_sp.npy ^
  --test_sp  %sp_dir%test_graph_sp.npy ^
  --cpnet_emb_path %emb_dir% ^
  --train_statements  data/%dataset%/statement/train.statement.jsonl ^
  --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
  --test_statements  data/%dataset%/statement/test.statement.jsonl ^
  --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
  --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

set model="bert-large-uncased"
set sp_dir=../SPA-Embeddings/%dataset%/roberta-concept_roberta-rel/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/roberta-large_concept_emb_v2.npy
set run_name=qavsa_bert-QA_roberta-enc_l2-norm

python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
  --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 1 ^
  --skip_type %skip_type% --skip_placement %skip_place% ^
  --hyperparameter_tuning False ^
  --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
  --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
  --train_adj data/%dataset%/graph/train.graph.adj.pk ^
  --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
  --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
  --train_sp %sp_dir%train_graph_sp.npy ^
  --dev_sp   %sp_dir%dev_graph_sp.npy ^
  --test_sp  %sp_dir%test_graph_sp.npy ^
  --cpnet_emb_path %emb_dir% ^
  --train_statements  data/%dataset%/statement/train.statement.jsonl ^
  --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
  --test_statements  data/%dataset%/statement/test.statement.jsonl ^
  --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
  --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

set model="bert-large-uncased"
set sp_dir=../SPA-Embeddings/%dataset%/roberta-concept_roberta-rel/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/roberta-large_concept_emb_v2.npy
set run_name=qavsa_bert-QA_roberta-enc_unitary

python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
  --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 1 ^
  --skip_type %skip_type% --skip_placement %skip_place% ^
  --hyperparameter_tuning False ^
  --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
  --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
  --train_adj data/%dataset%/graph/train.graph.adj.pk ^
  --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
  --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
  --train_sp %sp_dir%train_graph_sp.npy ^
  --dev_sp   %sp_dir%dev_graph_sp.npy ^
  --test_sp  %sp_dir%test_graph_sp.npy ^
  --cpnet_emb_path %emb_dir% ^
  --train_statements  data/%dataset%/statement/train.statement.jsonl ^
  --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
  --test_statements  data/%dataset%/statement/test.statement.jsonl ^
  --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
  --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

@REM set model="roberta-large"
@REM set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel/made-unitary/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy
@REM set run_name=qavsa_roberta-QA_bert-emb_unitary

@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_place% ^
@REM   --hyperparameter_tuning False ^
@REM   --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
@REM   --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
@REM   --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM   --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM   --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM   --train_sp %sp_dir%train_graph_sp.npy ^
@REM   --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM   --test_sp  %sp_dir%test_graph_sp.npy ^
@REM   --cpnet_emb_path %emb_dir% ^
@REM   --train_statements  data/%dataset%/statement/train.statement.jsonl ^
@REM   --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
@REM   --test_statements  data/%dataset%/statement/test.statement.jsonl ^
@REM   --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
@REM   --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

@REM set model="roberta-large"
@REM set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel/l2-norm/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy
@REM set run_name=qavsa_roberta-QA_bert-emb_l2-norm

@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_place% ^
@REM   --hyperparameter_tuning False ^
@REM   --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
@REM   --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
@REM   --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM   --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM   --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM   --train_sp %sp_dir%train_graph_sp.npy ^
@REM   --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM   --test_sp  %sp_dir%test_graph_sp.npy ^
@REM   --cpnet_emb_path %emb_dir% ^
@REM   --train_statements  data/%dataset%/statement/train.statement.jsonl ^
@REM   --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
@REM   --test_statements  data/%dataset%/statement/test.statement.jsonl ^
@REM   --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
@REM   --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

@REM set model="roberta-large"
@REM set sp_dir=../SPA-Embeddings/%dataset%/roberta-concept_roberta-rel/l2-norm/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/roberta-large_concept_emb_v2.npy
@REM set run_name=qavsa_roberta-QA_roberta-emb_l2-norm

@REM set save_model=1

@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_place% ^
@REM   --hyperparameter_tuning False ^
@REM   --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
@REM   --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
@REM   --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM   --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM   --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM   --train_sp %sp_dir%train_graph_sp.npy ^
@REM   --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM   --test_sp  %sp_dir%test_graph_sp.npy ^
@REM   --cpnet_emb_path %emb_dir% ^
@REM   --train_statements  data/%dataset%/statement/train.statement.jsonl ^
@REM   --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
@REM   --test_statements  data/%dataset%/statement/test.statement.jsonl ^
@REM   --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
@REM   --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%
  
call conda deactivate