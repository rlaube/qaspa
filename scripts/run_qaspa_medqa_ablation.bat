@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "datestamp=%YYYY%%MM%%DD%" & set "timestamp=%HH%%Min%%Sec%"
set "dt=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"
echo %dt%

set CUDA_VISIBLE_DEVICES=0


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

set dataset=medqa
set model=michiyasunaga/BioLinkBERT-large
  
set algebra=hrr
set k=3
set dlr=0.00018225539664628204
set elr=0.00002074328673987209
set bs=32
set mbs=2
set ebs=8
set unfreeze_epoch=5
set refreeze_epoch=25
set n_epochs=30
set max_epochs_before_stop=8
set lr_schedule=warmup_cosine_restarts
set warmup_steps=50
set cycles=1

set dropoutf=0.3
set dropoutspa=0.7
set qa_context=False
set sent_trans=False
set normalize_graphs=False
set sp_layer_norm=True
set normalize_embeddings=True
set max_seq_len=512
set skip_type=0
set skip_placement=0

set fp16=True
set seed=0
set encoder_only=False

set study_name=qavsa_ablations_2024-10-13_15-52-52
set save_model=1

@REM timeout /t 3600 /nobreak

@REM set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/made-unitary/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/umls/encoder_embs/umls_concept_embs_v2.npy

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/umls/encoder_embs/umls_concept_embs_v2.npy

@REM set model=bert-large-cased
@REM set run_name=qavsa_%model%_BioLink-emb
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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

@REM set model=roberta-large
@REM set run_name=qavsa_%model%_BioLink-emb
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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
@REM set model=michiyasunaga/BioLinkBERT-large

set sp_dir=../SPA-Embeddings/%dataset%/bert-large-concept_bert-large-rel_v2/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/umls/encoder_embs/bert-large-uncased_umls_concept_embs.npy

@REM set model=bert-large-cased
@REM set run_name=qavsa_%model%_bert-large-emb
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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

@REM set model=roberta-large
@REM set run_name=qavsa_%model%_bert-large-emb
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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

set run_name=qavsa_BioLinkBert_bert-large-emb
python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
  --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 1 ^
  --skip_type %skip_type% --skip_placement %skip_placement% ^
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

@REM set algebra=vtb
@REM set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/l2-norm/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/umls/encoder_embs/umls_concept_embs_v2.npy

@REM set normalize_graphs=True
@REM set run_name=qavsa_l2-norm_graph-norm_%algebra%
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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
@REM set algebra=hrr
@REM set normalize_graphs=False

@REM set skip_type=1
@REM set normalize_graphs=True
@REM set run_name=qavsa_l2-norm_skip%skip_type%_norm-graphs
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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
@REM set normalize_graphs=False

@REM set skip_type=2
@REM set run_name=qavsa_l2-norm_skip%skip_type%
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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

@REM set skip_type=3
@REM set run_name=qavsa_l2-norm_skip%skip_type%
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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


@REM set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/no-norm/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/umls/encoder_embs/umls_concept_embs_v2.npy
@REM set normalize_graphs=True
@REM set run_name=qavsa_graph-norm_concept-not-norm
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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
@REM set normalize_graphs=False

@REM set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/l2-norm/pruned/perm0_obj/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy
@REM set run_name=qavsa_perm-obj_seed0
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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

@REM @REM SAME EMBEDDINGS, DIFFERENT MODEL PARAMS

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/umls/encoder_embs/umls_concept_embs_v2.npy

@REM set normalize_graphs=True
@REM set run_name=qavsa_graph-norm
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM   --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 1 ^
@REM   --skip_type %skip_type% --skip_placement %skip_placement% ^
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
@REM set normalize_graphs=False

set qa_context=True
set normalize_graphs=True
set run_name=qavsa_qa-context_graph-norm
python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
  --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 1 ^
  --skip_type %skip_type% --skip_placement %skip_placement% ^
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
set qa_context=False
set normalize_graphs=False

set qa_context=True
set run_name=qavsa_qa-context
python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
  --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 1 ^
  --skip_type %skip_type% --skip_placement %skip_placement% ^
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
set qa_context=False


call conda deactivate