SETLOCAL ENABLEDELAYEDEXPANSION
@echo off

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "datestamp=%YYYY%%MM%%DD%" & set "timestamp=%HH%%Min%%Sec%"
set "dt=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"
echo %dt%

set CUDA_VISIBLE_DEVICES=0
set dataset=obqa
set model=roberta-large

@REM from group qasper_hyperparam-tuning_2024-05-05_17-58-00
set elr=0.00004175673865853067
set dlr=0.034127679167615325
set bs=64
set mbs=16
set dropoutf=0
set dropoutspa=0.4
set unfreeze_epoch=3
set n_epochs=15
set max_epochs_before_stop=6
set k=4
set lr_schedule=warmup_cosine_restarts
set warmup_steps=200

set cycles=2
set fp16=True
set algebra=hrr
set sent_trans=False
set qa_context=False
set normalize_graphs=False
set sp_layer_norm=True
set normalize_embeddings=True
set skip_placement=0
set skip_type=0

set seed=0

set score_mlp=False

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

@REM timeout /t 7200 /nobreak

set study_name=qavsa-ablations_2024-10-11_15-52-33

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy

set model=bert-large-uncased
set run_name=qavsa_%model%-QA_bert-emb
call python -u qaspa.py --dataset %dataset% ^
    --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
    --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
    --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
    --skip_placement %skip_placement% --skip_type %skip_type% --score_mlp %score_mlp% ^
    --hyperparameter_tuning False ^
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
    --log_interval 1 ^
    --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

set sp_dir=../SPA-Embeddings/%dataset%/roberta-concept_roberta-rel_v2/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/roberta-large_concept_emb_v2.npy

set model=roberta-large
set run_name=qavsa_%model%-QA_roberta-emb
call python -u qaspa.py --dataset %dataset% ^
    --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
    --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
    --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
    --skip_placement %skip_placement% --skip_type %skip_type% --score_mlp %score_mlp% ^
    --hyperparameter_tuning False ^
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
    --log_interval 1 ^
    --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

set model=bert-large-uncased
set run_name=qavsa_%model%-QA_roberta-emb
call python -u qaspa.py --dataset %dataset% ^
    --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
    --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
    --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
    --skip_placement %skip_placement% --skip_type %skip_type% --score_mlp %score_mlp% ^
    --hyperparameter_tuning False ^
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
    --log_interval 1 ^
    --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

@REM set skip_type=1
@REM set run_name=qavsa_mlp-score_skip%skip_type%

@REM call python -u qaspa.py --dataset %dataset% ^
@REM     --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
@REM     --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM     --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
@REM     --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
@REM     --skip_placement %skip_placement% --skip_type %skip_type% --score_mlp %score_mlp% ^
@REM     --hyperparameter_tuning False ^
@REM     --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM     --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM     --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM     --train_sp %sp_dir%train_graph_sp.npy ^
@REM     --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM     --test_sp  %sp_dir%test_graph_sp.npy ^
@REM     --cpnet_emb_path %emb_dir% ^
@REM     --train_statements  data/%dataset%/statement/train.statement.jsonl ^
@REM     --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
@REM     --test_statements  data/%dataset%/statement/test.statement.jsonl ^
@REM     --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
@REM     --log_interval 1 ^
@REM     --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

@REM set skip_type=2
@REM set run_name=qavsa_mlp-score_skip%skip_type%

@REM call python -u qaspa.py --dataset %dataset% ^
@REM     --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
@REM     --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM     --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
@REM     --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
@REM     --skip_placement %skip_placement% --skip_type %skip_type% --score_mlp %score_mlp%  ^
@REM     --hyperparameter_tuning False ^
@REM     --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM     --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM     --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM     --train_sp %sp_dir%train_graph_sp.npy ^
@REM     --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM     --test_sp  %sp_dir%test_graph_sp.npy ^
@REM     --cpnet_emb_path %emb_dir% ^
@REM     --train_statements  data/%dataset%/statement/train.statement.jsonl ^
@REM     --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
@REM     --test_statements  data/%dataset%/statement/test.statement.jsonl ^
@REM     --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
@REM     --log_interval 1 ^
@REM     --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

@REM set skip_type=3
@REM set run_name=qavsa_mlp-score_skip%skip_type%

@REM call python -u qaspa.py --dataset %dataset% ^
@REM     --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
@REM     --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM     --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
@REM     --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
@REM     --skip_placement %skip_placement% --skip_type %skip_type% --score_mlp %score_mlp%  ^
@REM     --hyperparameter_tuning False ^
@REM     --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM     --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM     --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM     --train_sp %sp_dir%train_graph_sp.npy ^
@REM     --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM     --test_sp  %sp_dir%test_graph_sp.npy ^
@REM     --cpnet_emb_path %emb_dir% ^
@REM     --train_statements  data/%dataset%/statement/train.statement.jsonl ^
@REM     --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
@REM     --test_statements  data/%dataset%/statement/test.statement.jsonl ^
@REM     --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
@REM     --log_interval 1 ^
@REM     --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

  
call conda deactivate