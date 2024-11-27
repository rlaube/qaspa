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

@REM from group no_qa-context_tuning_2024-04-13_15-37-21 (k4_dlr5.04e-05_unfreeze0_dropsp0.6_dropf0.7_warmup_cosine_restarts_steps-200), dev acc: 0.7797, test_acc: 0.7204
@REM set elr=1e-05
@REM set dlr=0.000050379266503616
@REM set bs=64
@REM set mbs=8
@REM set dropoutf=0.7
@REM set dropoutspa=0.6
@REM set unfreeze_epoch=0
@REM set n_epochs=15
@REM set max_epochs_before_stop=6
@REM set k=4
@REM set lr_schedule=warmup_cosine_restarts
@REM set warmup_steps=200

@REM set cycles=2
@REM set fp16=True
@REM set algebra=hrr
@REM set sent_trans=False
@REM set qa_context=False
@REM set normalize_graphs=False
@REM set sp_layer_norm=False
@REM set normalize_embeddings=True
@REM set seed=0

@REM from group qasper_hyperparam-tuning_2024-05-05_17-58-00
set elr=0.00004175673865853067
set dlr=0.034127679167615325
set bs=64
set mbs=4
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

@REM timeout /t 10800 /nobreak

set study_name=qavsa-seed-100s-runs_%dt%

@REM set sp_dir=../SPA-Embeddings/%dataset%/tzw-concept_bert-rel/made-unitary/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/tzw_concept_emb.npy
set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy

set save_model=0
set seed=0
set run_name=bert-concept_bert-rel_unitary_seed%seed%
call python -u qaspa.py --dataset %dataset% ^
    --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
    --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
    --hyperparameter_tuning False ^
    --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
    --train_adj data/%dataset%/graph/train.graph.adj.pk ^
    --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
    --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
    --train_sp %sp_dir%train_graph_sp.npy ^
    --dev_sp   %sp_dir%dev_graph_sp.npy ^
    --test_sp  %sp_dir%test_graph_sp.npy ^
    --cpnet_emb_path %emb_dir% ^
    --train_statements  data/%dataset%/statement/train-fact.statement.jsonl ^
    --dev_statements  data/%dataset%/statement/dev-fact.statement.jsonl ^
    --test_statements  data/%dataset%/statement/test-fact.statement.jsonl ^
    --run_name %run_name% --study_name %study_name% --use_wandb %wandb% ^
    --log_interval 1 ^
    --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%

@REM set save_model=0
@REM for /l %%s in (100, 100, 400) do (
@REM     set run_name=bert-concept_bert-rel_unitary_seed%%s
@REM     call python -u qaspa.py --dataset %dataset% ^
@REM         --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
@REM         --fp16 %fp16% --seed %%s --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM         --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
@REM         --hyperparameter_tuning False ^
@REM         --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
@REM         --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM         --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM         --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM         --train_sp %sp_dir%train_graph_sp.npy ^
@REM         --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM         --test_sp  %sp_dir%test_graph_sp.npy ^
@REM         --cpnet_emb_path %emb_dir% ^
@REM         --train_statements  data/%dataset%/statement/train-fact.statement.jsonl ^
@REM         --dev_statements  data/%dataset%/statement/dev-fact.statement.jsonl ^
@REM         --test_statements  data/%dataset%/statement/test-fact.statement.jsonl ^
@REM         --run_name !run_name! --study_name %study_name% --use_wandb %wandb% ^
@REM         --log_interval 1 ^
@REM         --save_dir %save_dir_pref%/%dataset%/%study_name%/!run_name! --save_model %save_model%
@REM )

@REM set save_model=0
@REM set study_name=qavsa-lm_only-seed100s-tests_%dt%
@REM set encoder_only=True

@REM for /l %%s in (0, 100, 400) do (
@REM     set run_name=lm-only_seed%%s
@REM     call python -u qaspa.py --dataset %dataset% --encoder_only %encoder_only% ^
@REM         --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
@REM         --fp16 %fp16% --seed %%s --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM         --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch%  ^
@REM         --hyperparameter_tuning False ^
@REM         --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --normalize_embeddings %normalize_embeddings% --normalize_graphs %normalize_graphs% ^
@REM         --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM         --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM         --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM         --train_sp %sp_dir%train_graph_sp.npy ^
@REM         --dev_sp   %sp_dir%dev_graph_sp.npy ^
@REM         --test_sp  %sp_dir%test_graph_sp.npy ^
@REM         --cpnet_emb_path %emb_dir% ^
@REM         --train_statements  data/%dataset%/statement/train-fact.statement.jsonl ^
@REM         --dev_statements  data/%dataset%/statement/dev-fact.statement.jsonl ^
@REM         --test_statements  data/%dataset%/statement/test-fact.statement.jsonl ^
@REM         --run_name !run_name! --study_name %study_name% --use_wandb %wandb% ^
@REM         --log_interval 1 ^
@REM         --save_dir %save_dir_pref%/%dataset%/%study_name%/!run_name! --save_model %save_model%
@REM )
  
call conda deactivate