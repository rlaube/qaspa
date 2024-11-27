@echo off

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "datestamp=%YYYY%%MM%%DD%" & set "timestamp=%HH%%Min%%Sec%"
set "dt=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"
echo %dt%

set CUDA_VISIBLE_DEVICES=0
set dataset=csqa
set model=roberta-large

@REM from group no_qa-context_tuning_2024-04-13_15-37-21 (k4_dlr5.04e-05_unfreeze0_dropsp0.6_dropf0.7_warmup_cosine_restarts_steps-200), dev acc: 0.7797, test_acc: 0.7204
@REM set elr=1e-5
@REM set bs=64
@REM set mbs=8
@REM set warmup_steps=200
@REM set n_epochs=15
@REM set max_epochs_before_stop=7
@REM set unfreeze_epoch=100

@REM set fp16=True
@REM set algebra=hrr
@REM set normalize_embeddings=True
@REM set sent_trans=False
@REM set normalize_graphs=True
@REM set sp_layer_norm=True

set elr=1e-5
set bs=64
set mbs=8
set unfreeze_epoch=100
set n_epochs=15
set max_epochs_before_stop=6
set warmup_steps=200
set cycles=1
set algebra=hrr
set fp16=True
set normalize_graphs=False
set sent_trans=False
set qa_context=False
set sp_layer_norm=True
set normalize_embeddings=True


set seed=0

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

set n_trials=300
set study_name=qavsa_frozen-lm_residual_tuning_%dt%

@REM set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel/made-unitary/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy
set sp_dir=../SPA-Embeddings/%dataset%/tzw-concept_bert-rel/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/tzw_concept_emb.npy

call python -u qaspa.py --dataset %dataset% --qa_context %qa_context% ^
    --algebra %algebra% -elr %elr% --encoder %model% -bs %bs% -mbs %mbs% ^
    --fp16 %fp16% --seed %seed% --warmup_steps %warmup_steps% --unfreeze_epoch %unfreeze_epoch% --cycles %cycles% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% ^
    --sent_trans %sent_trans% --normalize_embeddings %normalize_embeddings% --sp_layer_norm %sp_layer_norm% --normalize_graphs %normalize_graphs% ^
    --hyperparameter_tuning True --n_trials %n_trials% ^
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
    --study_name %study_name% --use_wandb %wandb% ^
    --log_interval 1 ^
    --save_dir %save_dir_pref%/%dataset%/%study_name%/%run_name% --save_model %save_model%
  
call conda deactivate