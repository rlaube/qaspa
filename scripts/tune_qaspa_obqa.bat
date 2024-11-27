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

@REM set elr=1e-5
@REM set bs=128
@REM set mbs=2
@REM set n_epochs=30
@REM set max_epochs_before_stop=10
@REM set unfreeze_epoch=31

@REM set fp16=True
@REM set algebra=hrr
@REM set normalize_embeddings=True
@REM set seed=0

@REM paper params (qavsa-seed-runs_2024-05-13_14-17-34)
set k=5
set elr=0.00001766729212078751
set bs=64
set mbs=8
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

@REM no skip connections
set skip_place=0

echo "***** hyperparameters *****"
echo "dataset: %dataset%"
echo "enc_name: %model%"
echo "batch_size: %bs%"
echo "learning_rate: elr %elr%"
echo "******************************"
   
@REM save_model=0 doesn't save, save_model=1 saves on best dev acc, save_model=2 saves every epoch
set wandb=True
set save_model=0
set save_dir_pref=runs
mkdir %save_dir_pref%

@REM timeout /t 8700 /nobreak

call conda activate qaspa

set n_trials=400
set study_name=qavsa_bert-emb_skip_tuning_%dt%

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy

@REM set sp_dir=../SPA-Embeddings/%dataset%/tzw-concept_bert-rel/made-unitary/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/tzw_concept_emb.npy

call python -u qaspa.py --dataset %dataset% --encoder_only %encoder_only% ^
    --encoder %model% -elr %elr% -bs %bs% -mbs %mbs% --unfreeze_epoch %unfreeze_epoch% ^
    --fp16 %fp16% --seed %seed% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% ^
    --hyperparameter_tuning True --n_trials %n_trials% --lr_schedule %lr_schedule% --cycles %cycles% ^
    --algebra %algebra% --normalize_embeddings %normalize_embeddings% ^
    --skip_place %skip_place% --sp_layer_norm %sp_layer_norm% --qa_context %qa_context% ^
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