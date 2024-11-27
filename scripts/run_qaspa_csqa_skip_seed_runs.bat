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
set model="roberta-large"


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

@REM @REM best from group qavsa_frozen-lm_residual_tuning_2024-05-27_15-01-56
@REM set dlr=0.0006969517923655713
@REM set dropoutf=0.1
@REM set dropoutspa=0.2
@REM set k=3
@REM set lr_schedule=warmup_cosine_restarts

@REM @REM best from group qavsa_lm-freeze_skip2_tests_2024-05-30_16-21-28
@REM set unfreeze_epoch=4
@REM set refreeze_epoch=100

@REM @REM encoder_lr from paper (qavsa-seed-runs_2024-05-13_14-17-34)
@REM set elr=0.00001766729212078751
@REM set bs=64
@REM set mbs=8
@REM set n_epochs=15
@REM set max_epochs_before_stop=6
@REM set warmup_steps=200
@REM set cycles=1
@REM set algebra=hrr
@REM set fp16=True
@REM set normalize_graphs=False
@REM set sent_trans=False
@REM set qa_context=False
@REM set sp_layer_norm=True
@REM set normalize_embeddings=True
@REM set seed=0

@REM paper params (qavsa-seed-runs_2024-05-13_14-17-34)
set k=5
set elr=0.00001766729212078751
set dlr=0.03713776900239807
@REM set dlr=0.001
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

@REM set sp_dir=../SPA-Embeddings/%dataset%/tzw-concept_bert-rel/made-unitary/pruned/%algebra%/
@REM set emb_dir=../SPA-Embeddings/cpnet/tzw_concept_emb.npy

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel/made-unitary/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy

@REM @REM from group: qavsa_full-skip-test_2024-06-22_19-55-57
@REM set skip_type=1
@REM set skip_place=2

@REM qavsa_full-skip-test_2024-08-04_18-22-42
set skip_type=2
set skip_place=0

set study_name=qavsa_bert-emb_skip%skip_type%_place%skip_place%_seed-runs_%dt%

for /l %%s in (0, 1000, 4000) do (
  set run_name=qavsa-s1-p2_seed%%s
  python -u qaspa.py --dataset %dataset% ^
    --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
    --fp16 %fp16% --seed %%s --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% --refreeze_epoch %refreeze_epoch% ^
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
    --run_name !run_name! --study_name %study_name% --use_wandb %wandb% ^
    --save_dir %save_dir_pref%/%dataset%/%study_name%/!run_name! --save_model %save_model%
)
  
call conda deactivate