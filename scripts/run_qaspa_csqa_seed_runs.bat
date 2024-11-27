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

@REM @REM from group no_qa-context_tuning_2024-04-13_15-37-21 (CSQA) (run k4_dlr2.16e-04_unfreeze0_dropsp0.7_dropf0.6_warmup_linear_steps-200)
@REM set elr=1e-05
@REM set dlr=0.0002156860938081096
@REM set bs=64
@REM set mbs=8
@REM set dropoutf=0.7
@REM set dropoutspa=0.6
@REM set unfreeze_epoch=0
@REM set n_epochs=15
@REM set max_epochs_before_stop=6
@REM set k=4
@REM set lr_schedule=warmup_linear
@REM set warmup_steps=200
@REM set cycles=1
@REM set algebra=hrr
@REM set fp16=True
@REM set normalize_graphs=False
@REM set sent_trans=False
@REM set qa_context=False
@REM set sp_layer_norm=False
@REM set normalize_embeddings=False
@REM set seed=0

@REM @REM best from group qavsa_frozen-lm_residual_tuning_2024-05-27_15-01-56
@REM set dlr=0.0006969517923655713
@REM set dropoutf=0.1
@REM set dropoutspa=0.2
@REM set k=3
@REM set lr_schedule=warmup_cosine_restarts

@REM @REM best from group qavsa_lm-freeze_skip2_tests_2024-05-30_16-21-28
@REM set unfreeze_epoch=4
@REM set refreeze_epoch=100

@REM paper params (qavsa-seed-runs_2024-05-13_14-17-34)
set k=5
@REM set elr=0.00001766729212078751
set elr=0.00001
@REM set dlr=0.03713776900239807
set dlr=0.001
set bs=64
set mbs=8
set dropoutf=0
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
set skip_place=0
set skip_type=0


echo "***** hyperparameters *****"
echo "dataset: %dataset%"
echo "enc_name: %model%"
echo "batch_size: %bs%"
echo "learning_rate: elr %elr% dlr %dlr%"
echo "fc sp: layer %k%"
echo "******************************"
   
@REM save_model=0 doesn't save, save_model=1 saves on best dev acc, save_model=2 saves every epoch
set wandb=True
set save_model=1
set save_dir_pref=runs
mkdir %save_dir_pref%

call conda activate qaspa

set study_name=qaspa_l2-norm_qagnn-lr_1000s-seed-runs_%dt%

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/cpnet/encoder_embs/bert-large-uncased_concept_emb_v2.npy

for /l %%s in (0, 1000, 4000) do (
  set run_name=qavsa_seed%%s
  python -u qaspa.py --dataset %dataset% --encoder_only %encoder_only% ^
    --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
    --fp16 %fp16% --seed %%s --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% --normalize_graphs %normalize_graphs% ^
    --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
    --skip_type %skip_type% --skip_placement %skip_place% ^
    --log_interval 1 ^
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
    --run_name !run_name! --study_name %study_name% --use_wandb %wandb% ^
    --save_dir %save_dir_pref%/%dataset%/!run_name! --save_model %save_model%
)

@REM set encoder_only=True
@REM set study_name=qavsa-lm_only-seed-10s-runs_%dt%

@REM for /l %%s in (0, 10, 40) do (
@REM   set run_name=qavsa-lm_only-seed%%s
@REM   python -u qaspa.py --dataset %dataset% --encoder_only %encoder_only% ^
@REM     --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% ^
@REM     --fp16 %fp16% --seed %%s --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
@REM     --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% --unfreeze_epoch %unfreeze_epoch% --refreeze_epoch %refreeze_epoch% ^
@REM     --log_interval 1 ^
@REM     --hyperparameter_tuning False ^
@REM     --algebra %algebra% --sent_trans %sent_trans% --qa_context %qa_context% ^
@REM     --normalize_graphs %normalize_graphs% --sp_layer_norm %sp_layer_norm% --normalize_embeddings %normalize_embeddings% ^
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
@REM     --run_name !run_name! --study_name %study_name% --use_wandb %wandb% ^
@REM     --save_dir %save_dir_pref%/%dataset%/%study_name%/!run_name! --save_model %save_model%
@REM )
