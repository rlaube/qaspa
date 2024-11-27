@echo off

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "datestamp=%YYYY%%MM%%DD%" & set "timestamp=%HH%%Min%%Sec%"
set "dt=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"
echo %dt%

set CUDA_VISIBLE_DEVICES=0
set dataset=medqa
set model=michiyasunaga/BioLinkBERT-large

@REM qavsa_my-embed_frozen-lm_tuning_2024-09-26_19-14-20 (k3_dlr1.82e-04_warmup_cosine_restarts_cycle2_warm50_bs32_mbs2_dropsp0.7_dropf0.0)
set algebra=hrr
set k=3
set dlr=0.00018225539664628204
set bs=32
set mbs=2
set ebs=8
set unfreeze_epoch=100
set n_epochs=30
set max_epochs_before_stop=6
set lr_schedule=warmup_cosine_restarts
set warmup_steps=50

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

@REM set algebra=hrr
@REM set k=3
@REM set mbs=32
@REM set ebs=32
@REM set unfreeze_epoch=100
@REM set n_epochs=30
@REM set max_epochs_before_stop=6
@REM set lr_schedule=warmup_cosine_restarts

@REM set dropoutspa=0.7
@REM set qa_context=False
@REM set sent_trans=False
@REM set normalize_graphs=False
@REM set sp_layer_norm=True
@REM set normalize_embeddings=True
@REM set max_seq_len=512
@REM set skip_type=0
@REM set skip_placement=0

@REM set fp16=True
@REM set seed=0
@REM set encoder_only=False


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

call conda activate qaspa

set n_trials=400
set study_name=qavsa_lm_l2-norm_tuning_%dt%

@REM timeout /t 9000 /nobreak

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/umls/encoder_embs/umls_concept_embs_v2.npy

call python -u qaspa.py --dataset %dataset% --max_seq_len %max_seq_len% --encoder_only %encoder_only% -k %k% ^
    --algebra %algebra% --encoder %model% -dlr %dlr% -bs %bs% -mbs %mbs% -ebs %ebs% --dropoutspa %dropoutspa% ^
    --normalize_embeddings %normalize_embeddings% --sent_trans %sent_trans% --normalize_graphs %normalize_graphs% ^
    --skip_placement %skip_placement% --skip_type %skip_type% --sp_layer_norm %sp_layer_norm% --qa_context %qa_context% ^
    --fp16 %fp16% --seed %seed% ^
    --n_epochs %n_epochs% --max_epochs_before_stop %max_epochs_before_stop% ^
    --hyperparameter_tuning True --n_trials %n_trials% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% ^
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
    --save_dir %save_dir_pref%/%dataset%/%study_name%/ --save_model %save_model%
  
call conda deactivate