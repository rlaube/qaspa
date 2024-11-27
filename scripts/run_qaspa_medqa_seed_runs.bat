@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "datestamp=%YYYY%%MM%%DD%" & set "timestamp=%HH%%Min%%Sec%"
set "dt=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"
echo %dt%

set CUDA_VISIBLE_DEVICES=0
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
set max_epochs_before_stop=7
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

echo "***** hyperparameters *****"
echo "dataset: %dataset%"
echo "enc_name: %model%"
echo "graph algebra: %algebra%"
echo "batch_size: %bs%"
echo "learning_rate: elr %elr% dlr %dlr%"
echo "fc sp: layer %k%"
echo "******************************"

@REM save_model=0 doesn't save, save_model=1 saves on best dev acc, save_model=2 saves every epoch
set wandb=True
set save_model=1
set save_dir_pref=runs
mkdir %save_dir_pref%

@REM timeout /t 7200 /nobreak

call conda activate qaspa

set study_name=qavsa-seed-runs_%dt%

set sp_dir=../SPA-Embeddings/%dataset%/bert-concept_bert-rel_v2/l2-norm/pruned/%algebra%/
set emb_dir=../SPA-Embeddings/umls/encoder_embs/umls_concept_embs_v2.npy


for /l %%s in (10, 10, 40) do (
  set run_name=qavsa_l2-norm_seed%%s
  call python -u qaspa.py --dataset %dataset% --max_seq_len %max_seq_len% --encoder_only %encoder_only% ^
      --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropoutf% --dropoutspa %dropoutspa% --sp_layer_norm %sp_layer_norm% ^
      --fp16 %fp16% --seed %%s --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% --cycles %cycles% ^
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
      --train_statements  data/%dataset%/statement/train.statement.jsonl ^
      --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
      --test_statements  data/%dataset%/statement/test.statement.jsonl ^
      --run_name !run_name! --study_name %study_name% --use_wandb %wandb% ^
      --log_interval 1 ^
      --save_dir %save_dir_pref%/%dataset%/%study_name%/!run_name! --save_model %save_model%
)



call conda deactivate
