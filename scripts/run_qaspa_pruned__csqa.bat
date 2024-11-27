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

@REM from study_csqa__enc-roberta-large__seed0__20231015_171649.log: dev_acc 0.772, test_acc 0.7196
@REM set elr="8.57827574864017e-06"
@REM set dlr="0.0002770642990109368"
@REM set bs=32
@REM set mbs=8
@REM set dropout=0.2
@REM set unfreeze_epoch=4
@REM set n_epochs=15
@REM set k=5
@REM set lr_schedule=fixed
@REM set warmup_steps=180
@REM set fp16=False

@REM  from study_csqa__enc-roberta-large__seed0__20240112_155625.log: dev_acc 0.7740
@REM set elr="4.247752071660842e-06"
@REM set dlr="0.007267756422754498"
@REM set bs=16
@REM set mbs=8
@REM set dropout=0.2
@REM set unfreeze_epoch=4
@REM set n_epochs=15
@REM set k=5
@REM set lr_schedule=warmup_linear
@REM set warmup_steps=45
@REM set fp16=True

@REM from enc-roberta-large__k4__bs64__seed0__2024-01-26_15-51-24
set elr=1e-05
set dlr=0.001
set bs=64
set mbs=8
set dropout=0.2
set unfreeze_epoch=4
set n_epochs=15
set k=4
set lr_schedule=fixed
set warmup_steps=45
set fp16=True

set seed=0

set algebra=hrr_normalized
set qa_context=True
set pruned=_pruned
set sent_trans=True

echo "***** hyperparameters *****"
echo "dataset: %dataset%"
echo "enc_name: %model%"
echo "batch_size: %bs%"
echo "learning_rate: elr %elr% dlr %dlr%"
echo "fc sp: layer %k%"
echo "******************************"

set save_dir_pref=saved_models
mkdir %save_dir_pref%
mkdir logs

python -u qaspa.py --dataset %dataset% ^
  --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropout% --dropoutspa %dropout% ^
  --fp16 %fp16% --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% ^
  --n_epochs %n_epochs% --max_epochs_before_stop 10 --unfreeze_epoch %unfreeze_epoch% ^
  --log_interval 50 ^
  --hyperparameter_tuning False ^
  --algebra %algebra% --qa_context %qa_context% --sent_trans %sent_trans% ^
  --train_adj data/%dataset%/graph/train.graph.adj.pk ^
  --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
  --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
  --train_sp data/%dataset%/spa/%algebra%%pruned%/train_graph_sp.npy ^
  --dev_sp   data/%dataset%/spa/%algebra%%pruned%/dev_graph_sp.npy ^
  --test_sp  data/%dataset%/spa/%algebra%%pruned%/test_graph_sp.npy ^
  --train_statements  data/%dataset%/statement/train.statement.jsonl ^
  --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
  --test_statements  data/%dataset%/statement/test.statement.jsonl ^
  >> logs/train_%dataset%%pruned%_%algebra%_k%k%_%dt%.log.txt

@REM do same thing but fp16 = False (to compare with prev results)
@REM python -u qaspa.py --dataset %dataset% ^
@REM   --encoder %model% -k %k% -elr %elr% -dlr %dlr% -bs %bs% -mbs %mbs% --dropoutf %dropout% --dropoutspa %dropout% ^
@REM   --fp16 False --seed %seed% --lr_schedule %lr_schedule% --warmup_steps %warmup_steps% ^
@REM   --n_epochs %n_epochs% --max_epochs_before_stop 10 --unfreeze_epoch %unfreeze_epoch% ^
@REM   --log_interval 50 ^
@REM   --hyperparameter_tuning False ^
@REM   --algebra %algebra% --qa_context %qa_context% --sent_trans %sent_trans% ^
@REM   --train_adj data/%dataset%/graph/train.graph.adj.pk ^
@REM   --dev_adj   data/%dataset%/graph/dev.graph.adj.pk ^
@REM   --test_adj  data/%dataset%/graph/test.graph.adj.pk ^
@REM   --train_sp data/%dataset%/spa/%algebra%%pruned%/train_graph_sp.npy ^
@REM   --dev_sp   data/%dataset%/spa/%algebra%%pruned%/dev_graph_sp.npy ^
@REM   --test_sp  data/%dataset%/spa/%algebra%%pruned%/test_graph_sp.npy ^
@REM   --train_statements  data/%dataset%/statement/train.statement.jsonl ^
@REM   --dev_statements  data/%dataset%/statement/dev.statement.jsonl ^
@REM   --test_statements  data/%dataset%/statement/test.statement.jsonl ^
@REM   >> logs/train_%dataset%%pruned%_%algebra%_k%k%_%dt%.log.txt


