@echo off

@REM CALLED BY generate_graph_sps.py

echo "split"
@REM assumes dataset graph info is in ./%dataset%
for /l %%k in (1, 1, %partitions%) do (
    if %permute-vector-seed%==Null (
        echo "No permute vector"
        start "%split%-%%k" cmd /c python graph_sps.py --dataset %dataset% --algebra %algebra% --split %split% ^
            --emb-dir %emb-dir% --concept-file %concept-file% --relation-file %relation-file% --output-dir %outdir% ^
            --pruned %pruned% --normalize-rel %normalize-rel% --normalize-concept %normalize-concept% ^
            --partition-total %partitions% --partition %%k 
            
    ) else (
        echo "Perm seed %permute-vector-seed%"
        start "%split%-%%k" cmd /c python graph_sps.py --dataset %dataset% --algebra %algebra% --permute-vector-seed %permute-vector-seed% --split %split% ^
            --emb-dir %emb-dir% --concept-file %concept-file% --relation-file %relation-file% --output-dir %outdir% ^
            --pruned %pruned% --normalize-rel %normalize-rel% --normalize-concept %normalize-concept% ^
            --partition-total %partitions% --partition %%k 
    )
      
)

:waitloop
set finished=1
for /l %%k in (1, 1, %partitions%) do (
    if not exist %outdir%%split%_graph_sp%%k.npy set finished=0
)
if %finished%==1 ( goto waitloopend )
timeout /t 60
goto waitloop
:waitloopend

set del-parts=True
call python reshape_graph_sps.py --dir %outdir% --dataset %dataset% --split %split% --mc-options %mc-options% --partitions %partitions% --delete-partitions %del-parts%
