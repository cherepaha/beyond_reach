set root=C:\Users\Arkady\Anaconda3

call %root%\Scripts\activate.bat py27_32

python D:\source\beyond_the_reach\data_collection\run_experiment.py %*

pause