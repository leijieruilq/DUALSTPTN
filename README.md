# Dual-STPTN


## running programme

### Single-process experiment: running exp.py

Firstly, please unzip raw_files.zip and then start running.

> >Running style


> >(1) Setting up the experimental task environment: you can do a manual setup of parser.add_argument in exp.py

> >1.1 "model_name":"dualstptn"

> >1.2 "dataset_name": The corresponding "help" in exp.py selects the dataset.

> >1.3 "inp_len": 96

> >1.4 "pred_len": 96/192/336/720

> >(2) Run it directly from the command line：nohup python -u exp.py > train.log 2&>1 &

> >(3) No pre-setting, run directly from the command line：

> > for example：nohup python -u exp.py --note "dualstptn-weather-96" --model_name "dualstptn" --dataset_name "weather" --inp_len 96 --pred_len 96 > train.log 2>&1 &

> > The results are in the corresponding train.log file.
