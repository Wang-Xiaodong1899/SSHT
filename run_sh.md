# If you want to run the "SSHT"Semi-supervised Source Hypothesis Transfer.
## Step1: train the source model  
For dataset Office-Home:  
python image_source.py --gpu_id 0 --net resnet34 --s 0 --dset office-home  

## Step2: run the SSHT task
For dataset Office-Home:  
python ssht.py --gpu_id 0 --net resnet34 --s 0 --dset office-home --method CDL --trade_off 1 --lamda 2.5 

