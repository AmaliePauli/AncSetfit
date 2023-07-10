# AncSetfit

AncSetfit is effecient method for extreme few shot text classification. It is an extension to the method SetFit (finetunning the sentence transformers). This method add semantic label information to the finetuning of the sentence transformers, and thereby help guiding the seperations of the sentence embedding into different classes. 

## Evaluation 
Ancsetfit and Setfit is evaluated on a list of different datasets, in a controlled few-shot setting, testing with balance sampling of 2,4,8,16,32,64 samples per classes.
The script used for running the experiments for both the origional SetFit method and the new AncSetfit is availeble. The scripts is modify and extended form [SetFit github](https://github.com/huggingface/setfit/tree/main/scripts/setfit)

The method ADAPET is also test with some dataset - script from [setfit github](https://github.com/huggingface/setfit/blob/main/scripts/adapet/ADAPET/setfit_adapet.py) (minimal modified)


### Commands
Repeat with 20 seeds and different dataset sizes.

**AncSetFit**
```python 
python run_fewshot_ancsetfit.py \
--sample_sizes 2 4 8 16 32 64 \
--loss=TripletLoss \
--margin 0.25 \
--is_test_set True \
--num_itaration 40 \
--seeds_num 20 \
--setting_name TEST 
```

**SetFit**
```python 
python run_fewshot_ancsetfit.py \
--sample_sizes 2 4 8 16 32 64 \
--loss=CosineSimilarityLoss \
--is_test_set True \
--num_itaration 20 \
--seeds_num 20 \
--setting_name TEST 
```