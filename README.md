# AncSetfit

AncSetfit is effecient method for extreme few shot text classification. It is an extension to the method [SetFit](https://arxiv.org/abs/2209.11055) (finetunning the sentence transformers). This method add semantic label information to the finetuning of the sentence transformers, and thereby help guiding the seperations of the sentence embedding into different classes. 

## Evaluation 
Ancsetfit and Setfit are evaluated on a list of different datasets, in a controlled few-shot setting, testing with balance sampling of 2,4,8,16,32,64 samples per classes.
The script used for running the experiments for both the original SetFit method and the new AncSetfit is available. The scripts are modify and extended from [SetFit github](https://github.com/huggingface/setfit/tree/main/scripts/setfit)


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

### Getting started example /demo
The jupyter notebook tiny_example.ipyn is a quick demo to try it - the notebook shows in a small hand-crafted example how AncSetFit can learn to generalize differently based on the same training data depending on the provided anchor statements of textual description of the classes.

### publication
To appear at EMNLP 2023, short paper: Anchoring Fine-tuning of Sentence Transformer with Semantic Label Information for Efficient Truly Few-shot Classification
 
