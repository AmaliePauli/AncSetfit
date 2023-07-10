'''
CREDITS
Modifyed script from: https://github.com/huggingface/setfit/tree/main/scripts/setfit
'''

from typing import TYPE_CHECKING, Dict, List, Tuple
import pandas as pd
import torch 
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset
from typing import TYPE_CHECKING, Dict, List, Tuple
import os
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

TokenizerOutput = Dict[str, List[int]]
#SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def load_data_splits(
    dataset: str, sample_sizes: List[int], seeds: List[int]
) -> Tuple[DatasetDict, Dataset]:
    """Loads a dataset from the Hugging Face Hub and returns the test split and few-shot training splits."""
    print(f"\n\n\n============== {dataset} ============")
        
    if dataset=='yahoo_answers_topics':
        df_train_split= load_dataset(f"{dataset}", split="train").to_pandas().rename(columns={'topic':'label'})
        df_test_split = load_dataset(f"{dataset}", split="test").to_pandas().rename(columns={'topic':'label'})
        df_test_split=df_test_split.sample(5000,random_state=0)
        df_train_split['text'] = df_train_split['question_title']+ ' ' + df_train_split['question_content']
        df_test_split['text'] = df_test_split['question_title']+ ' ' + df_test_split['question_content']
        train_split= Dataset.from_pandas(df_train_split[['label','text']])
        test_split= Dataset.from_pandas(df_test_split[['label','text']])
        
    elif dataset=='yelp_review_full':
        train_split = load_dataset(f"{dataset}", split="train") 
        df_test_split = load_dataset(f"{dataset}", split="test").to_pandas()
        df_test_split=df_test_split.sample(5000,random_state=0)
        test_split=Dataset.from_pandas(df_test_split[['label','text']])
    
    # if dataset is on setfit hub
    else:
        # Load one of the SetFit training sets from the Hugging Face Hub
        train_split = load_dataset(f"SetFit/{dataset}", split="train") 
        test_split = load_dataset(f"SetFit/{dataset}", split="test")
        if dataset=='imdb':
            test_split = Dataset.from_pandas(test_split.to_pandas().sample(4000,random_state=0))
        
    print(f"Test set: {len(test_split)}")
    train_splits = create_fewshot_splits(train_split, sample_sizes, dataset, seeds)
    return train_splits, test_split


def create_samples(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    examples = []
    for label in df["label"].unique():
        subset = df.query(f"label == {label}")
        if len(subset) > sample_size:
            examples.append(subset.sample(sample_size, random_state=seed, replace=False))
        else:
            examples.append(subset)
    return pd.concat(examples)




def create_fewshot_splits(
    dataset: Dataset, sample_sizes: List[int], dataset_name: str = None, seeds: List[int]=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ) -> DatasetDict:
    """Creates training splits from the dataset with an equal number of samples per class (when possible)."""
    splits_ds = DatasetDict()
    df = dataset.to_pandas()

    for sample_size in sample_sizes:
        for idx, seed in enumerate(seeds):
            split_df = create_samples(df, sample_size, seed)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    return splits_ds



def load_data_splits_multilabel(dataset: str, sample_sizes: List[int], SEEDS) -> Tuple[DatasetDict, Dataset]:
    """Loads a dataset and returns the test split and few-shot training splits in DataSet format."""
    # load dataset, train fit format to df, send to fkt for split and return dataset, test, format and return dataset
    print(f"\n\n\n============== {dataset} ============")
    
    if dataset =='go_emotions':
        # Load one of the SetFit training sets from the Hugging Face Hub
        df_train_split = load_dataset(f"SetFit/{dataset}", "multilabel", split="train").to_pandas()    
        test_split = load_dataset(f"SetFit/{dataset}", "multilabel", split="test")
        
    elif dataset=='abstract':
        #load local!!
        print('this dataset is loaded local! make sure path exisist')
        cwd=os.getcwd()
        path_abstract= os.path.join(cwd, 'abstract_train.csv')
        df_abstract = pd.read_csv(path_abstract)
        df_abstract['text'] = df_abstract['TITLE'] + ' ' + df_abstract['ABSTRACT'] 
        df_abstract=df_abstract.drop(columns = ['ID','TITLE','ABSTRACT'])
        df_abstract=df_abstract.rename(columns={'Computer Science':'ComputerScience','Quantitative Biology':'QuantitativeBiology', 'Quantitative Finance': 'QuantitativeFinance'})
        
        df_train_split=df_abstract.sample(frac=0.75,random_state=42)
        test_split=Dataset.from_pandas(df_abstract.drop(df_train_split.index), preserve_index=False)
        
    elif dataset == 'semeval2018task1':
        df_train_split = load_dataset(f"sem_eval_2018_task_1", "subtask5.english", split="train").to_pandas()
        df_train_split=df_train_split.drop(columns='ID')
        df_train_split=df_train_split.rename(columns={'Tweet': 'text'})
        
        df_test_split = load_dataset(f"sem_eval_2018_task_1", "subtask5.english", split="test").to_pandas()
        df_test_split=df_test_split.drop(columns='ID')
        df_test_split=df_test_split.rename(columns={'Tweet': 'text'})
        test_split = Dataset.from_pandas(df_test_split, preserve_index=False)
    

        
    else:
        print('*** dataset not implemented ***')
    
    train_splits = create_fewshot_splits_multilabel(df_train_split, sample_sizes, SEEDS)
    print(f"Test set: {len(test_split)}")
    return train_splits, test_split

#def old_create_fewshot_splits_multilabel(dataset: Dataset, sample_sizes: List[int], seeds: List[int]=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ) -> DatasetDict:
#    splits_ds = DatasetDict()
#    df = dataset.to_pandas()
#    for sample_size in sample_sizes:
#        for idx, seed in enumerate(seeds):
#            split_df = create_samples_multilabel(df, sample_size, seed)
#            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
#    return splits_ds

def create_fewshot_splits_multilabel(df, sample_sizes: List[int], seeds: List[int]=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ) -> DatasetDict:
    splits_ds = DatasetDict()
    #df = Dataset.to_pandas()
    for sample_size in sample_sizes:
        for idx, seed in enumerate(seeds):
            split_df = create_samples_multilabel(df, sample_size, seed)
            splits_ds[f"train-{sample_size}-{idx}"] = Dataset.from_pandas(split_df, preserve_index=False)
    return splits_ds


def create_samples_multilabel(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Samples a DataFrame to create an equal number of samples per class (when possible)."""
    examples = []
    column_labels = [_col for _col in df.columns.tolist() if _col != "text"]
    for label in column_labels:
        subset = df.query(f"{label} == 1")
        if len(subset) > sample_size:
            examples.append(subset.sample(sample_size, random_state=seed, replace=False))
        else:
            examples.append(subset)
    # Dropping duplicates for samples selected multiple times as they have multi labels
    return pd.concat(examples).drop_duplicates()







