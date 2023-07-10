import os
import re
import pandas as pd
import argparse
import json

def get_results(folder,dataset):
    df_results = pd.DataFrame(columns=['dataset','train_size','seed','accuracy'])
    settings = [setting for setting in os.listdir(folder)]
    for setting in settings:
        print(setting)
        setting_folder=os.path.join(folder,setting)
        results_path = os.path.join(setting_folder,'results.json')
        get_size = re.search('-(.*)-', setting)
        size = int(get_size.group(1))
        get_seed = re.search('.C*$', setting)
        seed = int(get_seed.group(0))
        with open(results_path, "r") as f:
            results = json.loads(f.read())
            acc=results["score"]
        list_results = [dataset,size,seed,acc]
        df_results.loc[len(df_results.index)] = list_results
        results_path = folder +'_results.csv'
    df_results.to_csv(results_path)
    return df_results

def average_results(df):
    df = df[df.seed<5]
    for size in df['train_size'].unique():
        print(size)
        print('shpae:', df[df.train_size==size].shape)
        print('Avg:', df[df.train_size==size].accuracy.mean())
        print('STD:', df[df.train_size==size].accuracy.std())

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--data",type=str, default=None)
    #args = parser.parse_args()
    print(os.getcwd())
    cwd = os.getcwd()
    dataset = 'ag_news'
    path = os.path.join(cwd,'seed_output/albert-xxlarge-v2',dataset)
    
    
    df=get_results(path,dataset)
    average_results(df)
    
    
