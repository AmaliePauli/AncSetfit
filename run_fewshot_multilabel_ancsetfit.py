'''
CREDITS
Modifyed script from: https://github.com/huggingface/setfit/tree/main/scripts/setfit
'''
import argparse
import copy
import json
import math
import os
import pathlib
import sys
from shutil import copyfile
from typing import TYPE_CHECKING, Dict, List, Tuple
from warnings import simplefilter
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from evaluate import load
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from torch.utils.data import DataLoader
from typing_extensions import LiteralString

from setfit.data import SAMPLE_SIZES
from setfit.modeling import SKLearnWrapper, sentence_pairs_generation_multilabel
from data import load_data_splits_multilabel, make_dataframe_task3
from templates import MULTI_TEMPLATE, MULTI_LABELS
from sklearn.metrics import f1_score

LOSS_NAME_TO_CLASS = {
    "CosineSimilarityLoss": losses.CosineSimilarityLoss,
    "TripletLoss": losses.TripletLoss,
}
SEEDS = list(range(10))

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["go_emotions"],
    )
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--num_itaration", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--classifier",
        default="logistic_regression")
    parser.add_argument(
        "--multi_target_strategy",
        default="one-vs-rest",
        choices=["one-vs-rest", "multi-output", "classifier-chain"],
    )
    parser.add_argument("--seeds_num", type=int, default=10)
    parser.add_argument("--loss", default="CosineSimilarityLoss")
    parser.add_argument("--margin", default=0.25, type=float)
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--add_normalization_layer", default=False, action="store_true")
    parser.add_argument("--is_dev_set", type=bool, default=False)
    parser.add_argument("--is_test_set", type=bool, default=False)
    parser.add_argument("--setting_name", type=str, default='multilabel')
    parser.add_argument("--override_results", default=False, action="store_true")
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--savesem", type=bool, default=False)
    
    args = parser.parse_args()

    return args

class SetFitBaseModel:
    def __init__(self,  model, max_seq_length: int, add_normalization_layer: bool) -> None:
        self.model = SentenceTransformer(model, device=DEVICE)
        self.model_original_state = copy.deepcopy(self.model.state_dict())
        self.model.max_seq_length = max_seq_length

        if add_normalization_layer:
            self.model._modules["2"] = models.Normalize()


def sentence_generates_achor_multiLoss(sentences, labels, TEMPLATE, LABELS_NAMES, input_pair):
#train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
#InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
 # this is adding all label words in the sampel to the statemnet   
    for first_idx in range(len(sentences)):
        current_sentence = sentences[first_idx]
        sample_labels = np.where(labels[first_idx, :] == 1)[0]
        if len(sample_labels)!=0:
           
            # get the multilabel anchor
            label_names = ', '.join(LABELS_NAMES[_lab] for _lab in sample_labels)
            anchor = TEMPLATE + label_names
            
            negative_idx = np.where(labels.dot(labels[first_idx, :].T) == 0)[0]
            
            negative_sentence = sentences[np.random.choice(negative_idx)]
            # input pair
            #train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            input_pair.append(InputExample(texts=[anchor, current_sentence, negative_sentence]))
                
    return input_pair

def generate_anchors(template: str, labels: list) -> dict:
    dicts = {}
    for i in range(len(labels)):
        dicts[i] = template + labels[i]
    
    return dicts



class RunFewShot:
    def __init__(self, args: argparse.Namespace) -> None:
        # Prepare directory for results
        self.args = args

        parent_directory = pathlib.Path(__file__).parent.absolute()
        self.output_path = (
            parent_directory
            / "results"
            / f"MULTILABEL-{args.setting_name}-{args.model.replace('/', '-')}-{args.loss}-{args.classifier}-itarations_{args.num_itaration}-batch_{args.batch_size}-{args.exp_name}--margin_{args.margin}".rstrip(
                "-"
            )
        )
        os.makedirs(self.output_path, exist_ok=True)

        # Save a copy of this training script and the run command in results directory
        train_script_path = os.path.join(self.output_path, "train_script.py")
        copyfile(__file__, train_script_path)
        with open(train_script_path, "a") as f_out:
            f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

        self.dataset_to_metric = {dataset: "f1" for dataset in args.datasets}

        # Configure loss function
        self.loss_class = LOSS_NAME_TO_CLASS[args.loss]

        # Load SetFit Model
        self.model_wrapper = SetFitBaseModel(
            self.args.model, max_seq_length=args.max_seq_length, add_normalization_layer=args.add_normalization_layer
        )
        self.model = self.model_wrapper.model

    def get_classifier(self, sbert_model: SentenceTransformer) -> SKLearnWrapper:
        if self.args.classifier == "logistic_regression":
            if self.args.multi_target_strategy == "one-vs-rest":
                multilabel_classifier = OneVsRestClassifier(LogisticRegression())
            elif self.args.multi_target_strategy == "multi-output":
                multilabel_classifier = MultiOutputClassifier(LogisticRegression())
            elif self.args.multi_target_strategy == "classifier-chain":
                multilabel_classifier = ClassifierChain(LogisticRegression())
            return SKLearnWrapper(sbert_model, multilabel_classifier)

    def train(self, data: Dataset, LABELS, TEMPLATE) -> SKLearnWrapper:
        "Trains a SetFit model on the given few-shot training data."
        self.model.load_state_dict(copy.deepcopy(self.model_wrapper.model_original_state))

        x_train = data["text"]
        y_train = data.remove_columns("text").to_pandas().values
        
        if self.loss_class is None:
            return

        # sentence-transformers adaptation
        batch_size = self.args.batch_size
              
       
                
                ## Add to loos
        if self.args.loss=='TripletLoss':
            train_loss = self.loss_class(
                model=self.model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=self.args.margin,
                        )
            
            train_examples = []
            for _ in range(self.args.num_itaration):
                # cahnges how to generate input pairs to fit achor
                
                train_examples = sentence_generates_achor_multiLoss(np.array(x_train), y_train, TEMPLATE, LABELS,train_examples)
            
            
        else: 
            train_loss = self.loss_class(self.model)
            train_examples = []
            for _ in range(self.args.num_itaration):
                train_examples = sentence_pairs_generation_multilabel(np.array(x_train), y_train, train_examples)
    
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_steps = len(train_dataloader)

        print(f"{len(x_train)} train samples in total, {train_steps} train steps with batch size {batch_size}")

        warmup_steps = math.ceil(train_steps * 0.1)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            steps_per_epoch=train_steps,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
        )


        # Train the final classifier
        classifier = self.get_classifier(self.model)
        classifier.fit(x_train, y_train)
        return classifier

    def eval(self, classifier: SKLearnWrapper, data: Dict[str, str], metric: str, results_path: str, LABELS) -> None:
        """Computes the metrics for a given classifier."""
        # Define metrics
        metric_fn = load(metric, config_name="multilabel")
        metric_ac = load('accuracy', config_name="multilabel")
        #metric_rec = load('precision', config_name="multilabel")
        #metric_pre = load('recall', config_name="multilabel")
        

        x_test = data["text"]
        y_test = data.remove_columns("text").to_pandas().values

        y_pred = classifier.predict(x_test)
        
        print('Other scoring function')
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)
        micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=1)
        print('macro new')
        print(macro_f1)
        print('micro_f1')
        print(micro_f1)
        #metrics = metric_fn.compute(predictions=y_pred, references=y_test)
        #print(f"{metric} -- {metrics[metric]:.4f}")
        acc = metric_ac.compute(predictions=y_pred, references=y_test)
        #pre = metric_pre.compute(predictions=y_pred, references=y_test,average='micro')
        #rec = metric_rec.compute(predictions=y_pred, references=y_test,average='micro')
        print('accuracy')
        print(acc['accuracy'])
        
        weF1 = metric_fn.compute(predictions=y_pred, references=y_test, average='weighted')
        print('weighted F1: ')
        print(weF1['f1'])
        maF1 = metric_fn.compute(predictions=y_pred, references=y_test, average='macro')
        print('macro F1: ')
        print(maF1['f1'])
        microF1 = metric_fn.compute(predictions=y_pred, references=y_test, average='micro')
        print('microF1: ')
        print(microF1['f1'])
        f1_ind = metric_fn.compute(predictions=y_pred, references=y_test, average=None)
        
        dict_label={label:score for label, score in zip(LABELS,f1_ind['f1'])}
        
        results_dict = {"macro-f1": maF1['f1'], "micro-f1": microF1['f1'], "accuracy": acc['accuracy']}
        results_dict.update(dict_label)
        
        with open(results_path, "w") as f:
            json.dump(results_dict, f)
        

    def create_results_path(self, dataset: str, split_name: str) -> LiteralString:
        results_path = os.path.join(self.output_path, dataset, split_name, "results.json")
        print(f"\n\n======== {os.path.dirname(results_path)} =======")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        return results_path

    def train_eval_all_datasets(self) -> None:
        """Trains and evaluates the model on each split for every dataset."""
        for dataset, metric in self.dataset_to_metric.items():
            few_shot_train_splits, test_data = load_data_splits_multilabel(dataset, self.args.sample_sizes, SEEDS)
            
            LABELS=MULTI_LABELS[dataset]
            TEMPLATE=MULTI_TEMPLATE[dataset]

            for split_name, train_data in few_shot_train_splits.items():
                results_path = self.create_results_path(dataset, split_name)
                if os.path.exists(results_path) and not self.args.override_results:
                    print(f"Skipping finished experiment: {results_path}")
                    continue

                # Train the model on the current train split
                classifier = self.train(train_data, LABELS, TEMPLATE)

                # Evaluate the model on the test data
                metrics = self.eval(classifier, test_data, metric, results_path, LABELS)

                #with open(results_path, "w") as f_out:
                #    json.dump({"score": metrics[metric] * 100, "measure": metric}, f_out, sort_keys=True)


def main():
    args = parse_args()
    global SEEDS
    SEEDS =[i for i in range(args.seeds_num)]
    global DEVICE
    DEVICE = args.device
    run_fewshot = RunFewShot(args)
    run_fewshot.train_eval_all_datasets()
    


if __name__ == "__main__":
    main()
