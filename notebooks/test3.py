import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5Config

# Assuming T5WithSVD, construct_prompt, device, collate_fn_fn, etc. are defined above...

def construct_prompt(sample, dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "agnews":
        # return "classify ag_news dataset: " + sample["text"]
        return (
            "What is the topic of the following paragraph? "
            "Choose one from the following options [World, Sports, Business, Science or Technology]. "
            + sample["sentence"]
        )
    elif dataset_name in ["amazon", "yelp"]:
        # return "classify amazon dataset: " + sample["content"]
        return (
            "What is the sentiment of the following paragraph? "
            "Choose one from the following options [very negative, negative, neutral, positive, very positive]. "
            + sample["sentence"]
        )
    elif dataset_name == "dbpedia":
        # return "classify dbpedia dataset: " + sample["content"]
        return (
            "What is the topic of the following paragraph? "
            "Choose one from the following options [Company, Educational Institution, Artist, Athlete, Office Holder, "
            "Mean of Transportation, Building, Natural Place, Village, Animal, Plant, Album, "
            "Film, Written Work]. "
            + sample["sentence"]
        )
    elif dataset_name == "yahoo":
        # return "classify yahoo dataset: " + sample["question_title"] + " " + sample["question_content"]
        return (
            "What is the topic of the following paragraph? "
            "Choose one from the following options [Sports, Entertainment & Music, Health, Education & Reference, Family & Relationships, Politics & Government, Science & Mathematics, Business & Finance, Computers & Internet, Society & Culture]. "
            + sample["sentence"]
        )
    elif dataset_name == "mnli":
        # return "classify mnli dataset: premise: " + sample["premise"] + " hypothesis: " + sample["hypothesis"]
        return (
            "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? Choose one from the following options [neutral, entailment, contradiction]. "
            + sample["sentence"]
        )
    elif dataset_name == "qqp":
        # return "classify qqp dataset: question1: " + sample["question1"] + " question2: " + sample["question2"]
        return (
            "Whether the 'first sentence' and the 'second sentence' have the same meaning? "
            "Choose one from the following options from [True, False]. "
            + sample["sentence"]
        )
    elif dataset_name == "rte":
        # return "classify rte dataset: sentence1: " + sample["sentence1"] + " sentence2: " + sample["sentence2"]
        return (
            "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? "
            "Choose one from the following options [entailment, contradiction]. "
            + sample["sentence"]
        )
    elif dataset_name == "sst-2":
        # return "classify sst2 dataset: sentence: " + sample["sentence"]
        return (
            "What is the sentiment of the following paragraph? "
            "Choose one from the following options [Bad, Good]. "
            + sample["sentence"]
        )
    elif dataset_name == "wic":
        # return "classify wic dataset: word: " + sample["word"] + " sentence1: " + sample["sentence1"] + " sentence2: " + sample["sentence2"]
        return (
            "Given a word and two sentences, whether the word is used with the same sense in both sentence? "
            "Choose one from the following options [True, False]. "
            + sample["sentence"]
        )
    elif dataset_name == "cb":
        # return "classify cb dataset: premise: " + sample["premise"] + " hypothesis: " + sample["hypothesis"]
        return (
            "What is the logical relationship between the 'sentence 1' and the 'sentence 2'? "
            "Choose one from the following options [neutral, entailment, contradiction]. "
            + sample["sentence"]
        )
    elif dataset_name == "copa":
        # return "classify copa dataset: premise: " + sample["premise"] + " choice1: " + sample["choice1"] + " choice2: " + sample["choice2"]
        return (
            "Which choice best explains or follows from the given premise? "
            "Choose one from the following options [A, B]. "
            + sample["sentence"]
        )
    elif dataset_name == "multirc":
        # return "classify multirc dataset: question: " + sample["question"] + " passage: " + sample["passage"]
        return (
            "According to the following passage and question, is the candidate answer true or false? "
            "Choose one from the following options [True, False]. "
            + sample["sentence"]
        )
    elif dataset_name == "boolqa":
        # return "classify boolq dataset: question: " + sample["question"] + " passage: " + sample["passage"]
        return (
            "According to the following passage, is the question true or false? "
            "Choose one from the following options [True, False]. "
            + sample["sentence"]
        )
    elif dataset_name == "imdb":
        # return "classify imdb dataset: " + sample["text"]
        return (
            "What is the sentiment of the following paragraph? "
            "Choose one from the following options [Bad, Good]. "
            + sample["sentence"]
        )
    else:
        return "classify dataset: " + sample.get("text", sample.get("content", ""))

####################################################################
# 1) New Class: EvaluationDataset
####################################################################
class EvaluationDataset(Dataset):
    """
    A dataset class that loads a JSON file and constructs prompts + labels.
    """
    def __init__(self, json_path, tokenizer, label_mapping, dataset_name):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping

        # Load data from JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        self.examples = []
        for sample in dataset:
            prompt = construct_prompt(sample, self.dataset_name)

            # Directly retrieve label as a string (assuming stored as a string in JSON)
            label_str = str(sample.get("label", sample.get("topic"))).strip()

            # Validate label exists in label_mapping values
            if label_str not in label_mapping.values():
                print(f"Warning: Invalid label '{label_str}' in {self.dataset_name}")
                continue

            self.examples.append((prompt, label_str))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]  # (prompt, label)

def collate_fn_fn(batch, tokenizer, max_source_length=512, max_target_length=16):
    inputs, targets = zip(*batch)
    input_encodings = tokenizer(list(inputs), padding=True, truncation=True, max_length=max_source_length, return_tensors="pt")
    
    # Ensure targets (labels) are strings before tokenization
    target_encodings = tokenizer(list(targets), padding=True, truncation=True, max_length=max_target_length, return_tensors="pt")
    
    input_encodings["labels"] = target_encodings["input_ids"]  
    return input_encodings

####################################################################
# 2) Evaluate Model on a Dataset and Return Accuracy
####################################################################
def evaluate_model_return(model, tokenizer, test_loader):
    """Evaluates the model on a test loader and returns the accuracy."""
    model.eval()
    total, correct = 0, 0

    printed = 0
    max_print = 5
    
    for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
        for key, val in batch.items():
            batch[key] = val.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=16
            )

        predictions = [
            tokenizer.decode(g, skip_special_tokens=True).strip().lower()
            for g in generated_ids
        ]
        targets = [
            tokenizer.decode(label, skip_special_tokens=True).strip().lower()
            for label in batch["labels"]
        ]
        
        for pred, target in zip(predictions, targets):
            total += 1
            if printed < max_print:
                print(f"\nExample {printed+1}")
                print(f"  Predicted: {pred}")
                print(f"  Actual:    {target}")
                printed += 1

            if pred.lower() == target.lower():
                correct += 1

    return correct / total if total > 0 else 0


####################################################################
# 3) Evaluate on All Tasks
####################################################################
def evaluate_on_all_tasks(model_checkpoint, dataset_infos):
    """
    Loads the final model from `model_checkpoint` and evaluates it on each 
    task defined in `dataset_infos`.
    Prints the accuracy for each task and the overall average accuracy.
    """
    # Load tokenizer and model config
    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.use_cache = False

    # Load trained model
    model = T5WithSVD(config, svd_config={}, initialize_svd=False)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device), strict=False)
    model.reinitialize_svd()
    model = model.to(device)
    model.eval()

    task_accuracies = {}

    # Loop over each task in dataset_infos
    for task_name, info in dataset_infos.items():
        print(f"\nEvaluating on {task_name} set:")

        json_path = info["json_path"]
        label_mapping = info["label_mapping"]

        # Check if JSON file exists
        if not os.path.exists(json_path):
            print(f"Warning: Test file not found for {task_name} at {json_path}")
            continue

        # 1) Load the dataset
        eval_dataset = EvaluationDataset(json_path, tokenizer, label_mapping, task_name)

        # 2) Create DataLoader
        test_loader = DataLoader(
            eval_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda batch: collate_fn_fn(batch, tokenizer)
        )

        # 3) Evaluate & store accuracy
        acc = evaluate_model_return(model, tokenizer, test_loader)
        task_accuracies[task_name] = acc
        print(f"{task_name} accuracy: {acc * 100:.2f}%")


    # Compute average accuracy across all tasks
    avg_acc = np.mean(list(task_accuracies.values())) if task_accuracies else 0
    print("\nAverage accuracy across all tasks: {:.2f}%".format(avg_acc * 100))
    return task_accuracies

####################################################################
# 4) Define Dataset Information
####################################################################
DATASET_INFOS = {
    "agnews": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/TC/agnews/test.json",
        "label_mapping": {0: "World", 1: "Sports", 2: "Business", 3: "Science or Technology"}
    },
    "amazon": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/amazon/test.json",
        "label_mapping": {0: "negative", 1: "positive", 2: "neutral", 3: "very positive", 4: "very negative"}
    },
    "yelp": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/yelp/test.json",
        "label_mapping": {0: "negative", 1: "positive", 2: "neutral", 3: "very positive", 4: "very negative"}
    },
    "dbpedia": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/TC/dbpedia/test.json",
        "label_mapping": {0: "Company", 1: "Educational Institution", 2: "Artist",
                           3: "Athlete", 4: "Office Holder", 5: "Mean of Transportation",
                           6: "Building", 7: "Natural Place", 8: "Village",
                           9: "Animal", 10: "Plant", 11: "Album", 12: "Film", 13: "Written Work"}
    },
    "yahoo": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/TC/yahoo/test.json",
        "label_mapping": {0: "Society & Culture", 1: "Science & Mathematics", 2: "Health", 3: "Education & Reference",
                           4: "Computers & Internet", 5: "Sports", 6: "Business & Finance", 7: "Entertainment & Music",
                           8: "Family & Relationships", 9: "Politics & Government"}
    },
    "mnli": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/NLI/MNLI/test.json",
        "label_mapping": {0: "entailment", 1: "neutral", 2: "contradiction"}
    }, 
    "qqp": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/QQP/QQP/test.json",
        "label_mapping": {0: "True", 1: "False"}
    },
    "rte": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/NLI/RTE/test.json",
        "label_mapping": {0: "entailment", 1: "contradiction"}
    },
    "sst-2": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/SST-2/test.json",
        "label_mapping": {0: "Bad", 1: "Good"}
    },
    "wic": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/WiC/WiC/test.json",
        "label_mapping": {0: "True", 1: "False"}
    },
    "cb": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/NLI/CB/test.json",
        "label_mapping": {0: "contradiction", 1: "entailment", 2: "neutral"}
    },
    "copa": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/COPA/COPA/test.json",
        "label_mapping": {0: "A", 1: "B"}
    },
    "multirc": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/MultiRC/MultiRC/test.json",
        "label_mapping": {0: "True", 1: "False"}  
    },
    "boolqa": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/BoolQA/BoolQA/test.json",
        "label_mapping": {0: "True", 1: "False"}
    },
    "imdb": {
        "json_path": "/workspace/O-LoRA/CL_Benchmark/SC/IMDB/test.json",
        "label_mapping": {0: "Bad", 1: "Good"}
    }
}


####################################################################
# 5) Run Evaluation
####################################################################
if __name__ == "__main__":
    final_model_path = "t5_svd_yahoo.pt"  # Update with the correct model checkpoint
    all_accuracies = evaluate_on_all_tasks(final_model_path, DATASET_INFOS)