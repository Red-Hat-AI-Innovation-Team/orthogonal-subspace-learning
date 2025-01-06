import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from dataset_utils.fever import FEVER
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress


class GraniteExperiment:

    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def effective_rank(self, matrix, threshold=1e-5):
        """
        Compute the effective rank of a matrix using its singular values.
        Singular values below the threshold are considered as zero.
        """
        u, s, v = torch.svd(matrix)
        effective_rank = torch.sum(s > threshold).item()
        return effective_rank

    def intervene(self, model, tokenizer, dataset, args, llm_name):

        original_params = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}

        time_edit_start = time.time()
        model_edit = LaserWrapper.get_edited_model(model=model,
                                                   lname=args.lname,
                                                   lnum=args.lnum,
                                                   rate=args.rate,
                                                   intervention=args.intervention,
                                                   logger=logger,
                                                   in_place=True)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        self.logger.log("Checking for parameter changes...")
        for name, param in model_edit.named_parameters():
            if name in original_params and not torch.equal(original_params[name], param.detach().cpu()):
                
                original_matrix = original_params[name]
                new_matrix = param.detach().cpu()

                # Compute effective ranks
                original_rank = self.effective_rank(original_matrix)
                new_rank = self.effective_rank(new_matrix)

                # Print details
                print(f"Layer changed: {name}")
                print(f"Original Effective Rank: {original_rank}")
                print(f"New Effective Rank: {new_rank}")
                print("-" * 50)

                print(f"Before change: {original_matrix}")
                print(f"After change: {new_matrix}")

        save_path = f"/new_data/experiments/nn-orthogonal/{args.lname}-{args.lnum}-{args.rate}"
        os.makedirs(save_path, exist_ok=True)
        model_edit.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")



if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with Granite')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=1, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None',
                                 'dont', 'all', 'mlp', 'attn'],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=28, help='Layers to edit', choices=list(range(-1, 40)))
    parser.add_argument('--model_path',
                        type=str,
                        default="/new_data/experiments/ap-8b-p10-rhel13-data-id-2/hf_format/samples_10597250",
                        help="Place where model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="/mnt/data/iclr2024/fever/llama_results",
                        help='Directory where the data is')
    parser.add_argument('--dataset_file', type=str,
                        default="/mnt/data/counterfact",
                        help='Directory where the data is')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = "Granite"
    llm_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file

    save_dir = f"./"

    logger = Logger(save_dir=save_dir, fname=f"dummy.txt")

    # Step 4: Create an experiment
    experiment = GraniteExperiment(save_dir=save_dir, logger=logger)

    # logger.log("=" * 50)
    # logger.log(f"Created a new Experiment. Model {llm_name}")
    # logger.log("=" * 50)

    # for k, v in args.__dict__.items():
    #     logger.log(f">>>> Command line argument {k} => {v}")
    # logger.log("=" * 50)

    # Step 5: Read the dataset
    # dataset_util = FEVER()
    # dataset = dataset_util.get_dataset(logger)
    dataset = None

    # Step 6: Run intervention
    experiment.intervene(model=model,
                         tokenizer=tokenizer,
                         dataset=dataset,
                         args=args,
                         llm_name=llm_name)

    logger.log("Experimented Completed.")
