# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning

This repository contains the code and experiments from our paper:

**[Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning](https://arxiv.org/abs/2504.07097)**  
Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Hao Wang, Aldo Pareja, Oleg Silkin, Mustafa Eyceoz, Akash Srivastava

📖 [Read the paper](https://arxiv.org/abs/2504.07097)  
🧠 [Blog post](https://ai-innovation.team/blog/orthogonal-subspace-learning)

---

## 🚀 Overview

Continual learning in large language models (LLMs) remains a core challenge for real-world deployment: models must incorporate new capabilities without forgetting prior expertise. Our method—**Adaptive SVD**—enables **full-parameter continual fine-tuning** by constraining updates to **low-rank subspaces orthogonal** to previously learned knowledge.

Key contributions:

- Strong retention of prior performance and general capabilities.
- Outperforms state-of-the-art baselines across multiple continual learning benchmarks.
- No extra parameters per task — fixed model size.

---

## 🗂️ Repository Structure

- `/notebooks/`:  
  notebooks for finetuning T5, LLaMA-2, and LLaMA-3.1 on continual learning tasks using Adaptive SVD.
  
- `/llama-auto-gpu-code/` and `/llama-single-gpu-code/`:  
  Training scripts and configs for full fine-tuning LLaMA-2 7B and 3.1 8B models with SVD and projected gradient descent on multiple GPUs and single GPU respectively.

- `/plotting_code/`:  
  Scripts to reproduce experiment plots from the paper and blog post.

- `/svd/`:  
  Implementation of **projected gradient descent** for constrained subspace optimization.

- `/src/`:  
  Source modules for task-level interventions and dataset handling, including older experiments on Granite models.

- `/scripts/`:  
  Utility scripts for running multiple jobs, loading dynamic matrix projections, and composing subspace-based adapters.

---

## 🧪 Running Experiments

Example: Fine-tune a T5-large model on 15-task continual learning benchmark sequence:

```bash
cd notebooks
bash run_finetuning.sh
