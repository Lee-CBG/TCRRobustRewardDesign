# Mitigating Goodhart's Law in Epitope-Conditioned TCR Generation Using Plug-and-Play Reward Designs

This repository provides the full experimental codebase, evaluation pipeline, and figures accompanying our RECOMB submission **“Mitigating Goodhart's Law in Epitope-Conditioned TCR Generation Using Plug-and-Play Reward Designs.”**  

---

## 📜 Abstract

Epitope-conditioned T cell receptor (TCR) generation extends protein language modeling to therapeutic design of TCRs, with recent works demonstrating improved performance using reinforcement learning (RL) post-training utilizing surrogate models. However, such frameworks suffer from Goodhart’s Law: optimizing imperfect surrogate rewards leads to reward hacking and biologically invalid sequences. We present a plug-and-play reward design framework that mitigates reward hacking without altering the generator and its training pipeline. The framework integrates multiple complementary strategies: heuristic biological priors to reject degenerate sequences, model ensembling to reduce per-model biases, and binding-specificity objectives to suppress cross-epitope overfitting and promote target-focused binding behavior. When applied to the RL-based fine-tuning process of TCR generators, these rewards stabilize optimization, preserve sequence diversity, and yield generations that are more biologically aligned with real TCR sequences. The resulting models produce more authentic epitope-specific receptors, demonstrating that our Goodhart-resistant reward design substantially improves reliability and controllability in biologically grounded sequence generation.


---

## 📊 Main Results

Our reward hacking mitigation strategies improve RL stability, reduce unrealistic reward exploitation, and preserve biologically meaningful structures in the generated TCRs.


---

### **Our Plug-and-Play Reward Design Framework for Epitope-Conditioned TCR Generation**

<p align="center">
  <img src="figs/method.png" width="750"/>
  <br/>
  <em>
  Overview of the plug-and-play reward design framework for epitope-conditioned TCR generation. 
  Reward components include heuristic priors, surrogate binding rewards, ensemble stabilization, 
  max-margin epitope specificity, and InfoNCE contrastive losses. Together, these modules reduce 
  reward hacking and stabilize RL-based sequence generation.
  </em>
</p>

---

### **Figure 2 — Reward Drift and Biological Evaluation Metrics**

<p align="center">
  <img src="figs/mitigation.png" width="850"/>
  <br/>
  <em>
  Reward drift and biological authenticity metrics across training. 
  The baseline RL model exhibits reward inflation and degraded biological scores 
  (TCRMatch similarity, GPT log-likelihood, TCR-BERT log-likelihood). 
  Our reward designs maintain stable authenticity and moderate rewards, 
  demonstrating reduced Goodhart failure and improved robustness.
  </em>
</p>

---

### **Figure 3 — Reward Designs Stabilize Distributional Trajectories**

<p align="center">
  <img src="figs/trajectory.png" width="850"/>
  <br/>
  <em>
  Centroid trajectories of generated TCR embeddings across training for multiple epitopes. 
  Baseline RL collapses toward unrealistic or off-manifold regions, while our reward 
  designs maintain stable trajectories that remain anchored to the ground-truth 
  TCR distribution. Mitigation strategies significantly improve distributional stability 
  and epitope-specific structure.
  </em>
</p>

---

### **Figure 4 — Early-Stopped vs Fully Trained Models**

<p align="center">
  <img src="figs/es_vs_full_training.png" width="850"/>
  <br/>
  <em>
  Comparison of early-stopped baseline RL and fully-trained models with reward hacking 
  mitigation. Fully trained models, when equipped with stabilization strategies, 
  maintain biological diversity, authenticity, and binding properties across 28 
  unseen epitopes. This illustrates that delaying failure enables stable learning 
  and better generalization.
  </em>
</p>

---





### 🔹 Figure 1: Mitigation Improves RL Stability

<p align="center">
  <img src="figs/training_stability.png" width="650"/>
  <br/>
  <em>Mitigation strategies suppress degenerate behavior and stabilize reward trajectories during RL training.</em>
</p>

---

### 🔹 Figure 2: Effects on TCR Diversity and Distribution

<p align="center">
  <img src="figs/sequence_diversity.png" width="650"/>
  <br/>
  <em>Compared to naive RL optimization, mitigation strategies maintain epitope-specific diversity and prevent mode collapse.</em>
</p>

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your_org/your_repo.git
cd your_repo

conda create -n tcr_env python=3.10 -y
conda activate tcr_env

pip install -r requirements.txt

```







# Mitigating Reward Hacking in TCR Sequence Generation  
Code Release for RECOMB Submission

This repository contains the full codebase used in our RECOMB submission **“Mitigating Reward Hacking in TCR Sequence Generation”**.  
It includes:

- Reinforcement learning training code  
- Reward model definitions and scoring functions  
- Reproducible evaluation pipeline  
- Data used in experiments  
- Figures for the paper  





---

## 🧬 Overview

We study **reward hacking** in TCR sequence generation models trained with reinforcement learning.  
To prevent models from exploiting degenerate heuristics, we implement:

- **Reward-model smoothing**  
- **Ensemble rewards**  
- **Distribution-level regularization**  


## 🚀 Installation

```bash
git clone https://github.com/<lab>/<repo>.git
cd <repo>

# Recommended: use conda
conda create -n tcr_env python=3.10 -y
conda activate tcr_env

pip install -r requirements.txt
```

