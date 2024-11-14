# Cross-Domain-Evaluation-for-Multi-Task-Learning-in-NLP
Cross-Domain Evaluation for Multi-Task Learning in NLP: A Unified Framework for Generalization and Robustness
Overview
This repository provides the code and resources for our research on enhancing model robustness and generalization in Natural Language Processing (NLP) through cross-domain evaluation and multi-task learning. Our approach combines three distinct datasets—WikiText-103, OpenWebText, and DROP—to evaluate and improve generalization across language modeling, conversational text processing, and complex reasoning tasks.

Key Contributions
Cross-Domain Evaluation Framework: A unified evaluation strategy to assess model performance across different domains and tasks.
Multi-Task Learning for Improved Generalization: By training models on diverse datasets, we show enhanced robustness and adaptability.
Empirical Insights: Comparative analysis of single-task and multi-task models, demonstrating performance improvements in unseen domains.

Project Structure
├── data/                   # Dataset storage (sample files or links to download full datasets)
├── notebooks/              # Jupyter notebooks for data processing and model training
├── src/                    # Source code for model architecture and evaluation functions
├── results/                # Outputs including evaluation metrics, figures, and logs
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies


Datasets
We use the following datasets for training and evaluation:

WikiText-103: Large-scale structured text dataset for lan guage modeling.
OpenWebText: Informal conversational text dataset, based on Reddit data.
DROP: Reasoning-focused dataset for evaluating models on logical and numerical questions.
Download links and preprocessing steps are in the from torchtext.datasets import WikiText103
train_iter, val_iter, test_iter = WikiText103()
from datasets import load_dataset
openwebtext = load_dataset("openwebtext")
drop = load_dataset("drop", "default")


Setup and Installation
1. Clone this repository: git clone https://github.com/farjana-yesmin/Cross-Domain-Evaluation-for-Multi-Task-Learning-in-NLP.git
2. Install dependencies: pip install -r requirements.txt
3. Download the datasets and place them in the data/ folder.
   Usage
Data Preprocessing
Run the preprocessing scripts to tokenize and format data for model training:
python src/preprocess.py
Training
To train the model in multi-task or single-task mode: python src/train.py --mode multi-task
Evaluation
Run evaluations on the selected datasets to generate results for cross-domain analysis: python src/evaluate.py --dataset DROP
Visualization
Use the notebook in notebooks/ to visualize training results and performance metrics: jupyter notebook notebooks/visualize_results.ipynb
Results
Our model demonstrates improved generalization and robustness across WikiText-103, OpenWebText, and DROP datasets.
Comparative plots of single-task vs. multi-task performance on each dataset can be found in the results/ directory.
Figures
Token Length Distribution

Perplexity Improvement on WikiText-103 and OpenWebText
Citation
If you use this framework, please cite our paper:

Farjana Yesmin, Cross-Domain Evaluation for Multi-Task Learning in NLP: A Unified Framework for Generalization and Robustness. Boise State University.

