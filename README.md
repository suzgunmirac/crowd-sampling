# Crowd Sampling
#### _Follow the Wisdom of the Crowd_: Effective Text Generation via Minimum Bayes Risk Decoding
— by Mirac Suzgun, Luke Melas-Kyriazi, Dan Jurafsky

* [![arXiv](https://img.shields.io/badge/arXiv-2211.07634-b31b1b.svg)](https://arxiv.org/abs/2211.07634)
 [Paper Link](https://arxiv.org/abs/2211.07634)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RmMDb2eE_Yc8oz3kpUsNzDcPfyb39_qH?usp=sharing) [ Colab Notebook](https://colab.research.google.com/drive/1RmMDb2eE_Yc8oz3kpUsNzDcPfyb39_qH?usp=sharing)

## Table of Contents
1. [Abstract](#abstract)
2. [Google Colab Notebook](#google-colab-notebook)
3. [Overview of Crowd Sampling](#overview-of-crowd-sampling)
4. [How to Use MBRD](#how-to-use-mbrd)
5. [Candidate Selection Strategies](#candidate-selection-strategies)
6. [Datasets, Prompts, and Outputs](#datasets-prompts-and-outputs)
7. [Main Results](#main-results)
8. [Additional Information](#additional-information)
9. [Citation](#citation)

## Abstract
In open-ended natural-language generation, existing text decoding methods typically struggle to produce text which is both diverse and high-quality. Greedy and beam search are known to suffer from text degeneration and linguistic diversity issues, while temperature, top-k, and nucleus sampling often yield diverse but low-quality outputs. In this work, we present **crowd sampling**, a family of decoding methods based on Bayesian risk minimization, to address this diversity-quality trade-off. Inspired by the principle of "the wisdom of the crowd," crowd sampling seeks to select a candidate from a pool of candidates that has the least expected risk (i.e., highest expected reward) under a generative model according to a given utility function. Crowd sampling can be seen as a generalization of numerous existing methods, including majority voting, and in practice, it can be used as a drop-in replacement for existing sampling methods. Extensive experiments show that crowd sampling delivers improvements of 3-7 ROUGE and BLEU points across a wide range of tasks, including summarization, data-to-text, translation, and textual style transfer, while achieving new state-of-the-art results on WebNLG and WMT'16.

## Google Colab Notebook
You can use the following Google Colab notebook to test our crowd sampling method.
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RmMDb2eE_Yc8oz3kpUsNzDcPfyb39_qH?usp=sharing) [ Colab Notebook](https://colab.research.google.com/drive/1RmMDb2eE_Yc8oz3kpUsNzDcPfyb39_qH?usp=sharing)

## Overview of Crowd Sampling
Given a collection of candidates $\mathcal{C}$, crowd sampling seeks to choose the candidate $\mathbf{c} \in \mathcal{C}$ which maximizes the sum of alignments with the whole crowd, that is:

[<img src="https://github.com/suzgunmirac/crowd-sampling/blob/main/fables/mbrd-equation.png" width="350" class="center"/>](https://github.com/suzgunmirac/crowd-sampling/blob/main/fables/mbrd-equation.png)

**Illustration of our *Crowd Sampling* method based on Minimum Risk Bayes Decoding:**

Given an input prompt, we first generate multiple candidate texts (outputs) using a stochastic sampling method such as temperature sampling under a generative language model. We then compare each candidate with the other candidates using a utility (i.e., alignment) function such as BERTScore and compute the overall alignment of each candidate with respect to others. Finally, we pick the candidate with the highest computed sum as the final output. This simple but effective meta-sampling/decoding method yields significant ROUGE and BLEU score improvements over standard text decoding methods across a wide range of NLG tasks and benchmarks

![Crowd-Sampling](https://github.com/suzgunmirac/crowd-sampling/blob/main/fables/crowd-sampling.png)


## How to Use MBRD
Crowd sampling can be used as a drop-in replacement for existing text decoding methods to improve text generation across a wide range of NLP settings. Let's show this through an illustrative example.

Temperature sampling, for instance, is typically used in the following form to generate text under a generative language model:
```python
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids, do_sample=True, temperature=temperature, max_length=128)
final_output = tokenizer.decode(outputs[0])

print(f'Final output: {decoded_output}')
```
Here we generate only one output candidate and then return it. 

However, we can instead generate multiple candidates and choose the one that "aligns" the most with the whole crowd according to a utility/alignment function such as BERTScore.
```python
k_samples = 16

# Let's sample K candidate outputs. 
candidates = []
for i in trange(k_samples):
    input_id = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, do_sample=True, temperature=temperature, max_length=128)
    decoded_output = tokenizer.decode(output.squeeze(0), skip_special_tokens=True)
    candidates.append(decoded_output)

import numpy as np
import torch
import datasets

# Load BERTScore metric
bertscore = datasets.load_metric("bertscore")

# Score candidates by MBRD criterion with BERTScore
score_matrix = np.zeros((k_samples, k_samples))
for j1, cand1 in enumerate(candidates):
    for j2, cand2 in enumerate(candidates):
        if j1 < j2:
            score = bertscore.compute(predictions=[cand1], references=[cand2], lang='en')['f1'][0]
            score_matrix[j1][j2] = score_matrix[j2][j1] = score

# Compute candidate with maximum score
sum_scores = np.sum(score_matrix, axis=1)
index = np.argmax(sum_scores)
final_output = candidates[index]

print(f'Final output: {final_output}')
```

## Candidate Selection Strategies
Here we elucidate the main text decoding and candidate selection strategies used in our experiments.

(a) **Sample-Once**: We generate a single output candidate using temperature sampling with $\tau=0.7$. Note that Sample-Once is the de facto sampling/decoding choice for many large language models.

(b) **Random**: We first generate $k$ output candidates using temperature sampling and then randomly select one of the $k$ candidates as the final output.

(c) **Majority Voting**: Like before, we first generate $k$ output candidates using temperature sampling with the same temperature value and then select the most common candidate in the sample pool.

(d) **MBRD-BLEURT**: We first generate $k$ output candidates and then select the final output according to the MBRD method with BLEURT as the utility/alignment function.

(e) **MBRD-BERTScore**: It is the same as (d), but uses BERTScore as its utility/alignment function.


## Datasets, Prompts, and Outputs
For convenience, we include the datasets, prompts, outputs, and ground-truth references used in our experiments. Please cite the original papers if you decide to use them in your research.
- Input files (data): `/data`.
- Prompts: `/prompts`.
- Codex outputs: `/outputs`.
- Ground-truth references: `/ground_truth`.

## Main Results
![Main-Results](https://github.com/suzgunmirac/crowd-sampling/blob/main/fables/main-results.png)


## Additional Information
Please make sure to install the following packages and libraries to be able to use our code.
```bash
pip install transformers
pip install accelerate
pip install bert-score
pip install sentencepiece
pip install datasets
pip install -i https://test.pypi.org/simple/ string2string
```

## Citation
If your research makes use of our data, code, or results, please consider citing our paper.
```bibtex
@article{suzgun2022crowdsampling,
    title={Follow the Wisdom of the Crowd: Effective Text Generation via Minimum Bayes Risk Decoding}, 
    author={Mirac Suzgun and Luke Melas-Kyriazi and Dan Jurafsky},
    year={2022},
    journal={arXiv preprint arXiv:2211.07634},
    url={https://arxiv.org/abs/2211.07634}
}
```

□ Q.E.D.