
# PELMS: Pre-training for Effective Low-Shot Multi-Document Summarization

  

This repository contains the accompanying code & resources for the paper:

**PELMS: Pre-training for Effective Low-Shot Multi-Document Summarization."** Joseph J. Peper, Wenzhao Qiu, Lu Wang. *NAACL 2024 Main Conference* [[pdf]](https://arxiv.org/abs/2311.09836).

  

The contributions of our work include:

1)  **PELMS**: A new pre-training objective and pre-trained model for multi-document summarization.

2)  **MultiPT**: A new large-scale pre-training corpora comprising over 6 Million topi-aligned document clusters.

3)  **MetaTomatoes**: A new multi-document summarization dataset containing paragraph-length meta-summaries of editorial movie reviews.

  

## Updates
**March 2024**: PELMS is accepted to NAACL 2024 as a main conference long paper.
**November 2023:** PELMS is uploaded to arXiv and pre-trained model and code are made available.


  

## PELMS Pre-training

PELMS consists of a novel pre-training objective aimed at improving zero-shot and few-shot performance in conventional pre-trained language models.

  

### PELMS Pre-training Technique

Our three-step pre-training target formulation uses 1) embedding-based clustering to identify prevalent information within the input, 2) faithfullness-oriented ranking process for identifying candidate sentences and 3) forms a pre-training objective that promotes coherent and abstractive outputs.

<img  width="1733"  alt="image"  src="https://github.com/jpeper/MDS_2023/assets/22947125/d42ec8ba-8134-4f92-9e83-227df3a7378b">

  

### Instructions

We train and release our PELMS model trained on the MultiPT pre-training dataset with a base Longformer Encoder-Decoder (LED) architecture.

You can either pre-train yourself or use our available pretrained weights available on HuggingFace (TODO add actual Link).

  

#### Usage for pre-trained Huggingface model:

```python

from transformers import AutoTokenizer, AutoModel

# load model and tokenizer

tokenizer = AutoTokenizer.from_pretrained('jpeper/PELMS')

model = AutoModel.from_pretrained('jpeper/PELMS')

```

**Note:** Pre-training requires significant GPU compute. In our work, we perform distributed training using 16 NVIDIA A40 GPUs, each with 48GB VRAM. This process takes approximately 5 days to complete. Pre-training code will be available soon.

---

## Datasets
You can request access to our new MultiPT and MetaTomatoes datasets by completing [this form](https://forms.gle/XvtB9uvZyrpHp9YV7).

## Citation

If you use this code, models, or datasets, please consider citing our work:

 

```bibtex

@inproceedings{peper-etal-2024-pelms,
    title = "{PELMS}: Pre-training for Effective Low-Shot Multi-Document Summarization",
    author = "Peper, Joseph  and
      Qiu, Wenzhao  and
      Wang, Lu",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.423",
    pages = "7645--7667",
    abstract = "We investigate pre-training techniques for abstractive multi-document summarization (MDS), which is much less studied than summarizing single documents. Though recent work has demonstrated the effectiveness of highlighting information salience for pre-training strategy design, they struggle to generate abstractive and reflective summaries, which are critical properties for MDS. To this end, we present **PELMS**, a pre-trained model that uses pre-training objectives based on semantic coherence heuristics and faithfulness constraints together with unlabeled multi-document inputs, to promote the generation of concise, fluent, and faithful summaries. To support the training of PELMS, we compile **MultiPT**, a multi-document pre-training corpus containing over 93 million documents to form more than 3million unlabeled topic-centric document clusters, covering diverse genres such as product reviews, news, and general knowledge. We perform extensive evaluation of PELMS in low-shot settings on a wide range of MDS datasets. Our approach consistently outperforms competitive comparisons with respect to overall informativeness, abstractiveness, coherence, and faithfulness, and with minimal fine-tuning can match performance of language models at a much larger scale (e.g., GPT-4).",
}
```
