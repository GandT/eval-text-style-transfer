import MeCab
import nltk
try:
    # synsets を呼ぶことで wordnet があるか確認
    from nltk.corpus import wordnet
    _ = wordnet.synsets("テスト")
except LookupError:
    nltk.download('wordnet')

from nltk.translate.bleu_score   import sentence_bleu
from rouge_score                 import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sacrebleu                   import TER
from bert_score                  import score as bert_score
from torchmetrics.text           import CHRFScore

import os

# 分かち書き
def tokenize_japanese(text):
    mecab = MeCab.Tagger('-Owakati')
    return mecab.parse(text).strip()

# 日本語サンプルデータ
original_text    = os.environ['ORIG_T']
transferred_text = os.environ['TRAN_T']
reference_text   = os.environ['REFE_T']

# 分かち書き
original_tokens    = tokenize_japanese(original_text)
transferred_tokens = tokenize_japanese(transferred_text)
reference_tokens   = tokenize_japanese(reference_text)

original_tokens_list    =    original_tokens.split()
transferred_tokens_list = transferred_tokens.split()
reference_tokens_list   =   reference_tokens.split()

# BLEU
bleu_score = sentence_bleu([reference_tokens_list], transferred_tokens_list)
print(f"BLEU: {bleu_score:.4f}")

# ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
rouge_scores = scorer.score(reference_tokens, transferred_tokens)
print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")

# METEOR
meteor = meteor_score([reference_tokens_list], transferred_tokens_list)
print(f"METEOR: {meteor:.4f}")

# TER
ter = TER()
ter_score = ter.sentence_score(transferred_tokens, [reference_tokens]).score
print(f"TER: {ter_score:.4f}")

# BERTScore（日本語対応）
P, R, F1 = bert_score([transferred_tokens], [reference_tokens], lang='ja')
print(f"BERTScore (F1): {F1.mean().item():.4f}")
print(f"P: {P}")
print(f"R: {R}")

# ChrF
chrf= CHRFScore()
chrf_score = chrf([transferred_tokens], [reference_tokens])
print(f"ChrF: {chrf_score:.4f}")