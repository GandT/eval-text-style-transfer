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

import json

# 分かち書き
def tokenize_japanese(text):
    mecab = MeCab.Tagger('-Owakati')
    return mecab.parse(text).strip()

def analyze_text(json_input):
    result_list = []

    for line in json_input:
        analyzed_line = {
            'original':   line['original'],
            'transfered': line['transfered'],
            'references': line['references']
        }

        # 分かち書き
        original_tokens    = tokenize_japanese(analyzed_line['original'])
        transferred_tokens = tokenize_japanese(analyzed_line['transfered'])
        reference_tokens   = tokenize_japanese(analyzed_line['references'])
        
        # リスト化
        original_tokens_list    =    original_tokens.split()
        transferred_tokens_list = transferred_tokens.split()
        reference_tokens_list   =   reference_tokens.split()

        # BLEU
        bleu_score = sentence_bleu([reference_tokens_list], transferred_tokens_list)
        analyzed_line['BLEU'] = bleu_score

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = scorer.score(reference_tokens, transferred_tokens)
        analyzed_line['ROUGE'] = {
           'ROUGE-1': rouge_scores['rouge1'],
           'ROUGE-2': rouge_scores['rouge2'],
           'ROUGE-L': rouge_scores['rougeL'],
        }

        # METEOR
        meteor = meteor_score([reference_tokens_list], transferred_tokens_list)
        analyzed_line['METEOR'] = meteor

        # TER
        ter = TER()
        ter_score = ter.sentence_score(transferred_tokens, [reference_tokens]).score
        analyzed_line['TER'] = ter_score

        # BERTScore（日本語対応）
        P, R, F1 = bert_score([transferred_tokens], [reference_tokens], lang='ja')
        analyzed_line['BERTScore'] = {
            'score': float(F1),
            'P':     float(P),
            'R':     float(R)
        }

        # ChrF
        chrf= CHRFScore()
        chrf_score = chrf([transferred_tokens], [reference_tokens])
        analyzed_line['ChrF'] = float(chrf_score)

        result_list.append(analyzed_line)

    return result_list


def main():
    input_path = "sample_input.json"
    output_path = "sample_input_evaluated.json"

    # --- 入力の読み込み（JSON → リスト）---
    with open(input_path, "r", encoding="utf-8") as file_input:
        # reader はイテレータなのでリスト内包表記で全レコードを取得
        json_input = json.load(file_input)

    # --- 分析関数に渡して評価結果を取得 ---
    results = analyze_text(json_input)

    # --- 出力の書き出し（リスト → JSON）---
    with open(output_path, "w", encoding="utf-8") as file_output:
        # ensure_ascii=False で日本語をエスケープせず、indent=2 で見やすく整形
        json.dump(results, file_output, ensure_ascii=False, indent=2)


if __name__ == '__main__':
  main()

