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
        # 分析結果格納用辞書
        baseline_result = {}
        proposed_result = {}

        analyzed_line = {
            'original':   line['original'],
            'transfered': line['transfered'],
            'references': line['references'],
            'baseline': baseline_result,
            'proposed': proposed_result
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
        baseline_blue = sentence_bleu([ original_tokens_list], transferred_tokens_list)
        proposed_bleu = sentence_bleu([reference_tokens_list], transferred_tokens_list)
        analyzed_line['baseline']['BLEU'] = baseline_bleu_score
        analyzed_line['proposed']['BLEU'] = proposed_bleu_score

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        baseline_rouge_scores = scorer.score(reference_tokens,    original_tokens)
        proposed_rouge_scores = scorer.score(reference_tokens, transferred_tokens)
        analyzed_line['baseline']['ROUGE'] = {
           'ROUGE-1': baseline_rouge_scores['rouge1'],
           'ROUGE-2': baseline_rouge_scores['rouge2'],
           'ROUGE-L': baseline_rouge_scores['rougeL'],
        }
        analyzed_line['proposed']['ROUGE'] = {
           'ROUGE-1': proposed_rouge_scores['rouge1'],
           'ROUGE-2': proposed_rouge_scores['rouge2'],
           'ROUGE-L': proposed_rouge_scores['rougeL'],
        }

        # METEOR
        baseline_meteor = meteor_score([reference_tokens_list],    original_tokens_list)
        proposed_meteor = meteor_score([reference_tokens_list], transferred_tokens_list)
        analyzed_line['baseline']['METEOR'] = baseline_meteor
        analyzed_line['proposed']['METEOR'] = proposed_meteor

        # TER
        ter = TER()
        baseline_ter_score = ter.sentence_score(   original_tokens, [reference_tokens]).score
        proposed_ter_score = ter.sentence_score(transferred_tokens, [reference_tokens]).score
        analyzed_line['baseline']['TER'] = baseline_ter_score
        analyzed_line['proposed']['TER'] = proposed_ter_score

        # BERTScore（日本語対応）
        baseline_P, baseline_R, baseline_F1 = bert_score([   original_tokens], [reference_tokens], lang='ja')
        proposed_P, proposed_R, proposed_F1 = bert_score([transferred_tokens], [reference_tokens], lang='ja')
        analyzed_line['baseline']['BERTScore'] = {
            'score': float(baseline_F1),
            'P':     float(baseline_P),
            'R':     float(baseline_R)
        }
        analyzed_line['proposed']['BERTScore'] = {
            'score': float(proposed_F1),
            'P':     float(proposed_P),
            'R':     float(proposed_R)
        }

        # ChrF
        chrf= CHRFScore()
        baseline_chrf_score = chrf([   original_tokens], [reference_tokens])
        proposed_chrf_score = chrf([transferred_tokens], [reference_tokens])
        analyzed_line['baseline']['ChrF'] = float(baseline_chrf_score)
        analyzed_line['proposed']['ChrF'] = float(proposed_chrf_score)

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

