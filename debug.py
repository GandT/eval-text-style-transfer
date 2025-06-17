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

def convert_tokenized_text_to_id_string(tokenized_text: str) -> str:
    """
    トークナイズ済みの日本語文字列（単語が空白区切り）を受け取り、
    各単語にユニークな整数IDを割り振った上で
    空白区切りのID文字列として返す。
    """
    # 1. 空白で分割してトークン（単語）のリストを作成
    tokens = tokenized_text.split()

    # 2. 単語→ID のマッピングを保持する辞書を用意
    token_to_id = {}
    next_id = 1

    # 3. 各トークンを走査し、IDを決定
    id_list = []
    for token in tokens:
        # まだ見たことのない単語なら、新しいIDを割り振る
        if token not in token_to_id:
            token_to_id[token] = next_id
            next_id += 1
        # マッピング済みのIDをリストに追加
        id_list.append(str(token_to_id[token]))

    # 4. IDリストを空白で結合して返却
    return " ".join(id_list)

def analyze_text(json_input):
    result_list = []

    input_lengh = len(json_input)
    bleu_base_sum = 0
    bleu_prop_sum = 0
    rouge_base_sum = 0
    rouge_prop_sum = 0
    meteor_base_sum = 0
    meteor_prop_sum = 0
    ter_base_sum = 0
    ter_prop_sum = 0
    bert_base_sum = 0
    bert_prop_sum = 0
    chrf_base_sum = 0
    chrf_prop_sum = 0

    for line in json_input:
        # 分析結果格納用辞書
        baseline_result = {}
        proposed_result = {}

        analyzed_line = {
            'original':    line['original'],
            'transferred': line['transferred'],
            'reference':   line['reference'],
            'baseline': baseline_result,
            'proposed': proposed_result
        }

        # 分かち書き
        original_tokens    = tokenize_japanese(analyzed_line['original'])
        transferred_tokens = tokenize_japanese(analyzed_line['transferred'])
        reference_tokens   = tokenize_japanese(analyzed_line['reference'])
        
        # リスト化
        original_tokens_list    =    original_tokens.split()
        transferred_tokens_list = transferred_tokens.split()
        reference_tokens_list   =   reference_tokens.split()

        # BLEU
        baseline_bleu = sentence_bleu([ original_tokens_list], transferred_tokens_list)
        proposed_bleu = sentence_bleu([reference_tokens_list], transferred_tokens_list)
        analyzed_line['baseline']['BLEU'] = baseline_bleu
        analyzed_line['proposed']['BLEU'] = proposed_bleu
        bleu_base_sum += baseline_bleu
        bleu_prop_sum += proposed_bleu

        # ROUGE
        original_ids   = convert_tokenized_text_to_id_string(   original_tokens)
        transfered_ids = convert_tokenized_text_to_id_string(transferred_tokens)
        reference_ids  = convert_tokenized_text_to_id_string(  reference_tokens)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        baseline_rouge_scores = scorer.score(reference_ids,   original_ids)
        proposed_rouge_scores = scorer.score(reference_ids, transfered_ids)
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
        rouge_base_sum += baseline_rouge_scores['rouge1'].fmeasure
        rouge_prop_sum += proposed_rouge_scores['rouge1'].fmeasure

        # METEOR
        baseline_meteor = meteor_score([reference_tokens_list],    original_tokens_list)
        proposed_meteor = meteor_score([reference_tokens_list], transferred_tokens_list)
        analyzed_line['baseline']['METEOR'] = baseline_meteor
        analyzed_line['proposed']['METEOR'] = proposed_meteor
        meteor_base_sum += baseline_meteor
        meteor_prop_sum += proposed_meteor

        # TER
        ter = TER()
        baseline_ter_score = ter.sentence_score(   original_tokens, [reference_tokens]).score
        proposed_ter_score = ter.sentence_score(transferred_tokens, [reference_tokens]).score
        analyzed_line['baseline']['TER'] = baseline_ter_score
        analyzed_line['proposed']['TER'] = proposed_ter_score
        ter_base_sum += baseline_ter_score
        ter_prop_sum += proposed_ter_score

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
        bert_base_sum += float(baseline_F1)
        bert_prop_sum += float(proposed_F1)

        # ChrF
        chrf= CHRFScore()
        baseline_chrf_score = chrf([   original_tokens], [reference_tokens])
        proposed_chrf_score = chrf([transferred_tokens], [reference_tokens])
        analyzed_line['baseline']['ChrF'] = float(baseline_chrf_score)
        analyzed_line['proposed']['ChrF'] = float(proposed_chrf_score)
        chrf_base_sum += float(baseline_chrf_score)
        chrf_prop_sum += float(proposed_chrf_score)

        result_list.append(analyzed_line)

    summary = {
        'BLEU': {
            'baseline': bleu_base_sum / input_lengh,
            'proposed': bleu_prop_sum / input_lengh,
        },
        'ROUGE': {
            'baseline': rouge_base_sum / input_lengh,
            'proposed': rouge_prop_sum / input_lengh,
        },
        'METEOR': {
            'baseline': meteor_base_sum / input_lengh,
            'proposed': meteor_prop_sum / input_lengh,
        },
        'TER': {
            'baseline': ter_base_sum / input_lengh,
            'proposed': ter_prop_sum / input_lengh,
        },
        'BERTScore': {
            'baseline': bert_base_sum / input_lengh,
            'proposed': bert_prop_sum / input_lengh,
        },
        'ChrF': {
            'baseline': chrf_base_sum / input_lengh,
            'proposed': chrf_prop_sum / input_lengh,
        }
    }

    return result_list, summary


def main():
    input_path   = "sentence.json"
    output_path  = "sentence_evaluated.json"
    summary_path = 'summary.json'

    # --- 入力の読み込み（JSON → リスト）---
    with open(input_path, "r", encoding="utf-8") as file_input:
        # reader はイテレータなのでリスト内包表記で全レコードを取得
        json_input = json.load(file_input)

    # --- 分析関数に渡して評価結果を取得 ---
    results, summary = analyze_text(json_input)

    # --- 出力の書き出し（リスト → JSON）---
    with open(output_path, "w", encoding="utf-8") as file_output:
        # ensure_ascii=False で日本語をエスケープせず、indent=2 で見やすく整形
        json.dump(results, file_output, ensure_ascii=False, indent=2)

    # --- 結果概要の書き出し（辞書 → JSON）---
    with open(summary_path, "w", encoding="utf-8") as file_output:
        # ensure_ascii=False で日本語をエスケープせず、indent=2 で見やすく整形
        json.dump(summary, file_output, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()

