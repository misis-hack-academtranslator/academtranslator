import pandas as pd
from torchmetrics.text import BLEUScore, BERTScore

if __name__ == '__main__':
    df = pd.read_excel('data/eval.xlsx')
    bleu_1 = BLEUScore(n_gram=1)
    bleu_2 = BLEUScore(n_gram=2)
    bleu_3 = BLEUScore(n_gram=3)
    bleu_4 = BLEUScore(n_gram=4)
    bertscore = BERTScore(model_name_or_path="microsoft/deberta-xlarge-mnli", device='cuda:0')
    bleu_1.update(df['translation_gen'], [[x] for x in df['translation_orig']])
    bleu_2.update(df['translation_gen'], [[x] for x in df['translation_orig']])
    bleu_3.update(df['translation_gen'], [[x] for x in df['translation_orig']])
    bleu_4.update(df['translation_gen'], [[x] for x in df['translation_orig']])
    bertscore.update(df['translation_gen'], df['translation_orig'])
    print('BLEU-1', bleu_1.compute(), 'BLEU-2', bleu_2.compute(), 'BLEU-3', bleu_3.compute(), 'BLEU-4', bleu_4.compute())
    bert_score = bertscore.compute()
    print('BERT Score', 'F1', bert_score['f1'].mean(), 'Precision', bert_score['precision'].mean(), 'Recall', bert_score['recall'].mean())


# BLEU по триграммам 52
# BERT Score по мнению deberta-xlarge-mnli F1 tensor(0.6496) Precision tensor(0.6504) Recall tensor(0.6503)