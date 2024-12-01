import pandas as pd
from datasets import Dataset

from tranhack.const import clean_text


def apply_clean_text(item):
    return {'name': clean_text(item['name'])}


def _drop_bad_data(item):
    if len(item['name']) < 6:
        return False
    if len(item['name']) / len(item['translation']) <= 0.47:
        return False
    if len(item['name']) / len(item['translation']) >= 2.5:
        return False
    return True


def load_data():
    df = pd.read_excel('data/train.xlsx')
    ds = Dataset.from_pandas(df)
    ds = ds.map(apply_clean_text, num_proc=4)
    ds = ds.filter(_drop_bad_data, num_proc=4)
    ds = ds.train_test_split(test_size=0.05, seed=0xBABABA)
    ds['train'].save_to_disk('data/train')
    ds['test'].save_to_disk('data/test')


if __name__ == '__main__':
    load_data()