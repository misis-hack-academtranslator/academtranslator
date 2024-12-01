import click
import datasets
import pandas as pd
from datasets import Dataset
from vllm import LLM, SamplingParams

from tranhack.const import make_prompt
from tranhack.prepare_data import apply_clean_text


class LLMGenerator:
    def __init__(self):
        llm = LLM(
            model='checkpoint/train-alma/merged',
            tokenizer='checkpoint/train-alma/merged',
            trust_remote_code=True,
            dtype='bfloat16',
            max_model_len=2048,
            gpu_memory_utilization=0.8
        )

        params = SamplingParams(
            best_of=1,
            n=1,
            temperature=0.0,
            top_p=0.9,
            ignore_eos=False,
            max_tokens=1024,
        )

        self._llm = llm
        self._params = params

    def generate(self, items):
        prompts = [make_prompt(x['name'], x['faculty'], x['direction']) for x in items]
        results = self._llm.generate(prompts, self._params)
        results = [x.outputs[0].text.strip() for x in results]
        return results


@click.command()
@click.option('--mode', type=click.Choice(['test', 'eval']), required=True)
def main(mode: str):
    if mode == 'eval':
        data = datasets.load_from_disk('data/test')
        llm = LLMGenerator()
        outs = llm.generate(data)
        pd.DataFrame({'translation_gen': outs, 'translation_orig': data['translation']}).to_excel('data/eval.xlsx',
                                                                                                  index=False)
    elif mode == 'test':
        df = pd.read_excel('data/test.xlsx')
        df['orig_name'] = df['name']
        ds = Dataset.from_pandas(df)
        ds = ds.map(apply_clean_text, num_proc=4)
        llm = LLMGenerator()
        outs = llm.generate(ds)
        pd.DataFrame({'name': ds['orig_name'], 'faculty': ds['faculty'], 'direction': ds['direction'],
                      'translation': outs}).to_excel('data/test_gen.xlsx',
                                                     index=False)


if __name__ == '__main__':
    main()
