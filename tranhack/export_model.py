import click
import torch
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel, PeftModelForCausalLM
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from tranhack.const import BASE_MODEL_NAME


@click.command()
@click.option('--device', type=str, default='cuda:0')
def main(device: str):
    AutoTokenizer.from_pretrained(BASE_MODEL_NAME).save_pretrained('checkpoint/train-alma/merged')
    model = AutoPeftModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()
    model = PeftModelForCausalLM.from_pretrained(model, './checkpoint/train-alma')
    model = model.merge_and_unload()
    model.save_pretrained('checkpoint/train-alma/merged')


if __name__ == '__main__':
    main()
