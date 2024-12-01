import datasets
import torch
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from torch.utils.data import Dataset
from torchmetrics import MeanMetric, Metric
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from xztrainer import XZTrainable, BaseContext, DataType, ModelOutputsType, ContextType, set_seeds, enable_tf32, \
    XZTrainer, XZTrainerConfig

from tranhack.const import BASE_MODEL_NAME, make_prompt
from tranhack.util import stack_pad


def prepare_model_for_lora(device):
    model = AutoLigerKernelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=64, lora_dropout=0.1,
        use_rslora=True,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']  # att
    )
    model = get_peft_model(model, peft_config)
    return model


class TranDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self._dataset = dataset

    def __getitem__(self, item_n):
        item = self._dataset[item_n]
        prompt = make_prompt(item['name'], item['faculty'], item['direction'])

        encoding = self._tokenizer.encode_plus(prompt).encodings[0]
        input_ids = encoding.ids
        attention_mask = encoding.attention_mask
        labels = [-100] * len(input_ids)

        encoding_tran = self._tokenizer.encode_plus(item['translation'], add_special_tokens=False).encodings[0]
        input_ids.extend(encoding_tran.ids)
        attention_mask.extend(encoding_tran.attention_mask)
        labels.extend(encoding_tran.ids)

        input_ids.append(self._tokenizer.eos_token_id)
        attention_mask.append(1)
        labels.append(self._tokenizer.eos_token_id)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

    def __len__(self):
        return len(self._dataset)


class TranCollator:
    def __call__(self, batch):
        return {
            'input_ids': stack_pad([x['input_ids'] for x in batch], 0),
            'labels': stack_pad([x['input_ids'] for x in batch], -100),
            'attention_mask': stack_pad([x['attention_mask'] for x in batch], 0),
        }


class TranTrainable(XZTrainable):
    def step(self, context: BaseContext, data: DataType) -> tuple[Tensor, ModelOutputsType]:
        outs = context.model(**data)
        return outs.loss, {
            'loss': outs.loss
        }

    def create_metrics(self, context_type: ContextType) -> dict[str, Metric]:
        return {
            'loss': MeanMetric()
        }

    def update_metrics(self, context_type: ContextType, model_outputs: dict[str, list], metrics: dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])


def train():
    accel = Accelerator(
        gradient_accumulation_steps=8,
        log_with='tensorboard',
        project_dir='.'
    )
    set_seeds(0xFAFA)
    enable_tf32()
    with torch.no_grad():
        lora_device = torch.empty(1).to(accel.device).device
    model = prepare_model_for_lora(lora_device)
    train_ds = datasets.load_from_disk('data/train')
    test_ds = datasets.load_from_disk('data/test')

    trainer = XZTrainer(
        config=XZTrainerConfig(
            experiment_name='train-alma-',
            minibatch_size=1,
            minibatch_size_eval=1,
            epochs=1,
            gradient_clipping=5.0,
            optimizer=lambda module: AdamW8bit(module.parameters(), lr=5e-5, weight_decay=1e-4),
            scheduler=lambda optimizer, total_steps: get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05),
                                                                                     total_steps),
            collate_fn=TranCollator(),
            dataloader_persistent_workers=False,
            dataloader_num_workers=4,
            dataloader_shuffle_train_dataset=True,
            dataloader_pin_memory=True,
            tracker_logging_dir='./logs',
            log_steps=10,
            save_steps=1000,
            eval_steps=1000000
        ),
        model=model,
        trainable=TranTrainable(),
        accelerator=accel
    )
    trainer.train(TranDataset(train_ds), TranDataset(test_ds))
    model.save_pretrained('checkpoint/train-alma')


if __name__ == '__main__':
    train()
