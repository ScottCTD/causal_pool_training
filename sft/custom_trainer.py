from functools import partial
from typing import Callable, Optional, Union, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
from transformers import Seq2SeqTrainer

if is_datasets_available():
    import datasets

class CausalPoolTrainer(Seq2SeqTrainer):
    
    def __init__(self, *args, data_collator_train=None, data_collator_eval=None, **kwargs):
        super().__init__(*args, **kwargs)
        if data_collator_train is None:
            data_collator_train = self.data_collator
        if data_collator_eval is None:
            data_collator_eval = self.data_collator
        self.data_collator_train = data_collator_train
        self.data_collator_eval = data_collator_eval

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        # with Seq2SeqTrainer, during eval, the eval_dataloader will use self.data_collator_eval
        data_collator = self.data_collator_train if is_training else self.data_collator_eval
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description=description)

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader
