import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import transformers
from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from transformers.trainer_callback import TrainerState
from transformers.utils import logging

from tqdm import tqdm

import math
import os
import numpy as np
from typing import List, Union, Dict, Optional

logger = logging.get_logger(__name__)


class InvariantTrainer(transformers.Trainer):

    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        optimizer, lr_scheduler = None, None
        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler

    def remove_dataparallel_wrapper(self):
        if hasattr(self.model, 'module'):
            self.model = self.model.module

    def invariant_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            training_logs=None,
            eval_logs=None,
            eval_fn=None,
            log_training_step=None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        if nb_steps is not None:
            max_steps = nb_steps
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            num_train_epochs = max(1, math.floor(max_steps / num_update_steps_per_epoch))
        else:
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            max_steps = num_update_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.lm_heads[env_name],
                                                                          num_training_steps=max_steps)
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.encoder, num_training_steps=max_steps)

        self.state = TrainerState()
        self.model.to(self.args.device)

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        for epoch in range(num_train_epochs):
            logger.info(f" Epoch: {epoch}")

            # make all dataloader iterateable
            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            for steps_trained_in_current_epoch in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")
                    # get a batch
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()

                    inputs = next(iter_loaders[env_name])

                    # make an update
                    loss = self.training_step(self.model, inputs)

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                       
                       torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )
                    
                    optimizer.step()
                    optimizers[env_name].step()

                    lr_scheduler.step()
                    lr_schedulers[env_name].step()

                    total_trained_steps += 1

                    if log_training_step is not None:
                        log_training_step(training_logs, total_trained_steps, loss.item())
                    
                    if eval_fn is not None and total_trained_steps % 100 == 0:
                        eval_metrics = eval_fn()
                        if eval_logs is not None and eval_metrics is not None:
                            eval_logs.append((total_trained_steps, eval_metrics))

                    if saving_heads:
                        if total_trained_steps % nb_steps_heads_saving == 0:
                            self.save_heads(total_trained_steps)
                    if saving_intermediary_models:
                        if total_trained_steps % nb_steps_model_saving == 0:
                            self.save_intermediary_model(total_trained_steps)


    def ensemble_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            training_logs=None,
            eval_logs=None,
            eval_fn=None,
            log_training_step=None,
            **kwargs,
    ):
        """
        Training the heads as en ensemble instead of following the IRM-games dynamic

        Args:
            nb_steps : permet de contrôler directement le nombre de optimizer.step() qui sont réalisés.
            num_train_epochs : nombre d'epochs d'entraînement à faire.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
       

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        # le nombre d'updates (steps) effectués durant une epoch.
        num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))

        if nb_steps is not None:
            max_steps = nb_steps
            num_train_epochs = max(1, math.floor(max_steps / num_update_steps_per_epoch))
        
        # Si on ne spécifie pas nb_steps, alors num_train_epochs détermine combien de fois on parcourt les données.
        else:
            max_steps = num_update_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.lm_heads[env_name],
                                                                          num_training_steps=max_steps)
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.encoder, num_training_steps=max_steps)

        self.state = TrainerState()

        self.model.to(self.args.device)
        
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps # nombre total d'exemples vu durant l'entraînement.

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        print("Num train epoch: ", num_train_epochs)
        print("Batch size: ", total_train_batch_size)
        print("Train data size: ", min_train_set_size)
        print("num_update_steps_per_epoch: ", num_update_steps_per_epoch)

        for epoch in range(num_train_epochs):
            logger.info(f" Epoch: {epoch}")
            print("epoch: ", epoch)
            # make all dataloader iterateable
            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            for steps_trained_in_current_epoch in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")
                    # get a batch
                    optimizer.zero_grad()
                    for e_n in training_set.keys():
                        optimizers[e_n].zero_grad()

                    batch = next(iter_loaders[env_name])
                    # uncomment it, for CPU only run
                    batch.to(self.args.device)

                    # loss.backward() is done inside training step
                    loss = self.training_step(self.model, batch)

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.max_grad_norm,
                        )

                   
                    optimizer.step()
                    for e_n in training_set.keys():
                        optimizers[e_n].step()

                    lr_scheduler.step()
                    for e_n in training_set.keys():
                        lr_schedulers[e_n].step()

                    total_trained_steps += 1

                    if log_training_step is not None:
                        log_training_step(training_logs, total_trained_steps, loss.item())
                    
                    if eval_fn is not None and total_trained_steps % 100 == 0:
                        eval_metrics = eval_fn()
                        if eval_logs is not None and eval_metrics is not None:
                            eval_logs.append((total_trained_steps, eval_metrics))

                    if saving_heads:
                        if total_trained_steps % nb_steps_heads_saving == 0:
                            self.save_heads(total_trained_steps)
                    if saving_intermediary_models:
                        if total_trained_steps % nb_steps_model_saving == 0:
                            self.save_intermediary_model(total_trained_steps)



    def multitask_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            training_logs=None,
            eval_logs=None,
            eval_fn=None,
            log_training_step=None,
            **kwargs,
    ):

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        if nb_steps is not None:
            max_steps = nb_steps
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            num_train_epochs = max(1, math.floor(max_steps / num_update_steps_per_epoch))
        else:
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            max_steps = num_update_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.lm_heads[env_name],
                                                                          num_training_steps=max_steps)
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.encoder, num_training_steps=max_steps)

        self.state = TrainerState()
        self.model.to(self.args.device)

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        for epoch in range(num_train_epochs):
            logger.info(f" Epoch: {epoch}")

            # make all dataloader iterateable
            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            for steps_trained_in_current_epoch in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")
                    # get a batch
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()

                    inputs = next(iter_loaders[env_name])
                    inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

                    # make an update
                    loss = self.training_step(self.model, inputs)

                    # Encoder forward
                    outputs = self.model.encoder(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    hidden_states = outputs[0]

                    # Head forward
                    logits = self.model.lm_heads[env_name](hidden_states)

                    # Compute loss (masked LM loss)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["labels"][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                       torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )
                    
                    optimizer.step()
                    optimizers[env_name].step()

                    lr_scheduler.step()
                    lr_schedulers[env_name].step()

                    total_trained_steps += 1

                    if log_training_step is not None:
                        log_training_step(training_logs, total_trained_steps, loss.item())
                    
                    if eval_fn is not None and total_trained_steps % 100 == 0:
                        eval_metrics = eval_fn()
                        if eval_logs is not None and eval_metrics is not None:
                            eval_logs.append((total_trained_steps, eval_metrics))

                    if saving_heads:
                        if total_trained_steps % nb_steps_heads_saving == 0:
                            self.save_heads(total_trained_steps)
                    if saving_intermediary_models:
                        if total_trained_steps % nb_steps_model_saving == 0:
                            self.save_intermediary_model(total_trained_steps)

    
    
    
    
    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        print("saving-heads")
        if not os.path.exists("lm_heads"):
            os.makedirs("lm_heads")

        for env, lm_head in self.model.lm_heads.items():
            if hasattr(lm_head, "dense"):
                # Cas RoBERTa
                weights = lm_head.dense.weight.data.cpu().numpy()
            elif hasattr(lm_head, "vocab_transform"):
                # Cas DistilBERT
                weights = lm_head.vocab_transform.weight.data.cpu().numpy()
            else:
                raise AttributeError("Inconnu : impossible d'extraire les poids de la tête")

            filepath = os.path.join("lm_heads", "{}-{}".format(env, step_count))
            np.save(filepath, weights)


    def get_single_train_dataloader(self, env_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator
        )
    

     