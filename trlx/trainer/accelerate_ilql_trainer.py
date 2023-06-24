import os
from typing import Union, cast

import numpy as np
import torch
import transformers
from rich.console import Console
from rich.table import Table
import torch.nn.functional as F
from contextlib import contextmanager
import gc
import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch, ILQLSeq2SeqBatch
from trlx.models.modeling_ilql import (
    AutoModelForCausalLMWithILQLHeads,
    AutoModelForSeq2SeqLMWithILQLHeads,
    ILQLConfig,
)
from trlx.pipeline.offline_pipeline import (
    ILQLRolloutStorage,
    ILQLSeq2SeqRolloutStorage,
    tokenize_dialogue,
)
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import to_device

logger = logging.get_logger(__name__)


def make_experience(samples, rewards, tokenizer=None, max_length=2048, verbose=True):  # noqa: C901
    """
    Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
    """

    if verbose:
        logger.info("Collecting rollouts")
    if tokenizer is not None:
        samples = [tokenize_dialogue(s, tokenizer, max_length) for s in samples]
    all_query_lens = []
    all_input_ids = []
    all_actions_ixs = []
    all_states_ixs = []
    all_dones = []
    cdx = 0
    for sample in samples:
        cdx += 1
        length = 0
        if len(sample) != 2:
            continue
        all_query_lens.append(torch.tensor([len(sample[0].tokens)], dtype=torch.int32)) 
        all_input_ids.append(torch.tensor(sum((s.tokens for s in sample), ())))
        actions_ixs = []
        found = False
        for dm in sample:
            if dm.is_output:
                found = True
                actions_ixs.append(torch.arange(length - 1, length + len(dm.tokens) - 1))
            length += len(dm.tokens)
        if found == False:
            print('cdx:',cdx)

        states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
        all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
        all_actions_ixs.append(torch.hstack(actions_ixs))
        all_states_ixs.append(states_ixs)

    if tokenizer is not None and os.environ.get("RANK", "0") == "0" and verbose:
        logger.info("Logging sample example")
        prompt = tokenizer.decode(all_input_ids[0][: all_states_ixs[0][1]])
        response = tokenizer.decode(all_input_ids[0][all_states_ixs[0][1] :])
        columns = ["Prompt", "Response", "Reward"]
        table = Table(*columns, title="Sample Example", show_lines=True)
        table.add_row(prompt, response, str(rewards[0]))
        Console().print(table)

    sample_lengths = np.array(list(map(len, all_input_ids)))
    output_lengths = np.array(list(map(len, all_actions_ixs)))
    prompt_lengths = sample_lengths - output_lengths
    returns = torch.tensor(rewards, dtype=float)

    if os.environ.get("RANK", "0") == "0" and verbose:
        logger.info("Logging experience string statistics")
        columns = ["Prompt Length", "Output Length", "Sample Length"]
        table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
        row = []
        for lengths in [prompt_lengths, output_lengths, sample_lengths]:
            row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
        table.add_row(*row)
        Console().print(table)

    returns = returns - returns.mean()
    std_returns = returns.std()
    if not torch.isnan(std_returns):
        returns = returns / (std_returns + torch.finfo(returns.dtype).eps)
    rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
    for rs, ret in zip(rewards, returns):
        rs[-1] = ret

    attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]

    return ILQLRolloutStorage(
        all_input_ids,
        attention_mask,
        rewards,
        all_states_ixs,
        all_actions_ixs,
        all_dones,
        all_query_lens
    )

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

class PPODecorators(object):
    @classmethod
    @contextmanager
    def empty_cuda_cache(cls):
        yield
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

@register_trainer
class AccelerateILQLTrainer(AccelerateRLTrainer):
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        if not isinstance(config.method, ILQLConfig):
            raise ValueError("config.method must be ILQLConfig")

        self.ilql: ILQLConfig = cast(ILQLConfig, config.method)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            max_length=self.max_length,
            logit_mask=self.logit_mask,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
        )

    def get_arch(self, config):
        if config.model.model_arch_type == "seq2seq":
            from_fn = AutoModelForSeq2SeqLMWithILQLHeads.from_pretrained
            if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
                from_fn = AutoModelForSeq2SeqLMWithILQLHeads.from_config
        else:
            from_fn = AutoModelForCausalLMWithILQLHeads.from_pretrained
            if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
                from_fn = AutoModelForCausalLMWithILQLHeads.from_config
        x = from_fn(
            config.model.model_path,
            two_qs=config.method.two_qs,
            alpha=config.method.alpha,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        x.base_model.gradient_checkpointing_enable()
        return x

    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()
    
    @PPODecorators.empty_cuda_cache() 
    def calculate_kl(self, batch):
        # Ref Model for KL
        mask_len = (batch.input_ids[:1]).sum().item()
        ref_model = self.accelerator.unwrap_model(self.model).ref_model
        with torch.no_grad():
            self.model.eval()
            self.accelerator.unwrap_model(self.model).eval()
            generate_ids = ref_model.generate(batch.input_ids[:1, :mask_len], max_length=2048)
        self.model.train()
        self.accelerator.unwrap_model(self.model).train()
        kl_inputs = generate_ids
        kl_attn_mask = torch.ones_like(kl_inputs)
        with torch.no_grad():
            logits_ref = ref_model(
                input_ids=kl_inputs,
                attention_mask=kl_attn_mask,
                output_hidden_states=True
            ).logits
        logits_base, _, _, _, _ = self.model(
            input_ids=kl_inputs,
            attention_mask=kl_attn_mask,
        )
        
        logprobs_ref = logprobs_from_logits(logits_ref[:, :-1, :], kl_inputs[:, 1:])
        logprobs_base = logprobs_from_logits(logits_base[:, :-1, :], kl_inputs[:, 1:])
        log_ratio = (logprobs_base - logprobs_ref)[:, mask_len:]
        ratio = torch.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        # approx_kl = ratio * -0.001 
        approx_kl = torch.mean((ratio - 1) - log_ratio) * 0.001
        with self.accelerator.no_sync(self.model):
            self.accelerator.backward(approx_kl)
        return approx_kl 
    
    @PPODecorators.empty_cuda_cache() 
    def loss(self, batch: Union[ILQLBatch, ILQLSeq2SeqBatch]):
        batch = to_device(batch, self.accelerator.device)
        kl_penalty = self.calculate_kl(batch)
        
        logits, qs, target_qs, vs, _ = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            actions_ixs=batch.actions_ixs,
            states_ixs=batch.states_ixs,
        )
        return self.ilql.loss((logits, (qs, target_qs, vs, kl_penalty)), batch)

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, train_dataloader, eval_dataloader)

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def make_experience_seq2seq(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        logger.info("Collecting rollouts")
        if self.tokenizer:
            samples = [tokenize_dialogue(s, self.tokenizer, max_length) for s in samples]

        all_input_ids = []
        all_output_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            all_input_ids.append(torch.tensor(sample[0].tokens))
            all_output_ids.append(torch.tensor(sample[1].tokens))
            actions_ixs = []
            length = 0
            for phrase in sample:
                if phrase.is_output:
                    length = len(phrase.tokens)
                    actions_ixs.append(torch.arange(0, length - 1))
            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        if self.tokenizer and os.environ.get("RANK", "0") == "0":
            logger.info("Logging sample example")
            prompt = self.tokenizer.decode(all_input_ids[0])
            response = self.tokenizer.decode(all_output_ids[0])
            columns = ["Prompt", "Response", "Reward"]
            table = Table(*columns, title="Sample Example", show_lines=True)
            table.add_row(prompt, response, str(rewards[0]))
            Console().print(table)

        sample_lengths = np.array(list(map(len, all_input_ids))) + np.array(list(map(len, all_output_ids)))
        output_lengths = np.array(list(map(len, all_output_ids)))
        prompt_lengths = sample_lengths - output_lengths
        returns = torch.tensor(rewards, dtype=float)

        if os.environ.get("RANK", "0") == "0":
            logger.info("Logging experience string statistics")
            columns = ["Prompt Length", "Output Length", "Sample Length"]
            table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
            row = []
            for lengths in [prompt_lengths, output_lengths, sample_lengths]:
                row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
            table.add_row(*row)
            Console().print(table)

        returns = (returns - returns.mean()) / (returns.std() + torch.finfo(returns.dtype).eps)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]
        self.store = ILQLSeq2SeqRolloutStorage(
            all_input_ids,
            attention_mask,
            all_output_ids,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )

    def make_experience(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """

        if self.config.model.model_arch_type == "seq2seq":
            return self.make_experience_seq2seq(samples, rewards, max_length)

        self.store = make_experience(samples, rewards, self.tokenizer, max_length=max_length, verbose=True)
