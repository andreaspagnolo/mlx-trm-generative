from typing import Any, Dict, Tuple
import mlx.core as mx
import mlx.nn as nn
from tasks.base import Task
from models import trm

import numpy as np
import random
import itertools
import string

# Vocabulary for Propositional Logic
# Vars: a-z
# Ops: &, |, >, ~
# Misc: (, ), |-
VARS = list(string.ascii_lowercase)
OPS = ['&', '|', '>']

VOCAB = {
    'pad': 0,
    **{v: i+1 for i, v in enumerate(VARS)},
    '&': 27, '|': 28, '>': 29, '~': 30,
    '(': 31, ')': 32,
    '|-': 33,
    'end': 34
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

def solve(premises_str, conclusion_str):
    """
    Checks if premises entail conclusion using truth tables.
    """
    # Parse unique variables
    full_expr = premises_str + " " + conclusion_str
    used_vars = sorted(list(set([c for c in full_expr if c in VARS])))
    
    if not used_vars:
        return False # Should not happen

    # Generate truth table
    for values in itertools.product([False, True], repeat=len(used_vars)):
        env = dict(zip(used_vars, values))
        
        # Evaluate premises
        # Replace operators with python equivalents for eval
        # & -> and, | -> or, > -> <= (implication), ~ -> not
        # We need to be careful with eval security, but inputs are controlled here.
        
        def eval_expr(expr):
            py_expr = expr.replace('&', ' and ').replace('|', ' or ').replace('>', ' <= ').replace('~', ' not ')
            return eval(py_expr, {}, env)

        try:
            prem_val = eval_expr(premises_str)
            if prem_val:
                conc_val = eval_expr(conclusion_str)
                if not conc_val:
                    return False # Counter-example found: Premises True, Conclusion False
        except:
             return False # Malformed

    return True # Valid in all models where premises are true

def generate_random_expr(vocab_subset, depth=0, max_depth=2):
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        return random.choice(vocab_subset)
    
    op = random.choice(OPS + ['~'])
    if op == '~':
        return f"~({generate_random_expr(vocab_subset, depth+1, max_depth)})"
    else:
        left = generate_random_expr(vocab_subset, depth+1, max_depth)
        right = generate_random_expr(vocab_subset, depth+1, max_depth)
        return f"({left}{op}{right})"

def generate_sample(seq_len=64, num_vars=4, max_depth=2):
    vocab_subset = VARS[:num_vars]
    while True:
        # Generate premises (randomly 1 to 3)
        num_premises = random.randint(1, 3)
        premises = [generate_random_expr(vocab_subset, 0, max_depth) for _ in range(num_premises)]
        premises_str = "(" + ")&(".join(premises) + ")"
        
        # Generate conclusion
        # valid/invalid balance
        if random.random() < 0.5:
            # Try to generate a valid one (likely related to premises)
            # Simple heuristic: pick a sub-expression or variation
            conclusion_str = random.choice(premises) if random.random() < 0.3 else generate_random_expr(vocab_subset, 0, max_depth)
        else:
            conclusion_str = generate_random_expr(vocab_subset, 0, max_depth)
            
        label = 1 if solve(premises_str, conclusion_str) else 0
        
        text = f"{premises_str}|-{conclusion_str}"
        
        # Tokenize
        tokens = [VOCAB.get(c, VOCAB.get(text[i:i+2], 0)) for i, c in enumerate(text)]
        # Fix double char token for |-
        clean_tokens = []
        skip = False
        for i, c in enumerate(text):
            if skip:
                skip = False
                continue
            if text[i:i+2] == '|-':
                clean_tokens.append(VOCAB['|-'])
                skip = True
            elif c in VOCAB:
                clean_tokens.append(VOCAB[c])
        
        if len(clean_tokens) > seq_len - 1:
            continue # Try again if too long
            
        # Pad
        clean_tokens.append(VOCAB['end'])
        tokens = clean_tokens + [VOCAB['pad']] * (seq_len - len(clean_tokens))
        
        return np.array(tokens), np.array(label)

class InfiniteLoader:
    def __init__(self, generator_func, steps_per_epoch):
        self.generator_func = generator_func
        self.steps_per_epoch = steps_per_epoch
        self.generator = None
    
    def reset(self):
        self.generator = self.generator_func()
    
    def __iter__(self):
        if self.generator is None:
            self.reset()
        for _ in range(self.steps_per_epoch):
            yield next(self.generator)

def fol_dataset(batch_size, seq_len=64, num_vars=4, max_depth=2, steps_per_epoch=100):
    
    def generator():
        while True:
            batch_tokens = []
            batch_labels = []
            for _ in range(batch_size):
                t, l = generate_sample(seq_len, num_vars, max_depth)
                batch_tokens.append(t)
                batch_labels.append(l)
            yield {'inputs': mx.array(np.stack(batch_tokens)), 'label': mx.array(np.stack(batch_labels))}

    # Mock meta for compatibility
    steps = steps_per_epoch
    meta = {
        "n_train": steps * batch_size,
        "n_test": 10 * batch_size,
        "steps_per_epoch": steps,
        "vocab_size": len(VOCAB)
    }
    
    return InfiniteLoader(generator, steps), InfiniteLoader(generator, 10), meta


class LogicInferenceTask(Task):
    def get_dataset(self, batch_size: int, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        steps_per_epoch = kwargs.get("steps_per_epoch", 20)
        return fol_dataset(batch_size, steps_per_epoch=steps_per_epoch)

    def get_model_config(self, meta: Dict[str, Any]) -> trm.ModelConfig:
        return trm.ModelConfig(
            vocab_size=meta.get("vocab_size", 256),
            depth=2,
            dim=64,
            heads=4,
            n_outputs=2,
        )

    def loss_fn(self, model_outputs: Dict[str, mx.array], batch: Dict[str, mx.array], carry: Dict[str, Any]) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
        label = carry["current_data"]["label"]
        pred = model_outputs["logits"]
        is_correct = mx.argmax(pred, axis=1) == label

        ce = nn.losses.cross_entropy(pred, label, reduction="mean")
        bce = nn.losses.binary_cross_entropy(
            model_outputs["q_halt_logits"], is_correct, with_logits=True, reduction="mean"
        )
        loss = ce + 0.5 * bce

        stats = {
            "q_prob_mean": mx.sigmoid(model_outputs["q_halt_logits"]).mean(),
            "frac_halted": carry["halted"].mean(),
            "avg_steps": carry["steps"].mean(),
        }

        return loss, mx.sum(is_correct), stats
