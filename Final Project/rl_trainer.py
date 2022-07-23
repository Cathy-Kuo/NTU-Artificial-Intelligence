import torch
from transformers import Seq2SeqTrainer
from datasets import load_metric
import tensorflow as tf
# Use CPU for ckiptagger
with tf.device('/CPU:0'):
    import tw_rouge

metric = load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def get_rewards(preds, refs):
    # rouge = metric.compute(predictions=predicts, references=refs, use_agregator=False)
    # rewards = torch.Tensor(rouge['rougeL'])[:, -1:]

    preds = [pred + '\n' for pred in preds] # Avoid empty hypothesis
    rouge = tw_rouge.get_rouge(preds, refs, avg=False)
    rewards = [r['rouge-l']['f'] + r['rouge-1']['f'] + r['rouge-2']['f'] for r in rouge]
    return torch.Tensor(rewards) / 3


class RLTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_weight = 0.0

    def set_rl_weight(self, weight):
        self.rl_weight = weight

    def compute_loss(self, model, inputs, return_outputs=False):
        xe_loss, output = super().compute_loss(model, inputs, True)
        log_probs, output_ids = output.logits.log_softmax(-1).max(-1)
        labels = inputs['labels']
        mask = (labels != -100)

        preds = self.tokenizer.batch_decode(output_ids)
        preds = [p.split(self.tokenizer.eos_token)[0] for p in preds]
        refs = self.tokenizer.batch_decode(labels * mask, skip_special_tokens=True)

        rewards = get_rewards(preds, refs)
        rewards = rewards.to(model.device).unsqueeze(-1)
        
        rl_loss = -(log_probs * rewards).sum(-1).mean()
        loss = self.rl_weight * rl_loss + (1.0 - self.rl_weight) * xe_loss

        if return_outputs:
            return (loss, output)
        return loss




