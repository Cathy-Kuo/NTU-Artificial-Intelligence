from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import argparse

def generate(model, loader, tokenizer, args):
    predicts = []
    with open(args.output, 'w+') as file:
        for batch in tqdm(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            y = model.generate(**batch,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    top_k=args.top_k,
                    top_p=args.top_p)
            predicts += tokenizer.batch_decode(y, skip_special_tokens=True)
        for x in predicts:
            print(x, file=file)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.cuda()
    
    def preprocess_fn(examples):
        inputs = examples['maintext']
        inputs = ['summarize: ' + x for x in inputs]
        return tokenizer(inputs, max_length=256, padding='max_length', truncation=True)

    dataset = load_dataset('json', data_files={'test': args.reference}, field='data')['test']
    dataset = dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=dataset.column_names)
    dataset.set_format(type='torch')
    loader = DataLoader(dataset, batch_size=args.batch_size)
    '''
    for b in [1, 4]:
        args.num_beams = b
        loader = DataLoader(dataset, batch_size=args.batch_size // b)
        for t in [1.0, 0.95]:
            args.temperature = t
            for k, p in [(1, 1.0), (50, 0.9), (10, 1.0)]:
                args.do_sample = (k != 1)
                args.top_k = k
                args.top_p = p
                args.output = f'outputs/baseline.rl/beam_{b}.temp_{t}.k_{k}.p_{p}.txt'
                generate(model, loader, tokenizer, args)
    '''
    generate(model, loader, tokenizer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--reference', default='./data/news.json')
    parser.add_argument('--output', default='generate.txt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=int, default=1.0)
    args = parser.parse_args()
    main(args)



