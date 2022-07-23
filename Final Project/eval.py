import tw_rouge
import json
import sys
import os

refs = [json.loads(line)['title'] for line in open(sys.argv[1])]
d = 'outputs/baseline'
for preds in os.listdir(d):
    if not preds.endswith('.txt'):
        continue
    preds = os.path.join(d, preds)
    print(preds)
    preds = [line for line in open(preds)]
    try:
        ret = tw_rouge.get_rouge(refs, preds)
    except:
        continue
    print('rouge-1: {:.2f}'.format(ret['rouge-1']['f'] * 100))
    print('rouge-2: {:.2f}'.format(ret['rouge-2']['f'] * 100))
    print('rouge-l: {:.2f}'.format(ret['rouge-l']['f'] * 100))
