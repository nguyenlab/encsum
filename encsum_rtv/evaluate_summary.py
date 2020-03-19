import os

import sklearn.metrics as skm
import numpy as np

from .lexical import run_rouge_common


def prfs(y_true, y_pred, **kwargs):
        p = skm.precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, **kwargs)
        if len(p) == 0: return []
        return [dict(zip(['P', 'R', 'F', 'S'], [float(v[i]) for v in p])) for i in range(len(p[0]))]

def prfs_evaluate(corpus_dir, predict_dir):
    all_y_true = []
    all_y_pred = []

    for pred_summary_file in os.listdir(predict_dir):
        if not pred_summary_file.endswith('.summary') or pred_summary_file.startswith('.'): continue

        doc_id = pred_summary_file[:-len('.summary')]
        with open(os.path.join(predict_dir, pred_summary_file)) as f:
            predictions = set([s.strip() for s in f])
        with open(os.path.join(corpus_dir, doc_id + '.summary')) as f:
            summary = set([s.strip() for s in f])
        with open(os.path.join(corpus_dir, doc_id + '.sentences')) as f:
            sentences = [s.strip() for s in f]

        y_true = np.zeros((len(sentences),), dtype='int32')
        y_pred = np.zeros((len(sentences),), dtype='int32')

        for i, s in enumerate(sentences):
            if s in summary: y_true[i] = 1
            if s in predictions: y_pred[i] = 1

        all_y_true += [y_true]
        all_y_pred += [y_pred]

    perfs = [prfs(y_true, y_pred)[-1] for y_true, y_pred in zip(all_y_true, all_y_pred)]
    macro = {k: np.average([p[k] for p in perfs]) for k in perfs[0]}
    return macro

def rouge_evaluate(corpus_dir, predict_dir):
    evaluations = [
        [os.path.join(predict_dir, filename), [os.path.join(corpus_dir, filename)]]
        for filename in os.listdir(predict_dir) if filename.endswith('.summary') and not filename.startswith('.')
    ]
    return run_rouge_common(evaluations, return_raw=True)


def main(corpus_dir, predict_dir):
    print(prfs_evaluate(corpus_dir, predict_dir))
    print(rouge_evaluate(corpus_dir, predict_dir))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--corpus-dir', required=True)
    p.add_argument('--predict-dir', required=True)
    args=p.parse_args()
    main(**vars(args))
    