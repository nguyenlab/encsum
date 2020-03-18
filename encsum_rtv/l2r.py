import itertools
from subprocess import call

from sklearn.datasets import dump_svmlight_file
import tempfile
import numpy as np
import os
from .common import save_npz, load_npz, IRMeta

class SVMRank:
    def __init__(self, model_file, **kwargs):
        """
            See https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html

            General Options:
         -?          -> this help
         -v [0..3]   -> verbosity level (default 1)
         -y [0..3]   -> verbosity level for svm_light (default 0)
Learning Options:
         -c float    -> C: trade-off between training error
                        and margin (default 0.01)
         -p [1,2]    -> L-norm to use for slack variables. Use 1 for L1-norm,
                        use 2 for squared slacks. (default 1)
         -o [1,2]    -> Rescaling method to use for loss.
                        1: slack rescaling
                        2: margin rescaling
                        (default 2)
         -l [0..]    -> Loss function to use.
                        0: zero/one loss
                        ?: see below in application specific options
                        (default 1)
Optimization Options (see [2][5]):
         -w [0,..,9] -> choice of structural learning algorithm (default 3):
                        0: n-slack algorithm described in [2]
                        1: n-slack algorithm with shrinking heuristic
                        2: 1-slack algorithm (primal) described in [5]
                        3: 1-slack algorithm (dual) described in [5]
                        4: 1-slack algorithm (dual) with constraint cache [5]
                        9: custom algorithm in svm_struct_learn_custom.c
         -e float    -> epsilon: allow that tolerance for termination
                        criterion (default 0.001000)
         -k [1..]    -> number of new constraints to accumulate before
                        recomputing the QP solution (default 100)
                        (-w 0 and 1 only)
         -f [5..]    -> number of constraints to cache for each example
                        (default 5) (used with -w 4)
         -b [1..100] -> percentage of training set for which to refresh cache
                        when no epsilon violated constraint can be constructed
                        from current cache (default 100%) (used with -w 4)
SVM-light Options for Solving QP Subproblems (see [3]):
         -n [2..q]   -> number of new variables entering the working set
                        in each svm-light iteration (default n = q).
                        Set n < q to prevent zig-zagging.
         -m [5..]    -> size of svm-light cache for kernel evaluations in MB
                        (default 40) (used only for -w 1 with kernels)
         -h [5..]    -> number of svm-light iterations a variable needs to be
                        optimal before considered for shrinking (default 100)
         -# int      -> terminate svm-light QP subproblem optimization, if no
                        progress after this number of iterations.
                        (default 100000)
Kernel Options:
         -t int      -> type of kernel function:
                        0: linear (default)
                        1: polynomial (s a*b+c)^d
                        2: radial basis function exp(-gamma ||a-b||^2)
                        3: sigmoid tanh(s a*b + c)
                        4: user defined kernel from kernel.h
         -d int      -> parameter d in polynomial kernel
         -g float    -> parameter gamma in rbf kernel
         -s float    -> parameter s in sigmoid/poly kernel
         -r float    -> parameter c in sigmoid/poly kernel
         -u string   -> parameter of user defined kernel
Output Options:
         -a string   -> write all alphas to this file after learning
                        (in the same order as in the training set)
Application-Specific Options:

The following loss functions can be selected with the -l option:
     1  Total number of swapped pairs summed over all queries.
     2  Fraction of swapped pairs averaged over all queries.

NOTE: SVM-light in '-z p' mode and SVM-rank with loss 1 are equivalent for
      c_light = c_rank/n, where n is the number of training rankings (i.e.
      queries).
        """
        if 'c' not in kwargs:
            kwargs['c']=0.01
        self.kwargs = {(k if k.startswith('-') else '-' + k): str(v)
                       for k, v in kwargs.items()}
        
        self.model_file = model_file

    def fit(self, X, y):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        with tempfile.NamedTemporaryFile() as datafile:
            write_svmrank_format(X, y, datafile.name)
            call(['svm_rank_learn'] + list(itertools.chain(*
                                                                   list(self.kwargs.items()))) + [datafile.name,
                                                                                                  self.model_file])


    def predict(self, X):
        with tempfile.NamedTemporaryFile() as datafile, \
                tempfile.NamedTemporaryFile() as scorefile:
            write_svmrank_format(X, None, datafile.name)
            call(['svm_rank_classify', datafile.name,
                          self.model_file, scorefile.name])
            with open(scorefile.name) as f:
                scores = [float(line.strip()) for line in f]
        return np.array_split(scores, np.cumsum([x.shape[0] for x in X[:-1]]))


def write_svmrank_format(X, y, filepath):
    y = y or [None] * len(X)
    with open(filepath, 'wb') as f:
        for qid, (X_t, y_t) in enumerate(zip(X, y), start=1):
            y_t = y_t if y_t is not None else [0] * X_t.shape[0]
            dump_svmlight_file(X_t, y_t, f=f, zero_based=False,
                               query_id=[qid] * X_t.shape[0])


def train(model_file,
        feature_files,
        gold_file,
        svm_kwargs={}):
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    _, data_x, gold_label = load_data(feature_files, gold_file)
    SVMRank(model_file, **svm_kwargs).fit(data_x,gold_label)

def predict(model_file,
            output_file_prefix,
            feature_files,
            select_top,
            meta_file,
            yourid='YOURID',
            sort_outputs=True):
    os.makedirs(os.path.dirname(output_file_prefix),exist_ok=True)
    meta = IRMeta.from_xml(meta_file)
    casenames, data_x = load_data(feature_files)
    pred_y = SVMRank(model_file).predict(data_x)
    os.makedirs(os.path.dirname(output_file_prefix),exist_ok=True)
    save_npz(output_file_prefix+'.scores.npz', keyed_arrays=dict(zip(casenames,pred_y)))
    
    top_ranked = [np.argsort(scores)[::-1][:select_top] for scores in pred_y]
    write_submission(top_ranked, 
        output_file_prefix+'.submission.txt', 
        yourid, casenames, meta,sort_outputs=sort_outputs)


def write_submission(prediction, submission_file, yourid, casenames, meta,sort_outputs=True):
    case_name2id = {entry.name: entry.id for entry in meta.entries.values()}
    cand_idx2id = {entry.id: [cand.id for cand in entry.candidates]
                   for entry in meta.entries.values()}
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    
    with open(submission_file, mode='w', encoding='utf-8') as f:
        for idx, casename in enumerate(casenames):
            yp = prediction[idx]
            if sort_outputs:
                yp=sorted(yp)
            for y in yp:
                f.write('%s %s %s\n' % (
                    case_name2id[casename],
                    cand_idx2id[case_name2id[casename]][y],
                    yourid)
                        )


def evaluate_submission(submission_file, gold_file):
    with open(submission_file) as f:
        pred = ['/'.join(line.split()[:2]) for line in f]
    with open(gold_file) as f:
        gold = ['/'.join(line.split()[:2]) for line in f]
    TP = sum(1 for p in pred if p in gold)
    precision = TP / len(pred) if pred else 0
    recall = TP / len(gold) if gold else 0
    fmeasure = 2 * precision * recall / (precision + recall) if (precision+recall) else 0
    print('P: %s R: %s F: %s' % (precision, recall, fmeasure))


def load_data(feature_files, gold_file=None):
    casenames, data_x = load_grouped_features(feature_files)
    if not gold_file: return casenames, data_x
    with load_npz(gold_file) as labels:
        gold_label = [None] * len(casenames)
        for cidx, casename in enumerate(casenames):
            gold_label[cidx] = np.zeros((data_x[cidx].shape[0]))
            gold_label[cidx][labels[casename]] = 1
    return casenames, data_x, gold_label


def load_grouped_features(feature_file_paths):
    group_names = []
    data_x = []
    for feature_filepath in feature_file_paths:
        with load_npz(feature_filepath) as features:
            if not group_names:
                group_names = sorted(features.files)
                data_x = [[] for _ in group_names]
            for idx, group_name in enumerate(group_names):
                data_x[idx] += [features[group_name]]
    data_x = [np.hstack(x) for x in data_x]
    return group_names, data_x


def main_train():
    p=argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--model-file',required=True)
    p.add_argument('--feature-files',required=True,nargs='+')
    p.add_argument('--gold-file',required=True)
    p.add_argument('--svm-kwargs',default={},type=json.loads)
    
    args=dict(p.parse_args().__dict__)
    del args['cmd']
    train(**args)


def main_predict():
    p=argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--model-file',required=True)
    p.add_argument('--feature-files',required=True,nargs='+')
    p.add_argument('--output-file-prefix',required=True)
    p.add_argument('--select-top',required=True,type=int)
    p.add_argument('--meta-file',required=True)
    p.add_argument('--yourid',default='YOURID')
    p.add_argument('--sort-by-score',action='store_true')
    
    args=dict(p.parse_args().__dict__)
    args['sort_outputs'] = not args['sort_by_score']
    del args['cmd']
    del args['sort_by_score']
    predict(**args)


def main_evaluate():
    p=argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--submission-file',required=True)
    p.add_argument('--gold-file',required=True)
    args=dict(p.parse_args().__dict__)
    del args['cmd']
    evaluate_submission(**args)


def main():
    cmdparser = argparse.ArgumentParser()
    cmdparser.add_argument('cmd',choices=['train','predict','evaluate'])
    args, _ = cmdparser.parse_known_args()
    globals()[f'main_{args.cmd}']()


if __name__ == '__main__':
    import argparse
    import json
    main()

