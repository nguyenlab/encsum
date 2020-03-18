import functools
import itertools
import os
from collections import defaultdict
import numpy as np
import tempfile

import logging
logger=logging

import re
import traceback
from subprocess import check_output
from .common import IRMeta, run_async, save_npz

DEFAULT_ROUGE_SCRIPT = os.environ.get('ROUGE_SCRIPT',
                                       os.path.join(os.path.dirname(__file__), '..','ROUGE-1.5.5','ROUGE-1.5.5.pl'))
DEFAULT_ROUGE_DATA = os.environ.get('ROUGE_DATA',
                                    os.path.join(os.path.dirname(__file__), '..','ROUGE-1.5.5','data'))
DEFAULT_ROUGE_ARGS=os.environ.get('ROUGE_ARGS','-c 95 -2 -1 -U -r 1000 -n 2 -w 1.2 -a -d')


def rouge_report(text_dir, numeric_dir, meta_file, 
                 models=('summary.txt', 'fact.txt'), 
                 doc_types=('summary', 'paras'),
                 rouge_script=DEFAULT_ROUGE_SCRIPT,
                 rouge_data=DEFAULT_ROUGE_DATA,
                 rouge_args=DEFAULT_ROUGE_ARGS,
                 **kwargs):

    meta = IRMeta.from_xml(meta_file)
    cases = [meta.entries[entry_id].name for entry_id in meta.entry_ids]
    cands = {meta.entries[entry_id].name: [c.name for c in meta.entries[entry_id].candidates]
                for entry_id in meta.entry_ids}

    logger.info("processing %s cases", len(cases))

    jobs = [(case, model, doc_type, candidate_name) 
            for (case, model, doc_type) in itertools.product(cases, models, doc_types) 
            for candidate_name in cands[case]]

    results = run_async(calls=[functools.partial(
        run_rouge,
        query_dir=os.path.join(text_dir, case),
        query_name=model,
        candidate_dir=os.path.join(text_dir, case, 'candidates'),
        candidate_names=[f'{candidate_name}.{doc_type}'],
        report_file=os.path.join(numeric_dir, 'rouge-reports', f'{case}/{candidate_name}.{model}--{doc_type}.txt'),
        rouge_script=rouge_script,
        rouge_data=rouge_data,
        rouge_args=rouge_args
    ) for (case, model, doc_type, candidate_name) in jobs], 
        cpu_count=kwargs.get('cpu_count',2),use_mpi=kwargs.get('use_mpi',False))

    data = defaultdict(lambda : defaultdict(list))
    eval_methods = None
    for (case, model, doc_type, _ ), (eval_methods, eval_results) in zip(jobs, results):
        # eval_results: 3D list of list of list (eval_method, candidates=1, features=PRF)
        data[model +'--'+ doc_type][case]+=[eval_results]

    for lexpair in data.values():
        for casename in lexpair:
            vecs = lexpair[casename]
            lexpair[casename] = np.array(vecs,dtype='float32').reshape(len(vecs),-1)

    logger.info(eval_methods)
    os.makedirs(os.path.join(numeric_dir), exist_ok=True)
    for lexpair in data:
        save_npz(os.path.join(numeric_dir, f'rouge-reports.{lexpair}.npz'),
            keyed_arrays=data[lexpair])


def run_rouge(query_dir, query_name, candidate_dir, candidate_names, report_file,
              rouge_script=DEFAULT_ROUGE_SCRIPT,
              rouge_data=DEFAULT_ROUGE_DATA,
              rouge_args=DEFAULT_ROUGE_ARGS):
    """
    return tuple(methods, rouge-scores :list[methods :list[candidates :list[precision:float, recall:float, fmeasure:float]]])
    """

    return run_rouge_common(
        evaluations=[
            [
                os.path.join(candidate_dir, candidate_name),
                [os.path.join(query_dir, query_name)]
            ] for candidate_name in candidate_names

        ], report_file=report_file, rouge_script=rouge_script,
        rouge_data=rouge_data,
        rouge_args=rouge_args)


def run_rouge_common(evaluations, *,
              report_file=None,
              rouge_script=DEFAULT_ROUGE_SCRIPT,
              rouge_data=DEFAULT_ROUGE_DATA,
              rouge_args=DEFAULT_ROUGE_ARGS,
              return_raw=False):
    """
    evaluations: list[source_path, list[reference_paths]]
    return  tuple(methods, rouge-scores:list[methods :list[source :list[precision:float, recall:float, fmeasure:float]]])
    """
    if '-d' not in rouge_args: rouge_args += ' -d'

    try:
        if report_file and os.path.exists(report_file):
            with open(report_file, encoding='utf-8') as f:
                return parse_rouge_report(f.read()) if not return_raw else f.read()
        if report_file:
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with tempfile.NamedTemporaryFile() as conf_file, open(conf_file.name, mode='w', encoding='utf-8') as f:
            f.write(
                """<ROUGE-EVAL version="1.55">
                {evals}
                </ROUGE-EVAL>\n""".format(evals="\n".join(
                    """
                    <EVAL ID="{eid}">
                    <PEER-ROOT></PEER-ROOT>
                    <MODEL-ROOT></MODEL-ROOT>
                    <INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>
                    <PEERS>
                    {peers}
                    </PEERS>
                    <MODELS>
                    {models}
                    </MODELS>
                    </EVAL>
                    """.format(eid=eidx,
                               models='\n'.join(
                                   ['<M ID="M">{filepath}</M>'.format(filepath=os.path.abspath(filepath)) for filepath
                                    in
                                    evaluation[1]]),
                               peers='<P ID="P">{filepath}</P>'.format(filepath=os.path.abspath(evaluation[0])),
                               ) for eidx, evaluation in enumerate(evaluations))))
            f.close()

            results = check_output(
                [
                    'perl', rouge_script,
                    '-e', rouge_data
                ] + rouge_args.split() + ['-m', conf_file.name])
            results = results.decode('utf-8')

        if report_file:
            with open(report_file+'.tmp',mode='w', encoding='utf-8') as report_file_tmp:
                results = ''.join(line for line in results.splitlines(True) if re.match('^(\S+) (\S+) Eval.*$',line))
                report_file_tmp.write(results)
            os.rename(report_file+'.tmp',report_file)
        return parse_rouge_report(results)  if not return_raw else results
    except:
        traceback.print_exc()


def parse_rouge_report(txt):
    """
    return rouge-scores :list[methods :list[peers :list[precision:float, recall:float, fmeasure:float]]]
    """
    eval_methods = list()
    eval_results = list()
    # peer_idx=0
    for line in txt.splitlines():
        m = re.search(r"^(\S+) (\S+) Eval (\S+) R:(\S+) P:(\S+) F:(\S+)", line)
        if not m: continue
        # assert m.group(1) == m.group(3)[-len(m.group(1)):]
        # peer_id = m.group(1)
        eval_method = m.group(2)
        if eval_method not in eval_methods:
            eval_methods += [eval_method]
            eval_results += [[]]
            # peer_idx=0
        # eval_id = m.group(3)[:-len(peer_id) - 1]
        recall = float(m.group(4))
        precision = float(m.group(5))
        fmeasure = float(m.group(6))
        eval_results[-1] += [[precision, recall, fmeasure]]

    return eval_methods, eval_results


if __name__ == '__main__':
    logger.basicConfig(level=logging.INFO)
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--text-dir',required=True)
    p.add_argument('--numeric-dir',required=True)
    p.add_argument('--meta-file',required=True)
    p.add_argument('--cpu-count',default=2,type=int)
    p.add_argument('--use-mpi',action='store_true')
    args=p.parse_args()
    rouge_report(text_dir=args.text_dir, numeric_dir=args.numeric_dir, meta_file=args.meta_file,cpu_count=args.cpu_count,use_mpi=args.use_mpi)
