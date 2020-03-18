import itertools
import json
import os
import re
import sys
import functools
from subprocess import call
from collections import Counter, defaultdict
from nltk import sent_tokenize, word_tokenize
import numpy as np
import traceback
import tempfile
import shutil
from pathlib import Path
import unicodedata
import logging
logger=logging

from .common import IRMeta, run_async, save_npz


def prepare_label(corpus_dir, numeric_dir):
    data = {}
    meta = IRMeta.from_xml(str(Path(corpus_dir)/'IR.xml'))
    for entry in meta.entries.values():
        cand_id2idx = {candidate.id: idx for idx, candidate in enumerate(entry.candidates)}
        data[entry.name] = np.array([cand_id2idx[rel] for rel in entry.relevance], dtype='int32')
    save_npz(Path(numeric_dir)/'relevance.npz', keyed_arrays=data)


def w2i(w, word2idx, oov=Counter()):
    i=word2idx.get(w) or word2idx.get(re.sub(r'[\W0-9]', '', w.lower())) 
    if not i: 
        oov[w]+=1
        i=1
    return i


def pre_data(input_filepath,word2idx,min_line_length=5,encoding='utf-8'):
    input_filepath=Path(input_filepath)
    oov=Counter()
    sents = []
    sent_boundaries = [0]
    
    with open(input_filepath,encoding=encoding) as f:
        lines = f.readlines()

    for line in lines:
        if len(line) < min_line_length: continue  # skip one-word line
        tokens = line.split()
        if not tokens: continue
        token_idxs = [w2i(w,word2idx,oov) for w in tokens]
        sents += token_idxs
        sent_boundaries += [sent_boundaries[-1] + len(token_idxs)]

    if not sents:
        sents = [1]
        sent_boundaries = [0, 1]
    return [np.array(sents, dtype='int32'), np.array(sent_boundaries, dtype='int32')], oov
    

def text2numeric(text_dir, numeric_dir, meta_file, min_line_length=5, doctypes=['summary', 'paras', 'lead_sents'], encoding='utf-8', cpu_count=2):
    # code name:
    # case_id/summary
    # case_id/paras (fact --> para)
    # case_id/candidates/candidate_idx/summary
    # case_id/candidates/candidate_idx/paras
    # case_id/candidates/candidate_idx/lead_sents
    logger.info('text2id')
    text_dir = Path(text_dir)
    numeric_dir = Path(numeric_dir)
    numeric_dir.mkdir(parents=True,exist_ok=True)
    doctypes_set = set(doctypes)
    meta = IRMeta.from_xml(str(meta_file))
    with open(numeric_dir/'emb_vocab.json', encoding=encoding) as f:
        word2idx = json.load(f)

    oov=Counter()
    cases = [entry.name for entry in meta.entries.values()]
    data = {}
    jobs=[]
    for casename in cases:
        jobs+=[
            ('/'.join([casename, 'summary']), text_dir/casename/'summary.txt'),
            ('/'.join([casename, 'paras']), text_dir/casename/'fact.txt'),
        ] + [
            ('/'.join([casename, 'candidates', name, ext[1:]]), text_dir/casename/'candidates'/filename) 
            for filename in os.listdir(text_dir/casename/'candidates') for name,ext in [os.path.splitext(filename)] 
            if ext[1:] in doctypes_set
        ]
    
    results = run_async([
        functools.partial(
            pre_data,
            input_filepath=job[1],
            word2idx=word2idx,
            min_line_length=min_line_length,
            encoding=encoding) 
        for job in jobs], cpu_count=cpu_count,chunksize=max(1,len(jobs)//cpu_count//10))

    logger.info('writing results')
    for job,result in zip(jobs,results):
        key, _ = job
        arr, oov_ = result
        data[key] = arr
        oov.update(oov_)

    save_npz(numeric_dir/'data.npz', keyed_arrays=data)
    outdata = defaultdict(dict)
    for count, key in enumerate(data, start=1):
        key_parts = key.split('/')
        outdata[key_parts[-1]]['/'.join(key_parts[:-1])] = data[key]
        if count % 1000 == 0:
            logger.debug(' %s/%s', count, len(data))
    for datatype in outdata:
        save_npz(numeric_dir/(datatype + '.npz'), keyed_arrays=outdata[datatype])

    logger.info('OOV: %s',sorted(oov.items(),key=lambda x:x[1],reverse=True)[:100])


def get_lead_sentence(buff, max_sent_len):
    buff = ' '.join(buff).strip()[:max_sent_len * 10]
    if not buff:
        return []
    lead_sent = sent_tokenize(buff)[0]
    lead_sent = re.sub(r'[^\w\s_-]', '', lead_sent)
    return list(itertools.islice((m.group() for m in re.finditer(r'\w+', lead_sent)), max_sent_len))


def list_para(content, encoding='utf-8'):
    if hasattr(content, 'readlines'):
        content = content.readlines()
    elif isinstance(content, (list, tuple)):
        pass
    elif isinstance(content, bytes):
        content = content.decode(encoding).split('\n')
    elif isinstance(content, str):
        content = content.split('\n')
    else:
        raise ValueError('unsupported input')
    paras = []

    last_pid = 0
    for line in content:
        m = re.search(r'^(\[ (\d+) \]|(\d+)\s*[.]?\s*$)', line)
        if m and int(m.group(2) or m.group(3)) == last_pid + 1:
            paras += [[line[len(m.group(0)):]]]
            last_pid += 1
        else:
            if not paras: paras += [[]]  # this can be buggy ???
            paras[-1] += [line]
            # if m: logger.warning('wrong para id format??? %s %s %s',m.group(1),last_pid,filename)
    return paras


def prepare_corpus_meta(corpus_path, data_path, encoding='utf-8', **kwargs):
    os.makedirs(data_path, exist_ok=True)

    words = set()
    for rootpath, dirnames, filenames in os.walk(corpus_path):
        for filename in filenames:
            if filename.startswith('.'): continue # it's stupid cuzof macos
            with open(Path(rootpath)/filename, encoding=encoding) as f:
                tokens = f.read().split()
                words |= set(tokens)
    logger.info('n_words: %s', len(words))
    Path(data_path,'word_list.txt').write_text('\n'.join(sorted(words)),encoding='utf-8')


def process_embeddings(embeddings_file,output_dir,filter_words=None):
    output_dir=Path(output_dir)
    with open(embeddings_file,encoding='utf-8') as f:
        row = f.readline().split()
        if len(row) == 2:
            _,vec_size=list(map(int,row))
        else:
            f.seek(0)
            vec_size=len(row)-1

        vocab={0:0,1:1}
        vecs=[np.zeros((2,vec_size),dtype='float32')]

        for line in f:
            row = line.rsplit(maxsplit=vec_size)
            word = row[0]
            if filter_words is not None and word not in filter_words: continue
            if word in vocab: continue
            vocab[word]=len(vocab)
            vecs.append(np.array([row[1:]],dtype='float32'))
        del vocab[0],vocab[1] # remove padding and unk from lookup
        vecs = np.concatenate(vecs)
    with open(output_dir/'emb_vocab.json','w',encoding='utf-8') as f:
        json.dump(vocab,f,ensure_ascii=False)
    save_npz(output_dir/'embeddings.npz', keyed_arrays={'embeddings':vecs})


def tokenize_file(infilepath, outfilepath):
    infilepath=Path(infilepath)
    outfilepath=Path(outfilepath)
    try:
        if outfilepath.exists(): return
        with open(infilepath, encoding='utf-8') as f:
            doc = [word_tokenize(s) for line in f for s in sent_tokenize(line)]

        with open(outfilepath.with_suffix('.tmp'),'w',encoding='utf-8') as f:
            for s in doc:
                if not s: continue
                f.write(' '.join(s))
                f.write('\n')
        shutil.move(outfilepath.with_suffix('.tmp'),outfilepath)
    except:
        print('ERROR', infilepath)
        traceback.print_exc()
        raise

def remove_control_chars(s,*,
    control_char_re=re.compile(
        '[%s]' % re.escape(
                ''.join(c for c in (chr(i) for i in range(sys.maxunicode)) 
                            if unicodedata.category(c) == 'Cc')))):
    return control_char_re.sub('', s)


def parse_doc(doc, info=None):
    summary_start = summary_end = 0
    para_start = para_end = len(doc)

    for lineidx, line in enumerate(doc):
        if not summary_start:
            if line.startswith('Summary:'):
                summary_start = lineidx + 1
            elif line.startswith('Present:'):
                summary_start = lineidx

        if not summary_end and (
                line.startswith('Cases Noticed:')
                or line.startswith('Statutes Noticed:')
                or line.startswith('Counsel:')
                or line.startswith('Solicitors of Record:')
                or line.startswith('Cases Cited')
        ): summary_end = lineidx - 1

        if re.match(r'^(\[1\]|1\s*$)', line):
            if para_start < para_end:
                if doc[para_start][0] == '1' and doc[lineidx][0] == '[':
                    para_start = lineidx
                elif doc[para_start][0] == '1':
                    logger.warning(f'{info} wrong para start {lineidx}: {line}, prev {para_start}: {doc[para_start]} ')
            else:
                para_start = lineidx
    if summary_start > summary_end:
        summary_end = para_start
    return (summary_start, summary_end), (para_start, para_end)


def check_para(doc):
    paras = [line.strip() for line in doc if re.match(r'^(\[\d+\]|\d+)\s*$', line)]
    pbegin = None
    for idx, p in enumerate(paras):
        if (p[0] == '[' and p[1:-1] == '1') or p == '1':
            if pbegin is None:
                pbegin = idx
            else:
                raise ValueError('Wrong start')

    paras = paras[pbegin:]

    if len(paras) == 0:
        print(len(doc), doc[:5])
        raise ValueError()
    return len(paras) == int(paras[-1].strip()[1:-1])


def load_doc(fpath):
    with open(fpath, encoding='utf-8') as f:
        return [remove_control_chars(line) for line in f]


def parse_case(casename, case_dir, outdir, lead_sent_max_len=50):
    case_dir=Path(case_dir)
    cand_dir = case_dir/'candidates'
    outdir=Path(outdir)
    (outdir/'candidates').mkdir(parents=True,exist_ok=True)
    
    tokenize_file(case_dir/'fact.txt', outdir/'fact.txt')
    tokenize_file(case_dir/'summary.txt', outdir/'summary.txt')

    candidates = [candidate for candidate in os.listdir(cand_dir) if re.match('^[^.].*[.]txt$', candidate)]

    for candidate in candidates:
        if not all((outdir/'candidates'/(candidate + ext)).exists() for ext in ['.paras','.summary']):

            doc = load_doc(cand_dir/candidate)
            (summary_start, summary_end), (para_start, para_end) = parse_doc(doc, info=(casename, candidate))
            summary = doc[summary_start:summary_end]
            paras = doc[para_start:para_end]

            if not summary_start:
                logger.warning(f'{casename}/{candidate}: summary is not presented')
            elif not summary:
                logger.warning(f'{casename}/{candidate}: summary is not formated {summary_start,summary_end}')
            if not paras:
                logger.warning(f'{casename}/{candidate}: paras are not formated')

            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
                f.write('\n'.join(summary))
                f.write('\n')
                f.flush()
                tokenize_file(f.name, outdir/'candidates'/(candidate + '.summary'))

            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
                f.write('\n'.join(paras))
                f.write('\n')
                f.flush()
                tokenize_file(f.name, outdir/'candidates'/(candidate + '.paras'))

        if not((outdir/'candidates'/(candidate + '.lead_sents')).exists()):
            with tempfile.NamedTemporaryFile(dir=outdir/'candidates', mode='w', encoding='utf-8',delete=False) as f:
                for tokenized_lead_sent in (get_lead_sentence(para, lead_sent_max_len) for para in list_para(paras)):
                    f.write(' '.join(tokenized_lead_sent))
                    f.write('\n')
            shutil.move(f.name, outdir/'candidates'/(candidate + '.lead_sents'))


def run_parsing(input_dir,output_dir,meta_file,lead_sent_max_len=50,cpu_count=16):
    meta = IRMeta.from_xml(str(meta_file))
    input_dir=Path(input_dir)
    output_dir=Path(output_dir)

    run_async([functools.partial(parse_case, casename, input_dir/casename, output_dir/casename,lead_sent_max_len=lead_sent_max_len)
            for entry in meta.entries.values() for casename in [entry.name]],cpu_count=cpu_count)


def main(input_dir,output_dir,embeddings_file,mode='train',vocab_file=None,min_line_length=5,lead_sent_max_len=50,cpu_count=16,**kwargs):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    assert mode == 'train' or (vocab_file and Path(vocab_file).exists())
    run_parsing(input_dir/'data',output_dir/'text',meta_file=input_dir/'IR.xml',lead_sent_max_len=lead_sent_max_len,cpu_count=cpu_count)
    prepare_corpus_meta(input_dir/'data', output_dir/'numeric')
    if mode=='train':        
        process_embeddings(embeddings_file,output_dir/'numeric',set((output_dir/'numeric'/'word_list.txt').read_text(encoding='utf-8').split()))
    else:
        shutil.copy(vocab_file,output_dir/'numeric')
    text2numeric(output_dir/'text', output_dir/'numeric',meta_file=input_dir/'IR.xml',min_line_length=min_line_length,cpu_count=cpu_count)
    prepare_label(input_dir,output_dir/'numeric')


if __name__=='__main__':
    logger.basicConfig(level=logging.INFO)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--embeddings-file')
    p.add_argument('--vocab-file')
    p.add_argument('--mode',default='train',choices=['train','test'])
    p.add_argument('--cpu-count',default=2,type=int)
    args=p.parse_args()
    main(args.input_dir,args.output_dir,args.embeddings_file,mode=args.mode,vocab_file=args.vocab_file, cpu_count=args.cpu_count)
    