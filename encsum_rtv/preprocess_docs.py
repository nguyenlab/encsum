import argparse
import json
import io
import os
import re
import logging
from pathlib import Path

from nltk import word_tokenize

import numpy as np
from collections import defaultdict, Counter
from .common import save_npz, numpy_load

VOCAB_PAD_IDX=0
VOCAB_UNK_IDX=1

def parse_text(text, word_tokenize=word_tokenize, sent_tokenize=lambda doc:doc.splitlines()):
    """
    return list of sentences, each sentence is a list of tokens
    """
    return [word_tokenize(sentence) for sentence in sent_tokenize(text)]

def preprocess(corpus, vocab, unk_idx=VOCAB_UNK_IDX):
    """
    map tokens to indices by vocab
    """
    return {
        content_type: {
            docid: make_sentence_chunks(sentences, vocab, unk_idx=unk_idx)
            for docid, sentences in corpus[content_type].items()
        }
        for content_type in corpus
    }


def make_sentence_chunks(tokenized_sentences, vocab, unk_idx=VOCAB_UNK_IDX,oov=Counter()):
    sents=[]
    sent_boundaries=[0]
    for sentence in tokenized_sentences:
        sents += [w2i(w,vocab,unk_idx=unk_idx,oov=oov) for w in sentence]
        sent_boundaries += [sent_boundaries[-1] + len(sentence)]
    return [np.array(sents,dtype='int32'),np.array(sent_boundaries,dtype='int32')]

def w2i(w, vocab, unk_idx=VOCAB_UNK_IDX, oov=Counter()):
    i=vocab.get(w) or vocab.get(re.sub(r'[\W0-9]', '', w.lower())) 
    if not i: 
        oov[w]+=1
        i=unk_idx
    return i

def load_vocab(vocab_file):
    with io.open(vocab_file, encoding='utf-8') as f:
        return json.load(f)

def load_corpus(corpus_dir,content_type_map={'summary':'.summary','sentences':'.sentences'}):
    corpus = {'summary':{},'sentences':{}}
    count = defaultdict(set)
    for filename in os.listdir(corpus_dir):
        for content_type, ext in content_type_map.items():
            if filename.endswith(ext):
                doc_id = filename[:-len(ext)]
                break
        else:
            content_type=None
        
        if content_type not in corpus:continue
        with io.open(os.path.join(corpus_dir,filename),encoding='utf-8') as f:
            contents = parse_text(f.read())
            corpus[content_type][doc_id]=contents
        count[doc_id].add(content_type)

    for doc_id,content_types in count.items():
        if len(content_types) != len(corpus):
            logging.warning('document %s only has %s',doc_id,content_types)
            for k in corpus:
                if doc_id in corpus[k]:
                    del corpus[k][doc_id]
    return corpus

def process_embeddings(embeddings_file, output_dir=None, filter_words=None):
    if output_dir: output_dir=Path(output_dir)
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
    if output_dir:
        with open(output_dir/'emb_vocab.json','w',encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
        save_npz(output_dir/'embeddings.npz', keyed_arrays={'embeddings':vecs, 'vocab':[vocab]})
    else:
        return vecs, vocab

def main():
    parser=argparse.ArgumentParser('Phrase Scoring Model for Summarization')
    parser.add_argument('--corpus-dir',help='Path to document list')
    parser.add_argument('--embeddings-file',help='Path to embeddings text file (fastText, glove, google-word2vec in text format)')
    parser.add_argument('--output-data-dir',help='Path to directory for storing preprocessed data to be passed to train')
    parser.add_argument('--content-type-map',default={'summary':'.summary','sentences':'.sentences'},type=json.loads)
    args=parser.parse_args()
    os.makedirs(args.output_data_dir,exist_ok=True)
    if not os.path.exists(os.path.join(args.output_data_dir,'embeddings.npz')):
        vecs,vocab = process_embeddings(args.embeddings_file)
        save_npz(os.path.join(args.output_data_dir,'embeddings.npz'), keyed_arrays={'embeddings':vecs,'vocab':[vocab]})
    else:
        logging.warning('USING EXISTED EMBEDDINGS IN OUTPUT DATA DIR %s',os.path.join(args.output_data_dir,'embeddings.npz'))
        with numpy_load(os.path.join(args.output_data_dir,'embeddings.npz')) as f:
            vocab = f['vocab'][0]
    with io.open(os.path.join(args.output_data_dir,'emb_vocab.json'),mode='w',encoding='utf-8') as f:
        json.dump(vocab,f,indent=1,ensure_ascii=False)
    data = preprocess(load_corpus(args.corpus_dir, content_type_map=args.content_type_map),vocab)
    for content_type in data:
        save_npz(os.path.join(args.output_data_dir,content_type),keyed_arrays=data[content_type])


if __name__ == '__main__': main()
