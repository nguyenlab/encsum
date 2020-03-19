import json
import io
import os
import time
from collections import Counter
import logging
logger=logging
import numpy as np
from keras.models import model_from_yaml

from .encsum import OBJECTS
from .preprocess_docs import parse_text, preprocess, load_vocab, load_corpus

def get_model(config_file, weight_file):
    with open(config_file) as f:
        model = model_from_yaml(f.read(), custom_objects=OBJECTS)
        model.load_weights(weight_file, by_name=True)
    return model


def phrase_selection(
        doc_scores,
        doc_txt,
        sent_boundaries,
        top_anchor=10,
        ctx_radius=5,
        margin=1,
        max_len=0,
        whole_sentence=False,
        **kwargs):
    if top_anchor < 0 or max_len < 0: return []
    if top_anchor == 0 and max_len == 0: return []
    sent_txt = doc_txt  # original
    doc_txt = [w for l in doc_txt for w in l]  # flatten
    sel_txt = np.zeros(shape=(len(doc_txt),), dtype=bool)
    doc_scores = doc_scores.flatten()
    doc_len = doc_scores.shape[0]

    if top_anchor == 0:
        top_anchor = len(doc_txt)
    elif top_anchor < 1:
        top_anchor = int(top_anchor * len(doc_txt))
    else:
        top_anchor = int(top_anchor)

    if max_len == 0:
        max_len = len(doc_txt)
    elif 0 < max_len < 1:
        max_len = int(max_len * len(doc_txt))
    else:
        max_len = int(max_len)

    topk_words = np.argsort(doc_scores)[::-1][:top_anchor]
    sel_len = 0
    for widx in topk_words:
        if margin < 1:
            b_score = doc_scores[widx]
            jl = widx - 1
            jr = widx
            for jl in range(widx - 1, max(widx - ctx_radius - 1, 0), -1):
                if doc_scores[jl] - b_score > margin:
                    break
                b_score = doc_scores[jl]  # new
            b_score = doc_scores[widx]
            for jr in range(widx + 1, min(widx + ctx_radius + 1, doc_len)):
                if doc_scores[jr] - b_score > margin:
                    break
                b_score = doc_scores[jr]  # new
            jl = max(0, jl)
            jr = min(doc_len, jr)
        else:
            jl = max(widx - ctx_radius - 1, 0)
            jr = min(widx + ctx_radius + 1, doc_len)
        sel_len += jr - jl - 1 - sum(sel_txt[jl + 1:jr])
        if sel_len > max_len:
            break
        sel_txt[jl + 1:jr] = 1

    if whole_sentence:
        sel_sents = []
        b_offset = 0
        for sent in sent_txt:
            e_offset = b_offset + len(sent)
            if np.any(sel_txt[b_offset:e_offset]):
                sel_sents += [sent]
            b_offset = e_offset
        return sel_sents

    ccphrases = []
    cur_sent_idx = 0
    for i, p in enumerate(sel_txt):
        if not p:
            ccphrases += [[]]
        if i == sent_boundaries[cur_sent_idx]:
            ccphrases += [[]]
            cur_sent_idx += 1
        if p:
            ccphrases[-1] += [doc_txt[i]]
    return [c for c in ccphrases if c]


def norm_by_length(x):
    # return x/(1+np.log(1+len(x)))
    return x


def sentence_selection(
        doc_scores,
        doc_txt,
        sent_boundaries,
        top_anchor=10,
        max_len=0,
        **kwargs):
    if top_anchor < 0 or max_len < 0: return []
    if top_anchor == 0 and max_len == 0: return []

    if top_anchor == 0:
        top_anchor = len(doc_txt)
    elif top_anchor < 1:  # proportion to doc len
        top_anchor = int(top_anchor * len(doc_txt))
    else:
        top_anchor = int(top_anchor)

    if max_len == 0:
        max_len = len(doc_scores)
    elif 0 < max_len < 1:
        max_len = int(max_len * len(doc_scores))
    else:
        max_len = int(max_len)

    # compute sentence scores
    sent_scores = [norm_by_length(doc_scores[s_b:s_e]).sum() for s_b, s_e in zip(sent_boundaries, sent_boundaries[1:])]
    sorted_sent_indices = np.argsort(sent_scores)[::-1]
    out_sents = []
    count = 0
    total_length = 0
    for idx in sorted_sent_indices:
        count += 1
        total_length += len(doc_txt[idx])
        if count > top_anchor or total_length > max_len:
            break
        out_sents += [(idx, doc_txt[idx])]
    out_sents = sorted(out_sents)
    return [s[1] for s in out_sents]


SUMMARY_MODES = {
    'phrase_selection': phrase_selection,
    'sentence_selection': sentence_selection
}


def generate_summary(*args, **kwargs):
    mode = kwargs.get('summary_mode', 'phrase_selection')
    return SUMMARY_MODES[mode](*args, **kwargs)


def main(corpus_dir, vocab_file, config_file, weight_file, output_dir, **kwargs):
    vocab = load_vocab(vocab_file)
    corpus = load_corpus(corpus_dir)
    sentences = corpus['sentences']
    data = preprocess(corpus=corpus, vocab=vocab)['sentences']
    model = get_model(config_file=config_file, weight_file=weight_file)
    scores = {doc_id: model.predict([v[None] for v in data[doc_id]])[0] for doc_id in data}
    summaries = [(doc_id, generate_summary(scores[doc_id], sentences[doc_id], data[doc_id][1], **kwargs)) for doc_id in data]
    os.makedirs(output_dir, exist_ok=True)
    for doc_id, summary in summaries:
        with open(os.path.join(output_dir, doc_id + '.summary'), mode='w', encoding='utf-8') as f:
            for sentence in summary:
                f.write(' '.join(sentence))
                f.write('\n')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--config-file',required=True)
    p.add_argument('--weight-file',required=True)
    p.add_argument('--vocab-file',required=True)
    p.add_argument('--summary-mode',default='sentence_selection',choices=['phrase_selection','sentence_selection'])
    p.add_argument('--top-anchor',default=10,type=float)
    p.add_argument('--max-len',default=0.999,type=float)
    args=p.parse_args()
    main(**vars(args))
    