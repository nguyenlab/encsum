# EncSum

# Dependency
- python >= 3.6
- keras >= 2.2
- tensorflow >= 1.8
- h5py
- scikit-learn
- lxml
- nltk

[ROUGE-1.5.5](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)
Please place ROUGE-1.5.5 root folder (containing `ROUGE-1.5.5.pl` script) next to encsum_rtv folder.

[svm_rank](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)
Please add `svm_rank` root folder (containing `svm_rank_learn` and `svm_rank_classify`) to $PATH.

# Experiment
- Training EncSum Model
- COLIEE 2018 Task 1
- SUM-HOLJ Summarization Task

## Training EncSum Model
`path/to/coliee2018/train/corpus/dir`, `path/to/coliee2018/test/corpus/dir`: COLIEE 2018 train, test folders containing `IR.xml` file.

`path/to/glove/embedding/file`: path to text-format word embeddings file `glove.840B.300d.txt`.

- Preprocessing
```bash
python -m encsum_rtv.preprocess_coliee2018 \
    --mode train \
    --input-dir path/to/coliee2018/training/corpus/dir \
    --output-dir data/coliee2018/preprocessed/train \
    --embeddings-file path/to/glove/embedding/file 
```

- Training Neural Net
```bash
python -m encsum_rtv.encsum encsum_nn \
    --train \
    --data-dir data/coliee2018/preprocessed/train/numeric \
    --model-dir model \
    --mini-epoch-factor 50 \
    --nb-epochs 5 
```

## COLIEE 2018 Task 1
`path/to/coliee2018/train/corpus/dir`, `path/to/coliee2018/test/corpus/dir`: COLIEE 2018 train, test folders containing `IR.xml` file.

- Preprocessing
```bash
python -m encsum_rtv.preprocess_coliee2018 \
    --mode test \
    --input-dir path/to/coliee2018/test/corpus/dir \
    --output-dir data/coliee2018/preprocessed/test \
    --vocab-file data/coliee2018/preprocessed/train/numeric/emb_vocab.json 
```

- Generating EncSum representation

Train:
```bash 
python -m encsum_rtv.encsum encsum_nn \
    --infer-encsum \
    --data-dir data/coliee2018/preprocessed/train/numeric \
    --model-dir model \
    --output-dir data/coliee2018/preprocessed/train/encsum

python -m encsum_rtv.encsum encsum_relevance \
    --encsum-feature-file data/coliee2018/preprocessed/train/encsum/paras.encsum.npz \
    --output-file data/coliee2018/preprocessed/train/encsum/encsum_relevance.npz \
    --meta-file path/to/coliee2018/train/corpus/dir/IR.xml
```

Test:
```bash 
python -m encsum_rtv.encsum encsum_nn \
    --infer-encsum \
    --data-dir data/coliee2018/preprocessed/test/numeric \
    --model-dir model \
    --output-dir data/coliee2018/preprocessed/test/encsum

python -m encsum_rtv.encsum encsum_relevance \
    --encsum-feature-file data/coliee2018/preprocessed/test/encsum/paras.encsum.npz \
    --output-file data/coliee2018/preprocessed/test/encsum/encsum_relevance.npz \
    --meta-file path/to/coliee2018/test/corpus/dir/IR.xml
```

- Extracting lexical features:

Train
```bash
python -m encsum_rtv.lexical \
    --text-dir data/coliee2018/preprocessed/train/text \
    --numeric-dir data/coliee2018/preprocessed/train/numeric \
    --meta-file path/to/coliee2018/train/corpus/dir/IR.xml \
    --cpu-count $CPU_COUNT
```

Test
```bash
python -m encsum_rtv.lexical \
    --text-dir data/coliee2018/preprocessed/test/text \
    --numeric-dir data/coliee2018/preprocessed/test/numeric \
    --meta-file path/to/coliee2018/test/corpus/dir/IR.xml \
    --cpu-count $CPU_COUNT
```
`--cpu-count $CPU_COUNT`: accelerating with multiprocessing.

- Train and Evaluate Ranker

```bash
python -m encsum_rtv.l2r train \
    --model-file model/svm_rank.dat \
    --feature-files data/coliee2018/preprocessed/train/encsum/encsum_relevance.npz \
        data/coliee2018/preprocessed/train/numeric/rouge*.npz \
    --gold-file data/coliee2018/preprocessed/train/numeric/relevance.npz

python -m encsum_rtv.l2r predict \
    --model-file model/svm_rank.dat \
    --feature-files data/coliee2018/preprocessed/test/encsum/encsum_relevance.npz \
        data/coliee2018/preprocessed/test/numeric/rouge*.npz \
    --output-file-prefix output/coliee2018/test \
    --select-top 10 \
    --meta-file path/to/coliee2018/test/corpus/dir/IR.xml

python -m encsum_rtv.l2r evaluate \
    --submission-file output/coliee2018/test.submission.txt \
    --gold-file path/to/task1_true_answer/file
```

## SUM-HOLJ Summarization Task

- Preprocessing
```bash
python -m encsum_rtv.preprocess_holj \
    path/to/raw/corpus/dir \
    data/holj/preprocessed
```

- Generating Summary
```bash
python -m encsum_rtv.generate_summary_holj \
    --corpus-dir data/holj/preprocessed \
    --output-dir output/holj \
    --config-file model/score_model.yaml \
    --weight-file model/model_weights.hdf5 \
    --vocab-file model/emb_vocab.json \
    --summary-mode sentence_selection \
    --top-anchor 0.10
```
`--top-anchor 0.10`: 10% of #sentences as summary. 

- Evaluation
```bash
python -m encsum_rtv.evaluate_summary_holj \
    --corpus-dir data/holj/preprocessed \
    --predict-dir output/holj
```

# Reference
This code is an implementation of this paper.
```
Tran, V., Le Nguyen, M., Tojo, S. et al. Encoded summarization: summarizing documents into continuous vector space for legal case retrieval. Artif Intell Law (2020). https://doi.org/10.1007/s10506-020-09262-4
```
