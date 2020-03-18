import logging
logger=logging

import numpy as np
import os
import io
import re
import json
import traceback
import sys
import h5py as h5
import yaml
import warnings
import functools

import keras
from keras.layers import Input, Embedding, Conv1D, Dense, concatenate, multiply
from keras.models import model_from_yaml, Model, model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 
from keras import optimizers
import keras.backend as K

from .nn_layers import Max, Avg, StandardDeviation, HierarchicalConcatenate, LocalMaxPooling1D, Broadcasting, SumToOne, ReduceSum
from .common import load_npz, save_npz, IRMeta, run_async

OBJECTS = {
    "Max":Max, 
    "Avg":Avg, 
    "StandardDeviation":StandardDeviation, 
    "HierarchicalConcatenate":HierarchicalConcatenate, 
    "LocalMaxPooling1D":LocalMaxPooling1D, 
    "Broadcasting":Broadcasting,
    "SumToOne":SumToOne,
    "ReduceSum":ReduceSum
}

def chain_func(funcs, v):
    for f in funcs:
        v = f(v)
    return v

def load_embeddings(embedding_file):
    return np.load(embedding_file)['embeddings']

def get_model(names, 
              embedding_file=None,
              model_dir=None,
              build_new_model=True, 
              model_hyper_parameters={},
              lazy_embedding=False):

    models = {}
    model_weight_file = os.path.join(model_dir,'model_weights.hdf5')    
    if model_dir:
        try:
            logger.info('Loading model... %s',names)
            for name in names:
                with open(model_dir + '/%s_model.yaml' % name) as f:
                    models[name] = model_from_yaml(f.read(),custom_objects=OBJECTS)
        except:
            models={}
            logger.warning("loading model config failed!")
            logger.debug(traceback.format_exc())

    if not models and not build_new_model:
        raise ValueError('neither building new model nor loading model from config')

    if not models and build_new_model:
        logger.info('Building new model...')
        train_model, score_model, encsum_model = build_model(emb_matrix=load_embeddings(embedding_file), **model_hyper_parameters)
        named_models = dict(train=train_model,score=score_model,encsum=encsum_model)
        os.makedirs(model_dir, exist_ok=True)
        for name, model in named_models.items():
            with open(os.path.join(model_dir, '%s_model.yaml' % name), 'w') as f:
                f.write(model.to_yaml())
        models = {name: named_models[name] for name in names}
    
    if os.path.exists(model_weight_file):
        logger.info("loading model weights")
        for model in models.values():
            model.load_weights(model_weight_file)
    return models


def get_loss_function(score_margin=1, nce=0,
                a1=1.00,
                a2=1.70,
                b1=0.30,
                b2=0.70,
                b3=0,
                b4=0,
                c1=0,
                c2=0):
    def loss(y_true, y_pred):
        y_c_mean = y_pred[:, 0:1]
        y_c_std = y_pred[:, 1:2]
        y_s_mean = y_pred[:, 2:3]
        y_s_std = y_pred[:, 3:4]

        _zero=0
        
        pl = _zero
        if a1:
            pl += a1 * (y_c_mean - y_s_mean)  # Ec > Es
        if b1:
            pl += b1 * (y_c_mean + y_c_std - y_s_mean - y_s_std)  # Mc > Ms
        if b2:
            pl += b2 * (y_c_mean - y_c_std - y_s_mean)  # mc > Es

        if c1:
            pl += c1 * (y_s_mean + y_s_std - y_c_mean + y_c_std) # Ms > mc
        if c2:
            pl += c2 * (y_c_mean - y_c_std - y_s_mean + y_s_std) # mc > ms

        nl = _zero
        if a2:
            nl += a2 * (y_s_mean - y_c_mean)  # Es' > Ec

        vl = _zero
        if b3:
            vl += b3 * y_c_std
        if b4:
            vl += b4 * y_s_std  # Vc > 0, Vs > 0

        loss_com = y_true * K.maximum(score_margin - pl, 0.)

        if nce and nl is not _zero:
            loss_com += (1 - y_true) * \
                K.maximum(score_margin - nl, 0) / (1 + nce)

        if vl is not _zero:
            loss_com -= vl
        return K.mean(loss_com, axis=-1)
    return loss


def train(data_dir,
          model_dir,
          optimizer=keras.optimizers.Adam(lr=0.0001,clipnorm=5.0),
          doctypes={'summary': 'summary', 'sentences': 'sentences'},
          nb_epochs=1,
          mini_epoch_factor=1,
          validation_split=0.02,
          nce=2,
          score_margin=0.5,
          loss_coef=dict(
                a1=1.00,
                a2=1.70,
                b1=0.30,
                b2=0.70,
                b3=0,
                b4=0,
                c1=0,
                c2=0),
          model_hyper_parameters={}):
    model = get_model(
        names=['train'], 
        embedding_file=os.path.join(data_dir,'embeddings.npz'),
        model_dir=model_dir,
        build_new_model=True, 
        model_hyper_parameters=model_hyper_parameters        
        )['train']
        
    if os.path.exists(os.path.join(data_dir,'emb_vocab.json')) and not os.path.exists(os.path.join(model_dir,'emb_vocab.json')):
        os.copy(os.path.join(data_dir,'emb_vocab.json'),model_dir)
    else:
        logger.warning(f'Make sure emb_vocab.json is either in {data_dir} or {model_dir}.')
    
    logger.info('Compiling...')
    model.compile(optimizer, loss=get_loss_function(score_margin=score_margin, nce=nce, **loss_coef))
    logger.info('Loading data...')
    data = load_data(data_dir=data_dir,doctypes=doctypes)
    train_data = data[:int((1-validation_split)*len(data))]
    val_data = data[len(train_data):]
    logger.info('n_train: %s, n_validation: %s',len(train_data),len(val_data))
    logger.info('Training...')
    callbacks = [
        ReduceLROnPlateau(
            factor=0.5,
            patience=5, 
            min_lr=1e-6,
            verbose=1),
        ModelCheckpoint(filepath=os.path.join(model_dir, 'model_weights.miniepoch-{epoch:03d}.hdf5'), save_weights_only=True)
    ]

    model.fit_generator(make_generator(train_data, shuffle=True, nce=nce),
                        validation_data=make_generator(val_data, nce=nce),
                        validation_steps=(1 + nce) * len(val_data),
                        steps_per_epoch=(1 + nce) * len(train_data)//mini_epoch_factor,
                        epochs=nb_epochs*mini_epoch_factor,
                        callbacks=callbacks)

    model.save_weights(os.path.join(model_dir, 'model_weights.hdf5'))
    logger.info('Training Completed.')


def make_generator(data, shuffle=False, nce=0):
    indices = list(range(len(data)))
    if not indices:
        raise StopIteration
    while True:
        # perform non-replacement sampling
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield data[i] + (np.ones((1, 1), dtype='float32'),)
            # perform NCE
            for _ in range(nce):
                j = np.random.randint(len(indices) - 1)
                if j >= i:
                    j += 1
                yield (data[i][0][0:2] + data[j][0][2:4],) + data[i][1:] + (np.zeros((1, 1), dtype='float32'),)


def load_data(data_dir, doctypes={}):
    data_summary = load_npz(os.path.join(data_dir, '%s.npz' % doctypes['summary']))
    data_sentences = load_npz(os.path.join(data_dir, '%s.npz' % doctypes['sentences']))
    data =[list(data_summary[f]) + list(data_sentences[f]) for f in data_summary.files]
    new_data = [x for x in data if x[1][-1] > 5]
    logger.info('keep: %s/%s'%(len(new_data),len(data)))
    return [([xx[None] for xx in x],) for x in new_data]


def build_model(emb_matrix, conv_filters=300, window_size=5, mlp_hidden_units=300):
    """
        return 3 models: training, scoring, encoded summarization
    """

    x_s = Input(shape=(None,), dtype='int32')
    x_s_pos = Input(shape=(None,), dtype='int32')
    x_c = Input(shape=(None,), dtype='int32')
    x_c_pos = Input(shape=(None,), dtype='int32')

    emb_layer = Embedding(input_dim=emb_matrix.shape[0],
                          output_dim=emb_matrix.shape[1],
                          weights=[emb_matrix],
                          trainable=False)

    y_s = emb_layer(x_s)
    y_c = emb_layer(x_c)

    conv_layer = Conv1D(filters=conv_filters,
                        kernel_size=window_size,
                        padding='same',
                        use_bias=False,
                        activation='relu',
                        )

    y_s = conv_layer(y_s)
    y_c = conv_layer(y_c)

    y_s_max_s = LocalMaxPooling1D()([y_s, x_s_pos])
    y_s_max_d = Max(axis=1, keepdims=True)(y_s_max_s)

    y_c_max_s = LocalMaxPooling1D()([y_c, x_c_pos])

    y_s = concatenate([HierarchicalConcatenate()([y_s, x_s_pos, y_s_max_s]),
                        Broadcasting(axis=1)([y_s_max_d, y_s])])
    y_c = concatenate([HierarchicalConcatenate()([y_c, x_c_pos, y_c_max_s]),
                        Broadcasting(axis=1)([y_s_max_d, y_c])])

    score_layer = [
        Dense(mlp_hidden_units, activation='tanh'),
        Dense(1, activation='sigmoid')
    ]

    y_s_score = chain_func(score_layer, y_s)
    y_c_score = chain_func(score_layer, y_c)

    y = concatenate([Avg(axis=1)(y_c_score), StandardDeviation(axis=1)(y_c_score),
                     Avg(axis=1)(y_s_score), StandardDeviation(axis=1)(y_s_score)])

    train_model = Model(inputs=[x_c, x_c_pos, x_s, x_s_pos], outputs=y)
    score_model = Model(inputs=[x_s, x_s_pos], outputs=[y_s_score])
    encsum_model = Model(inputs=[x_s, x_s_pos], outputs=ReduceSum(axis=1)(multiply([SumToOne(axis=1)(y_s_score),y_s])))

    return train_model, score_model, encsum_model


def infer_encsum(data_dir,model_dir,output_dir, doctypes={'sentences': 'sentences'},use_mpi=False):
    os.makedirs(output_dir,exist_ok=True)
    model = get_model(
        names=['encsum'], 
        model_dir=model_dir,
        build_new_model=False, 
        )['encsum']
    
    data_sentences = load_npz(os.path.join(data_dir, '%s.npz' % doctypes['sentences']))
    if use_mpi:
        outputs = run_async(
            [
                functools.partial(model.predict,[x[None] for x in data_sentences[f]])
             for f in data_sentences.files]
            ,use_mpi=use_mpi, )
        outputs = dict(zip(data_sentences.files,(o[0] for o in outputs)))
    else:
        outputs = {f:model.predict([x[None] for x in data_sentences[f]])[0] for f in data_sentences.files}

    save_npz(os.path.join(output_dir, '%s.encsum.npz' % doctypes['sentences']), keyed_arrays=outputs)


def compute_relevance_vector(encsum_features, use_mpi=False):
    """
        encsum_features: {
            "doc_id": encsum_feature_vector
        }
        - doc_id: "query_name", "query_name/candidates/candidate_name"

        return {
            "query_name/candidates/candidate_name": relevance_vector
        }
    """
    query_vectors = {}
    candidate_vectors = {}
    rel_vectors = {}

    for doc_id in encsum_features:
        id_parts = doc_id.split('/')
        if len(id_parts) == 1: query_vectors[doc_id]=encsum_features[doc_id]
        else: candidate_vectors[doc_id] = encsum_features[doc_id]
    
    
    if use_mpi:
        jobs=[]
        for doc_id in candidate_vectors:
            query_name = doc_id.split('/')[0]
            jobs.append((doc_id,functools.partial(
                np.multiply,candidate_vectors[doc_id],query_vectors[query_name]
            )))
        results = run_async([func for doc_id,func in jobs],use_mpi=use_mpi)
        rel_vectors = dict(zip([doc_id for doc_id,func in jobs],results))
    else:
        for doc_id in candidate_vectors:
            query_name = doc_id.split('/')[0]
            rel_vectors[doc_id] = candidate_vectors[doc_id] * query_vectors[query_name]

    return rel_vectors


def compute_and_export_relevance_vector(encsum_feature_file, meta_file, output_file, use_mpi=False):
    meta = IRMeta.from_xml(meta_file)
    cases = [meta.entries[entry_id].name for entry_id in meta.entry_ids]
    cands = {meta.entries[entry_id].name: [c.name for c in meta.entries[entry_id].candidates]
                for entry_id in meta.entry_ids}

    with load_npz(encsum_feature_file) as f:
        encsum_features = {key:f[key] for key in f.files}
    rel_vectors = compute_relevance_vector(encsum_features,use_mpi=use_mpi)

    grouped_rel_vectors = {
        casename: np.array([
            rel_vectors[f'{casename}/candidates/{candname}'] for candname in cands[casename]
        ])  for casename in cases
    }

    save_npz(output_file,keyed_arrays=grouped_rel_vectors)


def main_encsum_nn():
    p = argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--data-dir',required=True)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--doctypes',default={'summary': 'summary', 'sentences': 'paras'},type=json.loads)
    p.add_argument('--nb-epochs',default=5,type=int)
    p.add_argument('--mini-epoch-factor',default=10,type=int)
    p.add_argument('--nce',default=2,type=int)
    p.add_argument('--score-margin',default=0.5,type=float)
    p.add_argument('--loss-coef',default=dict(a1=1.00,a2=1.70,b1=0.30,b2=0.70,b4=0,c1=0,c2=0),type=json.loads)
    p.add_argument('--model-hyper-parameters',default={},type=json.loads)
    
    p.add_argument('--train',action='store_true')
    p.add_argument('--infer-encsum',action='store_true')
    p.add_argument('--output-dir')
    p.add_argument('--use-mpi',action='store_true')
    args=p.parse_args()

    if args.infer_encsum and not args.output_dir:
        p.error('--infer-encsum --output-dir')

    if args.train:
        logger.info('TRAIN')
        train(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            doctypes=args.doctypes,
            nb_epochs=args.nb_epochs,
            mini_epoch_factor=args.mini_epoch_factor,
            nce=args.nce,
            score_margin=args.score_margin,
            loss_coef=args.loss_coef,
            model_hyper_parameters=args.model_hyper_parameters,
            )
    if args.infer_encsum:
        logger.info('INFERENCE')

        infer_encsum(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            doctypes=args.doctypes,
            use_mpi=args.use_mpi
            )

def main_encsum_relevance():
    p = argparse.ArgumentParser()
    p.add_argument('cmd')
    p.add_argument('--encsum-feature-file',required=True)
    p.add_argument('--meta-file',required=True)
    p.add_argument('--output-file',required=True)
    p.add_argument('--use-mpi',action='store_true')

    args = p.parse_args()

    compute_and_export_relevance_vector(
        encsum_feature_file=args.encsum_feature_file,
        meta_file=args.meta_file,
        output_file=args.output_file,
        use_mpi=args.use_mpi
    )


def main():
    cmdparser = argparse.ArgumentParser()
    cmdparser.add_argument('cmd',choices=['encsum_nn','encsum_relevance'])
    args, _ = cmdparser.parse_known_args()
    globals()[f'main_{args.cmd}']()

if __name__ == '__main__':
    logger.basicConfig(level=logging.INFO)
    import argparse
    import json
    import sys
    logger.info(sys.argv)
    main()
    



    


