from collections import namedtuple
from multiprocessing import Pool
import lxml.etree as ET
import numpy as np
import os
import numpy as np
from numpy.lib import npyio

class IRMeta(namedtuple('IRMeta', ['entries', 'entry_ids'])):
    Entry = namedtuple(
        'Entry', ['id', 'name', 'fact', 'summary', 'candidates', 'relevance'])
    Candidate = namedtuple('Candidate', ['id', 'name', 'path'])

    @classmethod
    def from_xml(cls, filepath):
        meta = ET.parse(filepath)
        entries = {
            entry.attrib['id']: cls.Entry(
                entry.attrib['id'],
                entry.find('fact').text.split('/')[-2],
                entry.find('fact').text,
                getattr(entry.find('summary'), 'text', None),
                [
                    cls.Candidate(
                        candidate.attrib['id'],
                        candidate.text.split('/')[-1],
                        candidate.text
                    ) for candidate in entry.iter('candidate_case')
                ],
                getattr(getattr(entry.find('cases_noticed'), 'text', None), 'split', lambda s: list())(',')
            )
            for entry in meta.iter('entry')
        }

        entry_ids = [entry.attrib['id'] for entry in meta.iter('entry')]
        return cls(entries, entry_ids)



def run_mpi(calls):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    n_slaves = nprocs - 1

    if rank > 0:
        for job_idx, run_job in enumerate(calls):
            if job_idx % n_slaves + 1 != rank: continue
            try:
                comm.isend(run_job(), dest=0, tag=job_idx * 10 + 1)
            except Exception as err:
                comm.isend(None, dest=0, tag=job_idx * 10 + 1)
                raise err
        sys.exit(0)  # slave exits gracefully
    else:
        return [comm.recv(source=job_idx % n_slaves + 1, tag=job_idx * 10 + 1) for job_idx in range(len(calls))]


def call(f): return f()

def run_pool(calls, cpu_count=4, chunksize=None):
    if cpu_count == 1:
        return [c() for c in calls]
    
    cpu_count = cpu_count or 4

    if not chunksize:
        chunksize = max(1, len(calls)//cpu_count//10)

    with Pool(cpu_count) as pool:
        return pool.map(call, calls, chunksize=chunksize) #
    

def run_async(calls, use_mpi=False, cpu_count=4, chunksize=None):
    if use_mpi:
        return run_mpi(calls=calls)
    else:
        return run_pool(calls=calls, cpu_count=cpu_count, chunksize=chunksize)

def save_npz(filepath,*,arrays=[],keyed_arrays={}):
    np.lib.npyio._savez(filepath, arrays, keyed_arrays, True, allow_pickle=True)


def load_npz(filepath):
    return numpy_load(filepath,allow_pickle=True)


class NpzFileWrapper:
    def __init__(self, npz_file):
        self.npz_file = npz_file
        self.npz_file.allow_pickle=True
        self.__files = set(self.npz_file.files)

    def __getattr__(self, item):
        try:
            return self.npz_file.__getattribute__(item)
        except AttributeError:
            return self.npz_file.__getattr__(item)

    def __contains__(self, item):
        return self.__files.__contains__(item)

    def __getitem__(self, key):
        key = key + '.npy'
        bytes = self.zip.open(key)
        magic = bytes.read(len(npyio.format.MAGIC_PREFIX))
        bytes.close()
        if magic == npyio.format.MAGIC_PREFIX:
            bytes = self.zip.open(key)
            return npyio.format.read_array(bytes,
                                           allow_pickle=self.allow_pickle,
                                           pickle_kwargs=self.pickle_kwargs)
        else:
            raise ValueError('can only read npy format')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.npz_file.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self.npz_file.files.__iter__()

    def __del__(self):
        return self.npz_file.__del__()


def numpy_load(*args, **kwargs):
    """
    A work around version of numpy.load
    bypass key checking in the object NpzFile returned by numpy.load
    re-implementing __contains__ and __getitem__
    """
    obj = np.load(*args, **kwargs)
    return NpzFileWrapper(obj) if isinstance(obj, npyio.NpzFile) else obj
