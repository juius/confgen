import copy
import subprocess

from rdkit import Chem

# CONVERSION FACTORS
hartree2kcalmol = 627.5094740631


def stream(cmd, cwd=None, shell=True):
    """Execute command in directory, and stream stdout."""
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()
    popen.stdout.close()
    yield stderr

    return


def normal_termination(lines, pattern):
    """Check for pattern in lines."""
    for line in reversed(lines):
        if line.strip().startswith(pattern):
            return True
    return False


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def calc_chunksize(n_workers, len_iterable, factor=4):
    """Calculate chunksize argument for Pool-methods.

    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunksize, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunksize += 1
    return chunksize


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def sort_conformers(mol, property="energy"):
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    properties = [conf.GetDoubleProp(property) for conf in mol.GetConformers()]
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    sorted_ids = [
        cid for _, cid in sorted(zip(properties, conf_ids), key=lambda x: x[0])
    ]
    for i in sorted_ids:
        conf = mol.GetConformer(i)
        new.AddConformer(conf, assignId=True)

    return new


def combine_conformers(mol1, mol2):
    new_mol = copy.deepcopy(mol1)
    for conf in mol2.GetConformers():
        new_mol.AddConformer(conf, assignId=True)
    return new_mol
