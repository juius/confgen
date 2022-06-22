from itertools import repeat
from multiprocessing import Pool

import numpy as np
from rdkit.Chem.rdMolAlign import AlignMolConformers

from confgen.utils import calc_chunksize, chunks


def conformer_rmsd(mol, confIds=[], onlyHeavy=True):
    """Align pairs of conformers and return the RMSD between them."""

    # which atoms to calculate RMSD for
    if onlyHeavy:
        atomids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    else:
        atomids = [a.GetIdx() for a in mol.GetAtoms()]

    rmsds = []
    for pair in confIds:
        # align the two conformers
        AlignMolConformers(mol, confIds=pair, atomIds=atomids)
        # calculate the RMSD
        c1 = mol.GetConformer(pair[0]).GetPositions()
        c2 = mol.GetConformer(pair[1]).GetPositions()
        diff = np.take(c1 - c2, atomids, axis=0)
        N = diff.shape[0]
        rmsds.append(np.sqrt((diff * diff).sum() / N))
    return rmsds


def rmsd_matrix(mol, n_workers=1):
    """Calculate RMSD matrix (only taking heavy atoms into account)
    between all conformers of a molecule."""
    confIds = [conf.GetId() for conf in mol.GetConformers()]
    pairs = []
    for i in range(1, len(confIds)):
        for j in range(0, i):
            pairs.append((confIds[i], confIds[j]))

    if n_workers == 1:
        cmat = conformer_rmsd(mol, confIds=pairs)
    else:
        # make chunks of pair ids
        pairs = chunks(pairs, calc_chunksize(n_workers, len(pairs), factor=1))
        # calculate rmsd in parallel
        args = [arg for arg in zip(repeat(mol), pairs)]
        with Pool(n_workers) as pool:
            results = pool.starmap(conformer_rmsd, args)
        cmat = []
        for res in results:
            for r in res:
                cmat.append(r)
    return cmat


if __name__ == "__main__":

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from confgen import ConformerGenerator

    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)NCC(=O)O")
    mol = Chem.AddHs(mol)

    _ = AllChem.EmbedMultipleConfs(mol, numConfs=1000, useRandomCoords=True)

    cmat = rmsd_matrix(mol, n_workers=4)

    print(cmat)
