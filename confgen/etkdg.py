from rdkit import Chem
from rdkit.Chem import AllChem

from confgen.tools import Cluster, Filter, GeomOptimizer


def run_etkdg(mol, options, n_cores, verbose=False):
    """Embed multiple conformers of mol"""

    assert len(Chem.GetMolFrags(mol)) == 1, "Can not handle multiple fragments yet."

    coordMap = {}
    if "constrain_atoms" in options and len(options["constrain_atoms"]) > 0:
        conf = mol.GetConformer(0)
        for idxI in options["constrain_atoms"]:
            PtI = conf.GetAtomPosition(idxI)
            coordMap[idxI] = PtI
    if "embed_threshold" not in options:
        options["embed_threshold"] = 0

    cids = AllChem.EmbedMultipleConfs(
        mol=mol,
        numConfs=options["n_confs"],
        coordMap=coordMap,
        numThreads=n_cores,
        pruneRmsThresh=options["embed_threshold"],
        useRandomCoords=True,
        ETversion=2,
    )

    assert len(cids) > 0, "Embed failed"

    if "constrain_atoms" in options and len(options["constrain_atoms"]) > 0:
        rms = AllChem.AlignMolConformers(mol, atomIds=options["constrain_atoms"])

    if verbose:
        print(f"{mol.GetNumConformers()} Conformers after ETKDG Embedding+Pruning")
