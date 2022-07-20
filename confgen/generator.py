import copy
import logging
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol

from confgen import crest, etkdg
from confgen.tools import GeomOptimizer

from rdkit import RDLogger
from rdkit.rdBase import BlockLogs

RDLogger.DisableLog("rdApp.*")

from rdkit import RDLogger
from rdkit.rdBase import DisableLog

for level in RDLogger._levels:
    DisableLog(level)

class ConformerGenerator:
    def __init__(
        self, method, options, n_cores=1, scr=Path("."), log_lvl="info", verbose=False
    ):
        self.method = method
        self.options = options
        self.n_cores = n_cores
        self.scr = scr
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        logging.basicConfig(
            stream=sys.stdout, level=logging.getLevelName(log_lvl.upper())
        )

    def __repr__(self):
        return f"{self.method.upper()}({self.show_options(self.options)}, n_cores={self.n_cores})"

    def show_options(self, options):
        list_options = []
        for k in options:
            if options[k] is None:
                list_options.append(f"{k}")
            else:
                list_options.append(f"{k}:{options[k]}")
        str_options = ", ".join(list_options)
        return str_options

    def check_conformer(self, mol):
        if "constrain_atoms" in self.options or "constrain_bonds" in self.options:
            assert (
                mol.GetNumConformers() > 0
            ), "Mol has no conformer. Embed one conformer to constrain atoms to"

    def generate(self, mol, constrain_atoms=None, constrain_bonds=None):
        block = BlockLogs()
        self.check_conformer(mol)
        if constrain_atoms:
            self.options["constrain_atoms"] = constrain_atoms
        if constrain_bonds:
            self.options["constrain_bonds"] = constrain_bonds
        if self.method.lower() == "crest":
            crest.check_crest(self.logger)
            crest.run_crest(mol, self.options, self.n_cores, self.scr, self.logger)
        elif self.method.lower() == "etkdg":
            etkdg.run_etkdg(mol, self.options, self.n_cores, self.verbose)
            # Run follow up optimizations and clustering/filtering/etc.
            if "opt_flow" in self.options:
                for action in self.options["opt_flow"]:
                    if constrain_atoms and isinstance(action, GeomOptimizer):
                        action.options["constrain_atoms"] = constrain_atoms
                    mol = action.run(
                        mol, n_cores=self.n_cores, verbose=self.verbose, scr=self.scr
                    )
        else:
            raise Warning(f"{self.method} is not a valid option")
        del block
        return mol
