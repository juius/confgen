import copy
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

from rdkit import Chem
from rdkit.Chem import AllChem

from confgen.generator import BaseConformerGenerator
from confgen.utils import normal_termination, stream
from confgen.xtb_utils import (
    add_conformer2mol,
    parse_coordline,
    set_threads,
    write_xyz,
    xtb_calculate,
)

CREST_CMD = "crest"

_logger = logging.getLogger("crest")
_logger.setLevel(logging.INFO)


class CREST(BaseConformerGenerator):
    """Conformer Generator using CREST [DOI https://doi.org/10.1039/C9CP06869D]

    Note:
        Valid keywords (https://xtb-docs.readthedocs.io/en/latest/crestcmd.html)
        can be parsed to CREST as flags with `key=True` or `key=None` or parsed
        with a value as `key=value`

        >>> CREST(gfn2=True, alpb=None, mdlen='x0.5')
    """

    def __init__(
        self,
        # gfn2: bool = True,
        ewin: Union[int, float] = 6,
        mquick: bool = True,
        mdlen: str = "x0.5",
        **kwargs,
    ) -> None:
        super().__init__()
        # self.gfn2 = gfn2
        self.ewin = ewin
        self.mquick = mquick
        self.mdlen = mdlen
        self.__dict__.update(kwargs)

    def _preopt(self, mol: Chem.Mol, constrained_atoms=None) -> Chem.Mol:
        """Preoptimize mol object using same method used in CREST.

        Args:
            mol (Chem.Mol): Mol object

        Returns:
            Chem.Mol: Mol with optimized geometry
        """
        # TODO: remove hardcoding
        # TODO: add constrained bonds
        pre_options = {"opt": True, "gfn": 2}
        if constrained_atoms:
            pre_options["constrained_atoms"] = constrained_atoms
        _logger.info(f"Running preoptimization with {pre_options}")
        mol_opt = xtb_calculate(mol, options=pre_options, n_cores=self.n_cores)
        return mol_opt

    @staticmethod
    def _atom_constrains(constrained_atoms, scr):
        """Write .xcontrol file containing atom constrains for CREST."""
        cs = f"{CREST_CMD} mol.xyz -constrain "
        # make atom constrains
        for i in constrained_atoms:
            cs += f"{i},"
        cmd = cs[:-1]  # remove last comma
        lines = stream(cmd, cwd=scr)
        lines = list(lines)
        if not normal_termination(lines, "STOP <.xcontrol.sample> written."):
            _logger.error("".join(lines))
            raise RuntimeError("Could not write Constrains file.")

    @staticmethod
    def _bond_constrains(constrained_bonds, scr):
        """Adds bond constrains to CREST constrain input file."""
        if not (scr / Path(".xcontrol.sample")).exists():
            cmd = f"{CREST_CMD} -constrain mol.xyz "
            lines = stream(cmd, cwd=scr)
            lines = list(lines)
            if not normal_termination(lines, "STOP <.xcontrol.sample> written."):
                _logger.error("".join(lines))
                raise RuntimeError("Could not write Constrains file.")
        with open(scr / Path(".xcontrol.sample"), "r") as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            if "metadyn" in l:
                break
        for bond in constrained_bonds:
            lines.insert(i, f"  distance: {bond[1]}, {bond[0]}, auto\n")
        with open(scr / Path(".xcontrol.sample"), "w") as f:
            f.writelines(lines)

    def _get_crest_cmd(self, constrained_atoms, constrained_bonds, tmp_scr):
        """Generate CREST command."""
        cmd = f"{CREST_CMD} mol.xyz -T {self.n_cores} "
        # Generate CREST Constrains
        if constrained_atoms or constrained_bonds:
            # Remove xcontrol file if already present
            if (tmp_scr / Path(".xcontrol.sample")).exists():
                os.remove(tmp_scr / Path(".xcontrol.sample"))
            cmd += "-cinp .xcontrol.sample "
            if constrained_atoms:
                self._atom_constrains(constrained_atoms, tmp_scr)
            if constrained_bonds:
                self._bond_constrains(constrained_bonds, tmp_scr)
        # Generate CREST Command
        options = {
            k: v for k, v in self.__dict__.items() if k not in ["n_cores", "scr"]
        }

        for key, value in options.items():
            if (value is None) or (not value) or (value is True):
                cmd += f"--{key} "
            elif value:
                cmd += f"--{key} {str(value)} "
        return cmd

    @staticmethod
    def _read_all_conformers(mol, scr):
        """Returns a molecule containing all conformers found by CREST in
        ascending order of energy.

        The electronic energy is stored in the property 'energy'.
        """
        org_confid = mol.GetConformer().GetId()
        # Read all conformers from CREST output
        tmp = Path(scr).glob("crest_conformers*xyz")
        if len(list(tmp)) == 0:
            time.sleep(5)
            tmp = Path(scr).glob("crest_conformers*xyz")
        for crest_file in tmp:
            break
        with open(crest_file, "r") as f:
            lines = f.readlines()
        n_atoms = int(lines[0].lstrip().rstrip("\n"))
        block_len = n_atoms + 2
        n_confs = len(lines) // block_len
        # check if lines divisible by n_atoms+2
        assert len(lines) / block_len == float(n_confs)

        # Append all conformers to mol
        for i in range(n_confs):
            block = lines[i * block_len : i * block_len + block_len]
            energy = float(block[1].lstrip().rstrip("\n"))
            atoms = []
            coords = []
            for line in block[2:]:
                a, c = parse_coordline(line)
                atoms.append(a)
                coords.append(c)
            add_conformer2mol(mol, atoms, coords, energy)
        # Reset Conformers
        mol.RemoveConformer(org_confid)
        confs = mol.GetConformers()
        for i in range(len(confs)):
            confs[i].SetId(i)
        assert mol.GetNumConformers() == n_confs

    @staticmethod
    def check_conformer(mol):
        n_confs = mol.GetNumConformers()
        if n_confs < 1:
            _logger.info("Embed one conformer to start CREST from")
            assert (
                AllChem.EmbedMolecule(mol, useRandomCoords=True) == 0
            ), "Failed to embed molecule"
        if n_confs > 1:
            _logger.info(
                "Multiple conformers in mol. Will start CREST from first conformer"
            )
            confs = mol.GetConformers()
            conf_ids = [conf.GetId() for conf in confs]
            for i in conf_ids[1:]:
                mol.RemoveConformer(i)

    def generate(
        self,
        mol: Chem.Mol,
        constrained_atoms: Optional[list] = None,
        constrained_bonds: Optional[list] = None,
    ) -> Chem.Mol:
        """Generate conformers of mol object using CREST.

        Args:
            mol (Chem.Mol): Mol object
            constrained_atoms (Optional[list]): List of atomids to constrain
            constrained_bonds (Optional[list]): List of atomids of bonds to constrain (id1, id2)

        Returns:
            Chem.Mol: Mol object with conformers embedded

        Note:
            Conformers are sorted in ascending order of energy.
            Energy is stored in mol.GetConformer(0).GetProp("energy").
        """

        set_threads(self.n_cores)
        mol3d = copy.deepcopy(mol)
        self.check_mol(mol3d)
        self.check_conformer(
            mol3d,
        )
        # Run preoptimization
        mol3d = self._preopt(mol3d, constrained_atoms)

        tempdir = tempfile.TemporaryDirectory(dir=self.scr, prefix="CREST_")
        tmp_scr = Path(tempdir.name)
        atoms = [a.GetSymbol() for a in mol3d.GetAtoms()]
        coords = mol3d.GetConformer().GetPositions()
        _ = write_xyz(atoms, coords, tmp_scr)
        cmd = self._get_crest_cmd(constrained_atoms, constrained_bonds, tmp_scr)
        _logger.info(f"Running CREST: {cmd}")
        lines = stream(cmd, cwd=tmp_scr)
        lines = list(lines)
        _logger.info("\n".join(lines))
        if normal_termination(lines, "CREST terminated normally"):
            self._read_all_conformers(mol3d, tmp_scr)
            if constrained_atoms:
                # align conformers
                Chem.rdMolAlign.AlignMolConformers(mol3d, atomIds=constrained_atoms)
            _logger.debug("".join(lines))
        else:
            _logger.error("ABNORMAL TERMINATION")
            _logger.info("".join(lines))

        return mol3d
