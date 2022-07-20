import os
import shutil
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from confgen.utils import normal_termination, stream
from confgen.xtb_utils import (
    add_conformer2mol,
    check_xtb,
    parse_coordline,
    set_threads,
    write_xyz,
)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

CREST_CMD = "crest"


def check_crest(logger):
    # """Check CREST executable and xTB"""
    # lines = stream(f"{CREST_CMD} --version")
    # lines = list(lines)
    # print(lines)
    # if normal_termination(lines, "FITNESS FOR A PARTICULAR PURPOSE"):
    #     for l in lines:
    #         if "Compatible with xTB version" in l:
    #             req_xtb_v = l.split()[-1]
    #             check_xtb(req_xtb_v, logger)
    # else:
    #     raise Warning(f"Could not find CREST ({CREST_CMD})")
    pass

def atom_constrains(options, scr):
    """Write .xcontrol file containing atom constrains for CREST"""
    cs = f"{CREST_CMD} mol.xyz -constrain "
    # make atom constrains
    for i in options["constrain_atoms"]:
        cs += f"{i},"
    cmd = cs[:-1]  # remove last comma
    lines = stream(cmd, cwd=scr)
    lines = list(lines)
    if not normal_termination(lines, "<.xcontrol.sample> written"):
        print("Could not write Constrains file")
        print("".join(lines))


def bond_constrains(options, scr):
    """Adds bond constrains to CREST constrain input file"""
    if not (scr / Path(".xcontrol.sample")).exists():
        cmd = f"{CREST_CMD} -constrain mol.xyz "
        lines = stream(cmd, cwd=scr)
        lines = list(lines)
        if not normal_termination(lines, "<.xcontrol.sample> written"):
            print("Could not write Constrains file")
            print("".join(lines))
    with open(scr / Path(".xcontrol.sample"), "r") as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if "metadyn" in l:
            break
    for bond in options["constrain_bonds"]:
        lines.insert(i, f"  distance: {bond[1]}, {bond[0]}, auto\n")
    with open(scr / Path(".xcontrol.sample"), "w") as f:
        f.writelines(lines)


def get_crest_cmd(n_cores, scr, options):
    """Generate CREST command"""
    cmd = f"{CREST_CMD} mol.xyz -T {n_cores} "
    # Generate CREST Constrains
    if any([k in options for k in ["constrain_atoms", "constrain_bonds"]]):
        # Remove xcontrol file if already present
        if (scr / Path(".xcontrol.sample")).exists():
            os.remove(scr / Path(".xcontrol.sample"))
        cmd += f"-cinp .xcontrol.sample "
    if "constrain_atoms" in options.keys():
        atom_constrains(options, scr)
        del options["constrain_atoms"]
    if "constrain_bonds" in options.keys():
        bond_constrains(options, scr)
        del options["constrain_bonds"]
    # Generate CREST Command
    for key, value in options.items():
        if (value is None) or (not value):
            cmd += f"--{key} "
        elif value:
            cmd += f"--{key} {str(value)} "
    return cmd


def read_all_conformers(mol, scr):
    """
    Returns a molecule containing all conformers found by CREST in ascending order of energy.
    The electronic energy is stored in the property 'energy'.
    """
    org_confid = mol.GetConformer().GetId()
    # Read all conformers from CREST output
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
        for l in block[2:]:
            a, c = parse_coordline(l)
            atoms.append(a)
            coords.append(c)
        add_conformer2mol(mol, atoms, coords, energy)
    # Reset Conformers
    mol.RemoveConformer(org_confid)
    confs = mol.GetConformers()
    for i in range(len(confs)):
        confs[i].SetId(i)
    assert mol.GetNumConformers() == n_confs


def check_conformer(mol, logger):
    n_confs = mol.GetNumConformers()
    if n_confs < 1:
        logger.info("Embed one conformer to start CREST from")
        assert (
            AllChem.EmbedMolecule(mol, useRandomCoords=True) == 0
        ), "Failed to embed molecule"
    if n_confs > 1:
        logger.info("Multiple conformers in mol. Will start CREST from first conformer")
        confs = mol.GetConformers()
        conf_ids = [conf.GetId() for conf in confs]
        for i in conf_ids[1:]:
            mol.RemoveConformer(i)


def run_crest(mol, options, n_cores, scr, logger):
    """Runs CREST and returns rdkit.mol object with conformers and energies as property 'energy'"""
    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix=f"CREST_")
    tmp_scr = Path(tempdir.name)
    set_threads(n_cores)
    check_conformer(mol, logger)
    try:
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        xyz_file = write_xyz(atoms, coords, tmp_scr)
        cmd = get_crest_cmd(n_cores, tmp_scr, options)
        logger.info(f"Running CREST:\n{cmd}")
        lines = stream(cmd, cwd=tmp_scr)
        lines = list(lines)
        if normal_termination(lines, "CREST terminated normally"):
            read_all_conformers(mol, tmp_scr)
            if "constrains" in options:
                # align conformers
                Chem.rdMolAlign.AlignMolConformers(mol, atomIds=options["constrains"])
            logger.debug("".join(lines))
        else:
            logger.error("ABNORMAL TERMINATION")
            logger.info("".join(lines))
    finally:
        shutil.rmtree(tmp_scr)
