import tempfile

from rdkit import Chem

from confgen.generator import BaseConformerGenerator
from confgen.utils import normal_termination, stream


class OBabel(BaseConformerGenerator):
    def __init__(self, n_cores: int = 1, scr: str = ".") -> None:
        super().__init__(n_cores, scr)

    def generate(self, mol: Chem.Mol, **kwargs) -> Chem.Mol:
        tempdir = tempfile.mkdtemp()
        input_file = tempdir + "/mol.sdf"
        output_file = tempdir + "/obabel_out.sdf"
        Chem.SDWriter(input_file).write(mol)
        lines = list(
            stream(f"obabel -i sdf {input_file} -o sdf -O {output_file} --gen3d")
        )
        if not normal_termination(lines, "1 molecule converted"):
            raise Warning(f"OpenBabel failed with error:\n{''.join(lines)}")
        suppl = Chem.SDMolSupplier(output_file, removeHs=False, sanitize=False)
        mol = suppl[0]
        return mol
