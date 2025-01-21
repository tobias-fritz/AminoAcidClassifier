import torch
from torch import nn

class AminoAcidDataset(torch.utils.data.Dataset):
    '''AminoAcidDataset class
    
    This class loads the amino acid dataset from a PDB file and stores the coordinates and elements of each atom in the structure.

    Args:
        pdb_file (str): The path to the PDB file.
        padding (bool): If True, the coordinates and elements will be padded to have the same length for all residues.
    '''

    def __init__(self, pdb_file: str, padding: bool = False) -> None:
        self._amino_acids = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 
            'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
            'TYR', 'VAL'
        ]
        self._elements = ['C', 'N', 'O', 'S']

        self.coordinates = []
        self.elements = []
        self.residue = []
        self.input_shape = None

        with open(pdb_file, 'r') as f:
            residue_cords = []
            residue_elements = []
            residue_name = []

            for line in f:
                if line.startswith('ATOM'):
                    residue_cords.append([
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54])
                    ])
                    residue_elements.append(self.one_hot_elements(line[13:14].strip()))
                    residue_name.append(self.one_hot_residues(line[17:20].strip()))
                elif line.startswith('END'):
                    self.coordinates.append(torch.tensor(residue_cords))
                    self.elements.append(torch.stack(residue_elements))
                    self.residue.append(residue_name[0])
                    residue_cords, residue_elements, residue_name = [], [], []
        
        if padding:
            self.coordinates = nn.utils.rnn.pad_sequence(self.coordinates, batch_first=True)
            self.elements = nn.utils.rnn.pad_sequence(self.elements, batch_first=True)
            self.input_shape = self._get_input_shape()
    
    def one_hot_residues(self, residue: str) -> torch.Tensor:
        one_hot = torch.zeros(len(self._amino_acids))
        one_hot[self._amino_acids.index(residue)] = 1
        return one_hot

    def one_hot_elements(self, element: str) -> torch.Tensor:
        one_hot = torch.zeros(len(self._elements))
        one_hot[self._elements.index(element)] = 1
        return one_hot
    
    def one_hot_residues_reverse(self, residue: torch.Tensor) -> str:
        return self._amino_acids[residue.argmax().item()]
    
    def one_hot_elements_reverse(self, element: torch.Tensor) -> str:
        return self._elements[element.argmax().item()]
    
    def _get_input_shape(self) -> tuple:
        return self.coordinates.shape[1], len(self._elements)
    
    def __getitem__(self, idx: int) -> tuple:
        return self.coordinates[idx], self.elements[idx], self.residue[idx]
    
    def __len__(self) -> int:
        return len(self.coordinates)