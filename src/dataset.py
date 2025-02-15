import torch
from torch import nn
import numpy as np
from functools import lru_cache

class AminoAcidDataset(torch.utils.data.Dataset):
    _amino_acids = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
        'TYR', 'VAL'
    ]
    _elements = ['C', 'N', 'O', 'S']
    
    # Pre-compute mappings
    _aa_to_idx = {aa: idx for idx, aa in enumerate(_amino_acids)}
    _element_to_idx = {el: idx for idx, el in enumerate(_elements)}

    def __init__(self, pdb_file: str, padding: bool = False) -> None:
        self.coordinates = []
        self.elements = []
        self.residue = []
        
        # Process PDB file more efficiently
        current_data = {'coords': [], 'elements': [], 'residue': None}
        
        for line in open(pdb_file):
            if line.startswith('ATOM'):
                current_data['coords'].append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                current_data['elements'].append(self.one_hot_elements(line[13:14].strip()))
                if not current_data['residue']:
                    current_data['residue'] = self.one_hot_residues(line[17:20].strip())
            elif line.startswith('END') and current_data['coords']:
                self.coordinates.append(torch.tensor(current_data['coords'], dtype=torch.float32))
                self.elements.append(torch.stack(current_data['elements']))
                self.residue.append(current_data['residue'])
                current_data = {'coords': [], 'elements': [], 'residue': None}

        # One hot encoding for residues
        self.residue_encodings = torch.stack([self.one_hot_residues(res) for res in self._amino_acids])
        self.residue = torch.stack(self.residue)


        # One hot encode the elements
        self.elements = [torch.stack(e) for e in self.elements]
        
        
        if padding:
            self.coordinates = nn.utils.rnn.pad_sequence(self.coordinates, batch_first=True)
            self.elements = nn.utils.rnn.pad_sequence(self.elements, batch_first=True)
            self.input_shape = self._get_input_shape()

    def one_hot_residues(self, residue: str) -> torch.Tensor:
        idx = self._aa_to_idx[residue]
        one_hot = torch.zeros(len(self._amino_acids), dtype=torch.float32)
        one_hot[idx] = 1.0
        return one_hot

    def one_hot_elements(self, element: str) -> torch.Tensor:
        idx = self._element_to_idx[element]
        one_hot = torch.zeros(len(self._elements), dtype=torch.float32)
        one_hot[idx] = 1.0
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



class GraphAADataset(AminoAcidDataset):
    '''Dataset for amino acids represented as graphs.'''

    def __init__(self, pdb_file: str, padding: bool = False) -> None:
        super().__init__(pdb_file, padding)
        self._graph_cache = {}

    @lru_cache(maxsize=128)
    def _get_graph(self, idx: int) -> tuple:
        coords = self.coordinates[idx]
        elements = self.elements[idx]
        
        # Create adjacency matrix using vectorized operations
        diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # broadcasting
        distances = torch.norm(diffs, dim=2)
        adj = 1.0 / (distances + 1e-6)
        adj.fill_diagonal_(0)

        # Create node features
        node_features = torch.cat((coords, elements), dim=1)

        return adj, node_features

    def __getitem__(self, idx: int) -> tuple:
        if idx not in self._graph_cache:
            self._graph_cache[idx] = self._get_graph(idx)
        return self._graph_cache[idx], self.residue[idx]
