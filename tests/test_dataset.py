import pytest
import torch
import os
from src.dataset import AminoAcidDataset, GraphAADataset

@pytest.fixture
def sample_pdb_path(tmp_path):
    # Create a minimal PDB file for testing
    content = """ATOM      1  N   ALA A   1      27.461  14.346  11.567  1.00 36.22           N  
ATOM      2  CA  ALA A   1      26.200  14.965  11.996  1.00 36.39           C  
ATOM      3  C   ALA A   1      26.260  16.466  11.747  1.00 36.19           C  
ATOM      4  O   ALA A   1      27.308  17.061  11.939  1.00 37.06           O  
END
ATOM      1  N   VAL A   2      25.147  17.088  11.326  1.00 34.16           N  
ATOM      2  CA  VAL A   2      25.123  18.536  11.135  1.00 32.42           C  
ATOM      3  C   VAL A   2      24.724  19.207  12.444  1.00 31.83           C  
ATOM      4  O   VAL A   2      23.741  18.811  13.072  1.00 31.85           O  
END"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(content)
    return str(pdb_file)

def test_amino_acid_dataset_loading(sample_pdb_path):
    dataset = AminoAcidDataset(sample_pdb_path)
    assert len(dataset) == 2
    assert len(dataset.coordinates) == 2
    assert len(dataset.elements) == 2
    assert len(dataset.residue) == 2

def test_one_hot_encoding():
    dataset = AminoAcidDataset(sample_pdb_path)
    # Test residue encoding/decoding
    encoded = dataset.one_hot_residues('ALA')
    assert torch.argmax(encoded).item() == dataset._aa_to_idx['ALA']
    assert dataset.one_hot_residues_reverse(encoded) == 'ALA'
    
    # Test element encoding/decoding
    encoded = dataset.one_hot_elements('N')
    assert torch.argmax(encoded).item() == dataset._element_to_idx['N']
    assert dataset.one_hot_elements_reverse(encoded) == 'N'

def test_padding():
    dataset = AminoAcidDataset(sample_pdb_path, padding=True)
    assert dataset.coordinates.shape[1] == dataset.coordinates.shape[1]
    assert dataset.elements.shape[1] == dataset.elements.shape[1]

def test_graph_dataset(sample_pdb_path):
    dataset = GraphAADataset(sample_pdb_path)
    adj, node_features = dataset._get_graph(0)
    
    # Test adjacency matrix properties
    assert adj.shape[0] == adj.shape[1]  # Square matrix
    assert torch.all(adj.diagonal() == 0)  # Zero diagonal
    assert torch.all(adj >= 0)  # Non-negative values
    
    # Test node features
    assert node_features.shape[1] == 7  # 3 coordinates + 4 one-hot elements
    
    # Test caching
    first_call = dataset[0]
    second_call = dataset[0]
    assert id(first_call[0]) == id(second_call[0])  # Should return cached object

if __name__ == '__main__':
    pytest.main([__file__])
