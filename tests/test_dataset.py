import pytest
import torch
import os
from src.dataset import AminoAcidDataset, GraphAADataset

@pytest.fixture
def sample_pdb_path(tmp_path):
    # Create a minimal PDB file for testing
    content = """ATOM      1  N   ALA     1       0.000   0.000   0.000    1.00  0.00
ATOM      2  C   ALA     1      -0.434   1.384   0.115    1.00  0.00
ATOM      3  C   ALA     1      -1.431   1.739  -0.991    1.00  0.00
ATOM      4  O   ALA     1      -2.469   2.361  -0.741    1.00  0.00
ATOM      5  C   ALA     1       0.773   2.308   0.014    1.00  0.00 
END
ATOM      1  N   VAL     1       0.000   0.000   0.000    1.00  0.00
ATOM      2  C   VAL     1       0.531   0.691   1.165    1.00  0.00
ATOM      3  C   VAL     1       1.897   1.285   0.813    1.00  0.00
ATOM      4  O   VAL     1       2.140   2.452   1.104    1.00  0.00
ATOM      5  C   VAL     1       0.718  -0.260   2.399    1.00  0.00
ATOM      6  C   VAL     1       1.258   0.511   3.609    1.00  0.00
ATOM      7  C   VAL     1      -0.652  -0.841   2.800    1.00  0.00
END"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(content)
    return str(pdb_file)

def test_amino_acid_dataset_loading(sample_pdb_path):
    dataset = AminoAcidDataset(sample_pdb_path)
    assert len(dataset) == 2
    
    coords, elements, residue_enc = dataset[0]
    assert coords.size(0) == elements.size(0)  # Same number of atoms
    assert residue_enc.size(0) == len(dataset._aa_list)  # One-hot size matches amino acid count
    assert dataset.residue_encodings.size(0) == 2  # Two residues

def test_one_hot_encoding(sample_pdb_path):
    dataset = AminoAcidDataset(sample_pdb_path)
    # Test residue encoding/decoding
    encoded = dataset.one_hot_residues('ALA')
    idx = torch.argmax(encoded).item()
    assert idx == dataset._aa_to_idx['ALA']
    assert dataset.one_hot_residues_reverse(encoded) == 'ALA'
    
    # Test element encoding/decoding
    encoded = dataset.one_hot_elements('N')
    idx = torch.argmax(encoded).item()
    assert idx == dataset._element_to_idx['N']
    assert dataset.one_hot_elements_reverse(encoded) == 'N'

def test_padding(sample_pdb_path):
    dataset = AminoAcidDataset(sample_pdb_path, padding=True)
    coords, elements, residue_enc = dataset[0]
    
    # Test that all amino acids have same number of atoms
    first_shape = coords.size(0)
    assert elements.size(0) == first_shape
    
    # Test that the padding is correct (zero padding)
    assert (coords[5:] == 0).all()
    assert (elements[5:] == 0).all()

def test_graph_dataset(sample_pdb_path):
    dataset = GraphAADataset(sample_pdb_path)
    adj, node_features, residue_enc = dataset._get_graph(0)
    
    # Test adjacency matrix properties
    assert adj.size(0) == adj.size(1)  # Square matrix
    assert (adj.diagonal() == 0).all()  # Zero diagonal
    assert (adj >= 0).all()  # Non-negative values
    
    # Test node features
    assert node_features.size(1) == 7  # 3 coordinates + 4 elements
    
    # Test residue encoding
    assert residue_enc.size(0) == len(dataset._aa_list)  # One-hot size matches amino acid count
    
    # Test caching
    first_call = dataset[0]
    second_call = dataset[0]
    assert all(torch.equal(a, b) for a, b in zip(first_call, second_call))

if __name__ == '__main__':
    pytest.main([__file__])
