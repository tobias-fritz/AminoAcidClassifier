import unittest
from unittest.mock import patch, mock_open
from src.data_prep import download_pdb, list_all_pdbids, get_amino_acids, get_amino_acids_from_file, center_residue_to_first_atom, rotate_residue, augment_dataset
import torch
from src.data import AminoAcidDataset

class TestDataPrep(unittest.TestCase):

    @patch('src.data_prep.requests.get')
    def test_download_pdb(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "PDB content"
        pdb_id = "1234"
        result = download_pdb(pdb_id)
        self.assertEqual(result, "PDB content")

    @patch('src.data_prep.urlopen')
    def test_list_all_pdbids(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.readlines.return_value = [
            b"HEADER\n", b"HEADER\n", b"1234\n", b"5678\n"
        ]
        result = list_all_pdbids()
        self.assertEqual(result, ["1234", "5678"])

    def test_get_amino_acids(self):
        pdb_content = (
            "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n"
            "ATOM      2  CA  ALA A   1      12.104  14.207  11.000  1.00 20.00           C\n"
            "ATOM      3  C   ALA A   1      13.104  15.207  12.000  1.00 20.00           C\n"
            "ATOM      4  O   ALA A   1      14.104  16.207  13.000  1.00 20.00           O\n"
            "ATOM      5  CB  ALA A   1      15.104  17.207  14.000  1.00 20.00           C\n"
            "ATOM      6  N   ARG A   2      16.104  18.207  15.000  1.00 20.00           N\n"
        )
        result = get_amino_acids(pdb_content)
        expected = [
            "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n",
            "ATOM      2  CA  ALA A   1      12.104  14.207  11.000  1.00 20.00           C\n",
            "ATOM      3  C   ALA A   1      13.104  15.207  12.000  1.00 20.00           C\n",
            "ATOM      4  O   ALA A   1      14.104  16.207  13.000  1.00 20.00           O\n",
            "ATOM      5  CB  ALA A   1      15.104  17.207  14.000  1.00 20.00           C\n",
            "END\n"
        ]
        self.assertEqual(result, expected)

    @patch('builtins.open', new_callable=mock_open, read_data="ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n")
    def test_get_amino_acids_from_file(self, mock_file):
        pdb_file = "dummy.pdb"
        result = get_amino_acids_from_file(pdb_file)
        expected = [
            "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n",
            "END\n"
        ]
        self.assertEqual(result, expected)

class TestAugmentation(unittest.TestCase):

    def test_center_residue_to_first_atom(self):
        coordinates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected = torch.tensor([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])
        result = center_residue_to_first_atom(coordinates)
        self.assertTrue(torch.equal(result, expected))

    def test_rotate_residue(self):
        coordinates = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = rotate_residue(coordinates, 90, 0, 0)
        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_augment_dataset(self):
        class MockDataset(AminoAcidDataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                coordinates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
                elements = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
                residue = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                return coordinates, elements, residue

        dataset = MockDataset()
        output_file = 'test_output.txt'
        augment_dataset(dataset, output_file, n_orientations=1)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 5)  # 3 atoms + 1 END line + 1 newline

if __name__ == '__main__':
    unittest.main()
