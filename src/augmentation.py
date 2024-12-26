import torch
from .data import AminoAcidDataset

def center_residue_to_first_atom(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Center the coordinates to the first atom.

    Args:
        coordinates (torch.Tensor): Tensor of atom coordinates.

    Returns:
        torch.Tensor: Centered coordinates.
    """
    # center the coordinates to the first atom
    first_atom = coordinates[0]
    centered_coordinates = coordinates - first_atom
    
    return centered_coordinates

def rotate_residue(coordinates: torch.Tensor, phi: float, psi: float, theta: float) -> torch.Tensor:
    """
    Rotate the residue by given angles around the x, y, and z axes.

    Args:
        coordinates (torch.Tensor): Tensor of atom coordinates.
        phi (float): Rotation angle around the x-axis in degrees.
        psi (float): Rotation angle around the y-axis in degrees.
        theta (float): Rotation angle around the z-axis in degrees.

    Returns:
        torch.Tensor: Rotated coordinates.
    """
    phi, psi, theta = torch.deg2rad(torch.tensor(phi)), torch.deg2rad(torch.tensor(psi)), torch.deg2rad(torch.tensor(theta))
    # rotate the molecule by phi degrees around the x-axis
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, torch.cos(phi), -torch.sin(phi)],
                                    [0, torch.sin(phi), torch.cos(phi)]])
    rotated_residue = torch.matmul(coordinates, rotation_matrix)
    # rotate the molecule by psi degrees around the y-axis
    rotation_matrix = torch.tensor([[torch.cos(psi), 0, torch.sin(psi)],
                                    [0, 1, 0],
                                    [-torch.sin(psi), 0, torch.cos(psi)]])
    rotated_residue = torch.matmul(rotated_residue, rotation_matrix)
    # rotate the molecule by theta degrees around the z-axis
    rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                    [torch.sin(theta), torch.cos(theta), 0],
                                    [0, 0, 1]])
    rotated_residue = torch.matmul(rotated_residue, rotation_matrix)

    return rotated_residue

def augment_dataset(dataset: AminoAcidDataset, output_file: str, n_orientations: int = 1) -> None:
    """
    Augment the dataset by generating multiple orientations of each residue.

    Args:
        dataset (AminoAcidDataset): The dataset to augment.
        output_file (str): The file to save the augmented dataset.
        n_orientations (int, optional): Number of orientations to generate for each residue. Defaults to 1.
    """
    residue_to_name = {
        0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS', 5: 'GLN', 6: 'GLU',
        7: 'GLY', 8: 'HIS', 9: 'ILE', 10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE',
        14: 'PRO', 15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL'
    }
    element_to_name = ['C', 'N', 'O', 'S']

    augmented_lines = []
    for i in range(len(dataset)):
        coordinates, elements, residue = dataset[i]
        for _ in range(n_orientations):
            centered_residue = center_residue_to_first_atom(coordinates)
            rotated_coordinates = rotate_residue(
                centered_residue,
                torch.randint(0, 360, (1,)).item(),
                torch.randint(0, 360, (1,)).item(),
                torch.randint(0, 360, (1,)).item()
            )
            for j in range(len(rotated_coordinates)):
                augmented_lines.append(
                    f'ATOM {j + 1:6d} {element_to_name[int(torch.argmax(elements[j]))]:>2}   '
                    f'{residue_to_name[int(torch.argmax(residue))]:>3}     1    '
                    f'{rotated_coordinates[j][0]:8.3f}{rotated_coordinates[j][1]:8.3f}'
                    f'{rotated_coordinates[j][2]:8.3f}    1.00  0.00\n'
                )
            augmented_lines.append('END\n')
    with open(output_file, 'w') as f:
        f.writelines(augmented_lines)


