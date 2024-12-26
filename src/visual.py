import matplotlib.pyplot as plt
import torch
from torch import nn
from typing import Dict, Tuple, Any
from .data import AminoAcidDataset

def plot_aa_distributions(dataset: AminoAcidDataset) -> Dict[str, Any]:
    """
    Plots the distribution of amino acids in the given dataset.

    Args:
        dataset (AminoAcidDataset): The dataset containing amino acid information.

    Returns:
        Dict[str, Any]: A dictionary containing the figure, axis, and counter dictionary.
    """
    counter_dict = {
                "ALA":0, "ARG":0, "ASN":0, "ASP":0, "CYS":0, "GLN":0, "GLU":0, "GLY":0, 
               "HIS":0, "ILE":0, "LEU":0, "LYS":0, "MET":0, "PHE":0, "PRO":0, "SER":0, 
               "THR":0, "TRP":0, "TYR":0, "VAL":0}
    
    # Count occurrences of each amino acid in the dataset
    for i in range(len(dataset)):
        _,_, residue = dataset[i]
        residue = dataset.one_hot_residues_reverse(residue)
        counter_dict[residue] += 1

    # Plot the distribution as a pie chart
    fig, ax = plt.subplots(figsize=(10,10))
    ax.pie(counter_dict.values(), labels=counter_dict.keys(), autopct='%1.1f%%')
    ax.set_title('Amino acid distribution')
    
    return {"fig":fig, "ax":ax, "counter":counter_dict}

def plot_predicted_vs_true(model: nn.Module, dataloader_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
    """
    Plots the predicted vs true amino acid distribution.

    Args:
        model (nn.Module): The trained model for prediction.
        dataloader_tuple (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing coordinates, elements, and residue tensors.

    Returns:
        Dict[str, Any]: A dictionary containing the figure and axis.
    """
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
                   'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 
                   'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    coordinates, elements, residue = dataloader_tuple
    input_data = torch.cat((coordinates, elements), dim=1)
    input_data = input_data.unsqueeze(0).unsqueeze(0)

    model.eval()
    output = model(input_data)

    # Plot the predicted distribution as a bar chart
    fig, ax = plt.subplots(figsize=(6,2.1))
    ax.bar(range(20), output.detach().numpy().squeeze())
    ax.set_xticks(range(20))
    ax.set_xticklabels(amino_acids, rotation=45)
    correct_prediction = (torch.argmax(output) == torch.argmax(residue))
    
    # Set the title with prediction result
    ax.set_title(f'The prediction is {correct_prediction},\n'
                 f'actual label: ({amino_acids[int(torch.argmax(residue))]})')

    ax.set_ylim(0, 1)
    ax.set_xlim(-1, 20)

    return {"fig": fig, "ax": ax}

