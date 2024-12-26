# The main function loop
# it should have a mode thats either training or prediction
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.data_prep import AminoAcidDataset
from src.model import AminoAcidCNN
from src.train import train_model, evaluate_model


def main(mode: str,
         prediction_input_path: str = None,
         model_path: str = None,
         training_fpath: str = 'data/amino_acids_augmented.pdb',
         n_epochs: int = 100,
         train_test_split: float = 0.8,      
         model_out_path: str = None   
         ) -> None:
    

    if mode == "train":

        # Print out the settings
        print(f'Training model with the following settings:')
        print(f'  Training file: {training_fpath}')
        print(f'  Number of epochs: {n_epochs}')
        print(f'  Train/test split: {train_test_split}')
        print(f'  Model output path: {model_out_path if model_out_path is not None else "model.pth"}')

        # Load the training data
        dataset = AminoAcidDataset(training_fpath, padding=True)

        train_size = int(train_test_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=True)

        N, M = dataset.input_shape
        model = AminoAcidCNN(input_channels=1, # Since we added an extra dimension for batch and channel
                            input_height=N, # Number of atoms
                            input_width=3+M) # 3 coordinates + 4 element types
        
        criterion = nn.CrossEntropyLoss() # Use cross entropy loss for classification
        optimizer = optim.Adam(model.parameters(), lr=0.001) # Use Adam optimizer

        training_dict = train_model(model=model, 
                                    data_loader=train_dataloader, 
                                    criterion=criterion, 
                                    optimizer=optimizer, 
                                    n_epochs=n_epochs)

        training_dict.update(evaluate_model(training_dict["model"], test_dataloader))

        model_out_path = 'model.pth' if model_out_path is None else model_out_path
        torch.save(model.state_dict(), model_out_path)

        # save the training results in json file with the same name as the model
        training_results = model_out_path.replace('.pth', '_training_results.json')
        with open(training_results, 'w') as f:
            json.dump(training_dict, f)

    elif mode == "predict":

        # assert that the model path and prediction input path are provided
        assert model_path is not None, 'Please provide a model path'
        assert prediction_input_path is not None, 'Please provide a prediction input path'

        # Load the model
        model = AminoAcidCNN(input_channels=1, input_height=20, input_width=7)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Load the input data
        dataset = AminoAcidDataset(prediction_input_path, padding=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        coordinates, elements, residue = next(iter(dataloader))
        input_data = torch.cat((coordinates, elements), dim=2)
        input_data = input_data.unsqueeze(1)

        # Make a prediction
        output = model(input_data)
        prediction = dataset.one_hot_residues_reverse(torch.argmax(output).item())

        print(f'The predicted residue is: {prediction}')
