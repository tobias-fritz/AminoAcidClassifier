import torch
from torchmetrics import Accuracy, Precision, Recall
from torch.utils.data import DataLoader

def train_model(model: torch.nn.Module, 
                data_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                n_epochs: int) -> dict:
    """
    Train a PyTorch model.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        n_epochs (int): Number of epochs to train the model.

    Returns:
        dict: A dictionary containing the trained model, optimizer, total loss, and learning rate.
    """
    total_loss = []
    epoch_lr = []
    model.train()  # Set the model to training mode

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for coordinates, elements, residue in data_loader:
            # Combine coordinates and elements into a single tensor
            input_data = torch.cat((coordinates, elements), dim=2)
            input_data = input_data.unsqueeze(1)

            # Forward pass
            output = model(input_data)

            # Calculate the loss
            target = torch.argmax(residue, dim=1)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear the gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Accumulate the loss
            epoch_loss += loss.item()

        # Calculate the average loss for the epoch
        epoch_loss /= len(data_loader)
        total_loss.append(epoch_loss)
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # If the loss is not decreasing in the last 5 epochs, reduce the learning rate
        if len(total_loss) > 5 and all(total_loss[-i] >= total_loss[-i-1] for i in range(1, 6)):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f'Reducing learning rate to {param_group["lr"]}')

        # If the loss is not decreasing in the last 10 epochs, stop training
        if len(total_loss) > 10 and all(total_loss[-i] >= total_loss[-i-1] for i in range(1, 11)):
            print('Stopping training')
            break

        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.5f}', end='\r')

    training_dict = { 'model': model, 'optimizer': optimizer, 'total_loss': total_loss, 'learning_rate': epoch_lr }
    
    return training_dict


def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> dict:
    """
    Evaluate a PyTorch model on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation data.

    Returns:
        dict: A dictionary containing the accuracy of the model on the evaluation data.
    """
    model.eval()  # Set the model to evaluation mode
    accuracy =  Accuracy(task='MULTICLASS', num_classes=20)  # Adjust num_classes as needed
    accuracy.reset() # Reset the running accuracy
    precision = Precision(num_classes=20, average='macro', task='MULTICLASS')
    precision.reset()
    recall = Recall(num_classes=20, average='macro', task='MULTICLASS')
    recall.reset()

    with torch.no_grad():
        for coordinates, elements, residue in data_loader:
            # Combine coordinates and elements into a single tensor
            input_data = torch.cat((coordinates, elements), dim=2)
            input_data = input_data.unsqueeze(1)

            # Forward pass
            output = model(input_data)

            # Calculate the accuracy
            target = torch.argmax(residue, dim=1)
            accuracy.update(output, target)
            precision.update(output, target)
            recall.update(output, target)

    return {"accuracy": accuracy.compute().item(), "precision": precision.compute().item(), "recall": recall.compute().item()}

def get_failed_predictions(model: torch.nn.Module, dataset: torch.utils.data.Dataset) -> list:
    """
    Get the failed predictions of a model on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (torch.utils.data.Dataset): Dataset for the evaluation data.

    Returns:
        list: A list of dictionaries containing the actual and predicted labels for the failed predictions.
    """
    failed_predictions = []
    model.eval()  # Set the model to evaluation mode
    resid_dict = dataset._amino_acids
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for coordinates, elements, residue in dataloader:
        input_data = torch.cat((coordinates, elements), dim=2)
        input_data = input_data.unsqueeze(1)
        output = model(input_data)
        target = torch.argmax(residue, dim=1)
        for i in range(len(target)):
            if torch.argmax(output[i]) != target[i]:
                failed_predictions.append({
                    "actual_label": resid_dict[int(target[i])],
                    "predicted_label": resid_dict[int(torch.argmax(output[i]))]
                })

    return failed_predictions