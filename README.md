# Amino Acid Classification

This project focuses on the classification of amino acids using a Convolutional Neural Network (CNN). The dataset is derived from PDB files, and the model is trained to predict the type of amino acid based on atomic coordinates and elements.

## Notebooks

### [data.ipynb](data/data.ipynb)

This notebook focuses on data extraction and preprocessing.

#### Key Sections:

1. **Extracting Amino Acids**
    - Extract amino acids from PDB files.
    - Filter and clean the data.

2. **Saving Processed Data**
    - Save the processed data to new PDB files.

### [residues.ipynb](residues.ipynb)

This notebook contains the main workflow for the project, including data preprocessing, model training, and evaluation.

#### Key Sections:

1. **Data Loading and Preprocessing**
    - Load the amino acid dataset from PDB files.
    - Preprocess the data by extracting residues and augmenting the dataset with rotations.

2. **Model Definition**
    - Define the `AminoAcidCNN` model architecture.

3. **Training the Model**
    - Train the model using the training dataset.
    - Fine-tune the pretrained model for additional epochs.

4. **Evaluation**
    - Evaluate the model's performance on the test dataset.
    - Plot the training loss and accuracy.

5. **Visualization**
    - Visualize the results, including the model's predictions and the actual labels.

## Usage

### Requirements

- Python 3.8+
- PyTorch
- Matplotlib
- Seaborn

### Running the Notebooks

1. Clone the repository.
2. Install the required packages.
3. Open the notebooks in Jupyter and run the cells sequentially.

### Training the Model

To train the model, run the cells in [residues.ipynb](residues.ipynb). The training process includes data loading, model definition, training, and evaluation.

### Evaluating the Model

After training, evaluate the model's performance using the test dataset. The evaluation metrics and visualizations are provided in the notebook.

## Results

The model achieves high accuracy in classifying amino acids. The training and evaluation results are visualized in the notebooks.

## License

This project is licensed under the MIT License.

---

For more details, refer to the individual notebooks and the code comments.