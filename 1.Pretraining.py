from MCMG import Configuration as Configuration
from MCMG import Dataset as Dataset
from MCMG import Training

import os

base_path = os.getcwd()

config = Configuration.Config(

    # Base path, used to define the root directory for storing data and model files.
    base_path=base_path,
    # cycle_prefix, al_iteration, cycle_suffix: these three parameters are used to construct filenames
    cycle_prefix="pretrain",
    cycle_suffix="DRD2",
    al_iteration=0,  # use 0 for pretraining
    # Specify the filenames for training and validation data
    training_fname="combined_train.csv.gz",
    validation_fname="combined_valid.csv.gz",
    slice_data=None,
    verbose=True, 
)
config.set_training_parameters(mode="Pretraining", epochs=1)
datasets = Dataset.load_data(config=config, mode="Pretraining",desc_path="1_Pretraining/datasets_descriptors/combined_train.yaml")
model, trainer = Training.train_GPT(
    config=config, training_dataset=datasets[0], validation_dataset=datasets[1]
)