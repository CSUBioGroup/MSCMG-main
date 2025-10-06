import torch
from torch.nn import functional as F
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Fragments

from MCMG.Model import GPT
from MCMG.Configuration import Config, AdmetDict
from MCMG.Dataset import SMILESDataset

from typing import Set, List, Callable, Union, Optional, Any, Dict
RDLogger.DisableLog("rdApp.*")



@torch.no_grad()
def sample(
        model: GPT, x: torch.Tensor, steps: int, temperature: float = 1.0
) -> torch.Tensor:
    """Sample sequences from the model.

    Args:
        model (GPT): The GPT model.
        x (torch.Tensor): Input tensor.
        steps (int): Number of sampling steps.
        temperature (float): Sampling temperature.
    """
    # Get the model's block_size, which is the maximum input length the model can handle.
    # If the input sequence exceeds this length, it will be truncated.
    block_size = model.get_block_size()
    # Set the model to evaluation mode to ensure dropout or BatchNorm are not applied during inference.
    model.eval()

    batch_size = x.size(0)  # Automatically get the batch size
    log_probs = torch.zeros(batch_size, device=x.device)  # Initialize log probability accumulation

    # Use a loop to generate a sequence of the specified number of steps.
    # In each iteration, generate one new token, resulting in a sequence of length 'steps'.
    for k in range(steps):
        # x_cond: Construct the input condition x_cond, ensuring the input length does not exceed block_size.
        # If the length of x is greater than block_size, take the last block_size elements of x.
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

        # Feed x_cond into the model to get logits (raw output scores) and hidden states (unused output).
        logits, _ = model(x_cond)  # logits shape: (batch_size, seq_length, vocab_size)
        # Extract the logits of the last time step and scale them by temperature to adjust the distribution.
        # Higher temperature makes the distribution more uniform, increasing randomness.
        logits = logits[:, -1, :] / temperature
        log_prob = F.log_softmax(logits, dim=1)

        # Apply the softmax function to logits to convert them into a probability distribution `probs`,
        # representing the probability of generating each next token.
        probs = F.softmax(logits, dim=-1)
        # Sample an index `ix` from the probability distribution `probs` using torch.multinomial,
        # which represents the index of the next generated token.
        ix = torch.multinomial(probs, 1)
        # Accumulate the log-probabilities of the sampled tokens.
        log_probs += log_prob.gather(dim=-1, index=ix).squeeze(-1)  # shape: (batch_size,)
        # Append the newly generated token `ix` to the input sequence `x` to form the next input for sampling.
        x = torch.cat((x, ix), dim=1)
    return log_probs, x


def restricted_fg_checker(restricted_fgs: List[str]) -> Callable:
    def _check_mol(molecule: rdkit.Chem.Mol) -> bool:
        for restricted_key in restricted_fgs:
            method = getattr(Fragments, restricted_key)
            if method(molecule) != 0:
                return True
        return False

    return _check_mol


def admet_checker(admet_criteria: AdmetDict) -> Callable:
    def _check_mol(
        molecule: rdkit.Chem.Mol,
    ) -> bool:
        for key, _dict in admet_criteria.items():
            func = _dict["func"]
            assert callable(
                func
            ), f"Expected a callable for criteria {key}, got {type(func)}"
            val = func(molecule)
            if val < _dict["lower"] or val > _dict["upper"]:
                return False
        return True

    return _check_mol


def generate_smiles(config: Config):
    """Generate SMILES strings using the model.

    Args:
        config_dict (dict): Configuration dictionary.

    Returns:
        None: The function saves the generated molecules to disk.
    """
    regex = re.compile(config.regex_pattern)
    mconf = config.model_config
    if config.verbose:
        print(f"--- Starting generation")
        print(f"    Loading dataset descriptors ...")
    dataset = SMILESDataset()
    dataset.load_desc_attributes(mconf.generation_params["desc_path"])
    if config.verbose:
        print(f"    Creating a model and loading weights ...")
    mconf.set_dataset_attributes(
        vocab_size=dataset.vocab_size, block_size=dataset.block_size
    )
    model = GPT(mconf).to(mconf.device)
    model.load_state_dict(
        torch.load(
            mconf.generation_params["load_ckpt_path"],
            map_location=torch.device(mconf.device),
        )
    )
    model.to(mconf.device)
    torch.compile(model)

    block_size = model.get_block_size()
    assert (
        block_size == dataset.block_size
    ), "Warning: model block size and dataset block size are different"

    molecules_list: List[str] = []
    molecules_set: Set[str] = set()
    molecules_set_filtered: Set[str] = set()
    completions = []
    pbar = tqdm()
    if config.verbose:
        print(f"    Starting generation ...")
    while True:
        x = (
            torch.tensor(
                [
                    dataset.stoi[s]
                    for s in regex.findall(mconf.generation_params["context"])
                ],
                dtype=torch.long,
            )[None, ...]
            .repeat(mconf.generation_params["batch_size"], 1)
            .to(mconf.device)
        )
        log_probs, y = sample(model, x, block_size, temperature=mconf.generation_params["temp"])

        target_criterion = mconf.generation_params["target_criterion"]
        force_filters = mconf.generation_params["force_filters"]
        if force_filters is not None and "ADMET" in force_filters:
            satisfies_admet = admet_checker(mconf.generation_params["admet_criteria"])
        if force_filters is not None and "FGs" in force_filters:
            contains_restricted_fg = restricted_fg_checker(
                mconf.generation_params["restricted_fgs"]
            )
        for gen_mol in y:
            if target_criterion == "force_number_completions":
                pbar.update()
                pbar.set_description(f"Generated {len(molecules_list)} completions")
            completion = "".join([dataset.itos[int(i)] for i in gen_mol])
            completions.append(completion)
            if completion[0] == "!" and completion[1] == "~":
                completion = "!" + completion[2:]
            if "~" not in completion:
                continue
            mol_string = completion[1 : completion.index("~")]
            mol = get_mol(mol_string)

            if mol is not None:
                if target_criterion == "force_number_unique":
                    pbar.update()
                    pbar.set_description(
                        f"Generated {len(molecules_set)} unique canonical smiles"
                    )
                canonic_smile = Chem.MolToSmiles(mol)
                molecules_list.append(canonic_smile)
                molecules_set.add(canonic_smile)
                if force_filters is not None:
                    mol_passes = True
                    if "ADMET" in force_filters:
                        if not satisfies_admet(mol):
                            mol_passes = False
                    if "FGs" in force_filters:
                        if contains_restricted_fg(mol):
                            mol_passes = False
                    if mol_passes:
                        pbar.update()
                        pbar.set_description(
                            f"Generated {len(molecules_set_filtered)} unique canonical smiles that pass filters"
                        )
                        molecules_set_filtered.add(canonic_smile)
        target_number = mconf.generation_params["target_number"]
        match target_criterion:
            case "force_number_completions":
                if len(molecules_list) >= target_number:
                    break
            case "force_number_unique":
                if len(molecules_set) >= target_number:
                    break
            case "force_number_filtered":
                if len(molecules_set_filtered) >= target_number:
                    break
    pbar.close()

    completions_df = pd.DataFrame({"smiles": completions})
    completions_df.to_csv(config.cycle_temp_params["completions_fname"])

    molecules_df = pd.DataFrame({"smiles": list(molecules_set)})
    molecules_df.to_csv(config.cycle_temp_params["unique_smiles_fname"])

    if force_filters is not None:
        molecules_filtered_df = pd.DataFrame({"smiles": list(molecules_set_filtered)})
        molecules_filtered_df.to_csv(config.cycle_temp_params["filtered_smiles_fname"])

    # characterize_generated_molecules(config_dict, molecules_list)
    return molecules_set


def get_mol(smile_string: str) -> Union[None, rdkit.Chem.Mol]:
    """Get a molecule object from a SMILES string.

    Args:
        smile_string (str): The SMILES string.

    Returns:
        rdkit.Chem.Mol: The molecule object or None if invalid.
    """
    mol = Chem.MolFromSmiles(smile_string)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol