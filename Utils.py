from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
from tqdm import tqdm


# 用于计算SA
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def get_MolLogP(mol):
    return Descriptors.MolLogP(mol) 

def get_TPSA(mol):
    return Chem.rdMolDescriptors.CalcTPSA(mol)

def get_QED(mol):
    return QED.qed(mol)

def get_Lipinski_five(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    score = 0
    score = sum([mw <= 500, logp <= 5, h_donors <= 5, h_acceptors <= 10])
    if score >= 4:
        return 5  # bonus point
    return score

def get_SA(mol):
    try:
        sa = sascorer.calculateScore(mol)
    except:
        sa = -1
    return sa

def get_MolWt(mol):
    return Descriptors.MolWt(mol)


def filter_smiles_by_properties(smiles_list, target_properties):


    # Initialize an empty list to store the filtered SMILES
    filtered_smiles = []

    for smile in tqdm(smiles_list):
        try:
            # Parse the SMILES string to get the molecule object
            mol = Chem.MolFromSmiles(smile)

            # Calculate the properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            sa = get_SA(mol)
            qed = get_QED(mol)

            score = 0
 
            
            if abs(mw - target_properties["MolWt"]["value"]) <= target_properties["MolWt"]["std_dev"]:
                score += 1
            if abs(logp - target_properties["MolLogP"]["value"]) <= target_properties["MolLogP"]["std_dev"]:
                score += 1
            if abs(tpsa - target_properties["TPSA"]["value"]) <= target_properties["TPSA"]["std_dev"]:
                score += 1
            if abs(sa - target_properties["SA"]["value"]) <= target_properties["SA"]["std_dev"]:
                score += 1
            if abs(qed - target_properties["QED"]["value"]) <= target_properties["QED"]["std_dev"]:
                score += 1
            if score >= 3:
                filtered_smiles.append(smile)

        except Exception as e:
            # Handle any parsing or calculation errors
            print(f"Error processing SMILES: {smile}, Error: {e}")

    return filtered_smiles



def filter_smiles_from_trainset_by_properties(smiles_properties, target_properties):
    filtered_smiles = []

    for index, smile in tqdm(smiles_properties.iterrows()):
        try:
            score = 0
            if abs(smile["MolWt"] - target_properties["MolWt"]["value"]) <= target_properties["MolWt"]["std_dev"]:
                score += 1
            if abs(smile["MolLogP"] - target_properties["MolLogP"]["value"]) <= target_properties["MolLogP"]["std_dev"]:
                score += 1
            if abs(smile["TPSA"] - target_properties["TPSA"]["value"]) <= target_properties["TPSA"]["std_dev"]:
                score += 1
            if abs(smile["SA"] - target_properties["SA"]["value"]) <= target_properties["SA"]["std_dev"]:
                score += 1
            if abs(smile["QED"] - target_properties["QED"]["value"]) <= target_properties["QED"]["std_dev"]:
                score += 1

            if score >= 4:
                filtered_smiles.append(smile["SMILES"])

        except Exception as e:
            # Handle any parsing or calculation errors
            print(f"Error processing SMILES: {smile}, Error: {e}")

    return filtered_smiles



def expand_smiles_to_5000(smiles_list):
    if len(smiles_list) >= 5000:
        return smiles_list
    else:
        result_list = smiles_list.copy()
        while len(result_list) < 5000:
            result_list.extend(smiles_list)
        return result_list[:5000]