from typing import Set
import re
import torch
from torch.optim import Adam
from tqdm import tqdm
from MSCMG.Model import GPT
from MSCMG import Configuration as Configuration
from MSCMG.Dataset import SMILESDataset
from rdkit import Chem

from MSCMG.Generation import get_mol,sample
from unidock_score import run_docking
from Inception import Inception
from torch.nn.utils.rnn import pad_sequence
import os
import time
import numpy as np
import argparse

def gen_smiles (config,dataset,model,block_size):

    regex = re.compile(config.regex_pattern)
    mconf = config.model_config
    completions = []
    molecules_set = []
    uniq_y = []
    uniq_log_probs = []


    pbar = tqdm()

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
        # 调用 sample 函数，生成 y
        log_probs,y = sample(model, x, block_size, temperature=mconf.generation_params["temp"])
        
        
        # 遍历生成的分子 y
        for gen_mol,log_prob in zip(y, log_probs):

            # 将生成的张量 gen_mol 转换为 SMILES 字符串 completion。
            completion = "".join([dataset.itos[int(i)] for i in gen_mol])
            completions.append(completion)

            # 检查 SMILES 是否包含起始符号 "!" 和结束符号 "~"，并提取 mol_string
            if completion[0] == "!" and completion[1] == "~":
                completion = "!" + completion[2:]
            if "~" not in completion:
                continue
            mol_string = completion[1 : completion.index("~")]
            # 通过 get_mol 将 mol_string 转换为分子对象。
            mol = get_mol(mol_string)

            if mol is not None:
                
                # 将分子转换为规范化 SMILES，并添加到 molecules_list 和 molecules_set。
                canonic_smile = Chem.MolToSmiles(mol)

                molecules_set.append(canonic_smile)
                # 添加有效的 gen_mol 和 log_prob 到相应的列表
                uniq_y.append(gen_mol)
                uniq_log_probs.append(log_prob)
                
                # 更新分子集合进度。
                pbar.update()
                pbar.set_description(
                    f"Generated {len(molecules_set)} canonical smiles"
                    )
                # 检查当前生成的分子数量是否达到目标：
        if len(molecules_set) >= mconf.generation_params["batch_size"]:
            molecules_set = molecules_set[:mconf.generation_params["batch_size"]]
            uniq_y = uniq_y[:mconf.generation_params["batch_size"]]
            uniq_log_probs = uniq_log_probs[:mconf.generation_params["batch_size"]]
            break

    # while len(molecules_set) < 128:
    #     # 使用正则表达式 regex 匹配 context，将其转换为数值张量 x
    #     # x 重复 batch_size 次以形成批次并传输到设备。
    #     x = (
    #         torch.tensor(
    #             [
    #                 dataset.stoi[s]
    #                 for s in regex.findall(mconf.generation_params["context"])
    #             ],
    #             dtype=torch.long,
    #         )[None, ...]
    #         .repeat(mconf.generation_params["batch_size"], 1)
    #         .to(mconf.device)
    #     )

    #     # 调用 sample 函数，生成 y
    #     log_probs, y = sample(model, x, block_size, temperature=mconf.generation_params["temp"])
    #     uniq_y = []
    #     uniq_log_probs = []

    #     # 遍历生成的分子 y
    #     for gen_mol, log_prob in zip(y, log_probs):

    #         # 将生成的张量 gen_mol 转换为 SMILES 字符串 completion。
    #         completion = "".join([dataset.itos[int(i)] for i in gen_mol])
    #         completions.append(completion)

    #         # 检查 SMILES 是否包含起始符号 "!" 和结束符号 "~"，并提取 mol_string
    #         if completion[0] == "!" and completion[1] == "~":
    #             completion = "!" + completion[2:]
    #         if "~" not in completion:
    #             continue

    #         mol_string = completion[1:completion.index("~")]

    #         # 通过 get_mol 将 mol_string 转换为分子对象。
    #         mol = get_mol(mol_string)

    #         if mol is not None:
    #             # 将分子转换为规范化 SMILES，并添加到 molecules_list 和 molecules_set。
    #             canonic_smile = Chem.MolToSmiles(mol)

                

    #             # 如果 molecules_set 超过 128，则删除多余分子
    #             if len(molecules_set) == 128:
    #                 break
                    
    #             molecules_set.append(canonic_smile)
    #             # 添加有效的 gen_mol 和 log_prob 到相应的列表
    #             uniq_y.append(gen_mol)
    #             uniq_log_probs.append(log_prob)

    #             # 更新分子集合进度。
    #             pbar.update()
    #             pbar.set_description(
    #                 f"Generated {len(molecules_set)} canonical smiles"
    #             )

    # pbar.close()


 

    return uniq_y,uniq_log_probs,molecules_set

def train_with_reinforcement_learning(config,Agent_model_weight_path, Prior_model_weight_path,
                                      sigma=20,
                                      n_epoch=1000,
                                      experience_replay=1,
                                      save_dir ="./data",
                                      dock_file_dir="./ledock",
                                      work_dir="./ledock_",
                                      save_work_dir=False):

    mconf = config.model_config
    dataset = SMILESDataset()
    dataset.load_desc_attributes(config.pretrain_desc_path + config.training_fname.split(".")[0] + ".yaml")
    mconf.set_dataset_attributes(
        vocab_size=dataset.vocab_size, block_size=dataset.block_size
    )
    Agent_model = GPT(mconf).to(mconf.device)
    Prior_model = GPT(mconf).to(mconf.device)

    # Load the model
    Agent_model.load_state_dict(
        torch.load(
            Agent_model_weight_path,
            map_location=torch.device(mconf.device),
        )
    )
    Prior_model.load_state_dict(
        torch.load(
            Prior_model_weight_path,
            map_location=torch.device(mconf.device),
        )
    )

    Agent_model.to(mconf.device)
    torch.compile(Agent_model)
    block_size = Agent_model.get_block_size()
    print("The models have been loaded.")
    optimizer = Adam(Agent_model.parameters(), lr=1e-3)

    # Early stopping
    early_stopping_criteria = 0.01 
    best_loss = float('inf')
    loss_history = []
    patience = 0
    max_patience = 5 #config.model_config.train_params.get('patience', 5)

    config.set_generation_parameters(
        target_number=500,
        target_criterion="force_number_unique",
        load_model_weight=Agent_model_weight_path,
        dataset_desc_path=config.pretrain_desc_path + config.training_fname.split(".")[0] + ".yaml"
    )

    experience = Inception()
    start_time = time.time()
    step_score = [[], []]
    for epoch in range(n_epoch):# config.model_config.train_params['epochs']
        total_loss = 0
        agent_seqs,agent_likelihood,generated_smiles = gen_smiles(config,dataset,Agent_model,block_size)

        Agent_model.train() 
        agent_seqs = pad_sequence(agent_seqs, batch_first=True, padding_value=16)
        print("agent_seqs-pad",agent_seqs.shape)
                   
        prior_likelihood = Prior_model.likelihood(agent_seqs)
        print("prior_likelihood",prior_likelihood.shape)

        receptor = "ledock/DRD2_target.pdbqt"
        result_dir = "docking_results"
        pdbqt_dir = "intermediate_pdbqt"
        kwargs = {
            "search_mode": "balance",
            "scoring": "vina",
            "center_x": 10,
            "center_y": 6,
            "center_z": -9,
            "size_x": 30,
            "size_y": 30,
            "size_z": 30,
            "num_modes": 1
        }
        smiles,scores= run_docking(generated_smiles,receptor=receptor,result_dir=result_dir,pdbqt_dir=pdbqt_dir,keep_files=False,**kwargs)
        k = -10
        score = [max(_score, k) / k for _score in scores]
        score, scaffold, scaf_fp = experience.update_score(smiles, score)
        score = torch.tensor(score, dtype=torch.float32, requires_grad=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        score = score.to(device)
        augmented_likelihood = prior_likelihood + sigma * score
        agent_likelihood = torch.tensor(agent_likelihood, dtype=torch.float32)  # 确保数据类型匹配
        agent_likelihood = agent_likelihood.to(device)
        
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_seqs = torch.tensor(exp_seqs) if isinstance(exp_seqs, np.ndarray) else exp_seqs
            exp_seqs = exp_seqs.to(device)
            exp_agent_likelihood  = Agent_model.likelihood(exp_seqs.long())
            exp_agent_likelihood = exp_agent_likelihood.to(device)            
            if not isinstance(exp_score, torch.Tensor):
                exp_score = torch.tensor(exp_score, dtype=torch.float32)
            if not isinstance(exp_prior_likelihood, torch.Tensor):
                exp_prior_likelihood = torch.tensor(exp_prior_likelihood, dtype=torch.float32)
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score           
            exp_augmented_likelihood = exp_augmented_likelihood.to(device)
            exp_loss = torch.pow((exp_augmented_likelihood - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        smiles = smiles.detach().cpu().numpy() if isinstance(smiles, torch.Tensor) else smiles
        agent_seqs = agent_seqs.detach().cpu().numpy() if isinstance(agent_seqs, torch.Tensor) else agent_seqs
        score = score.detach().cpu().numpy() if isinstance(score, torch.Tensor) else score
        prior_likelihood = prior_likelihood.detach().cpu().numpy() if isinstance(prior_likelihood, torch.Tensor) else prior_likelihood
        scaffold = scaffold.detach().cpu().numpy() if isinstance(scaffold, torch.Tensor) else scaffold
        scaf_fp = scaf_fp.detach().cpu().numpy() if isinstance(scaf_fp, torch.Tensor) else scaf_fp
        experience.add_experience(smiles,agent_seqs,score, prior_likelihood, scaffold, scaf_fp)
        # Calculate loss
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Agent_model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        current_loss = loss.item()
        loss_history.append(current_loss)

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_epoch - epoch) / (epoch + 1)))
        print("\n       Step {}   Generate SMILES: {}  Time elapsed: {:.2f}h Time left: {:.2f}h"
        .format(epoch+1, len(smiles), time_elapsed, time_left))
	    
        sorted_indices = np.argsort(score)[::-1]
        sorted_score = score[sorted_indices]
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       sorted_score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(epoch + 1)
        step_score[1].append(np.mean(score))
        print(f"Epoch {epoch + 1}, Total Loss: {current_loss}")

        # Update the best model (if the current model has a lower loss).
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(Agent_model.state_dict(), f'best_model_{epoch + 1}'+'.pth')
            print(f"New best model saved with loss :{best_loss}")

        # Plot the loss curve and save it every 5 epochs.
        if (epoch + 1) % 5 == 0:
            # plt.figure(figsize=(10, 5))
            # plt.plot(loss_history, label='Loss')
            # plt.title('Loss Over Epochs')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(f'./Loss_fig/loss_plot_epoch_{epoch + 1}.png')
            # plt.close()
            # print(f"Loss plot saved for epoch {epoch + 1}")

            # Save memory and score
            experience.save_memory(os.path.join(save_dir, "memory.csv"))
            with open(os.path.join(save_dir, 'step_score.csv'), 'w') as f:
                f.write("step,score\n")
                for s1, s2 in zip(step_score[0], step_score[1]):
                    f.write(str(s1) + ',' + str(s2) + "\n")

        # Early stopping judgment
        if abs(best_loss - current_loss) < early_stopping_criteria:
            patience += 1
            if patience >= max_patience:
                print("Early stopping criteria met. Training stopped.")
                break
        else:
            patience = 0

    return Agent_model, optimizer


def filter_valid_smiles(smiles_list):
    valid_smiles = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:  
                valid_smiles.append(smi)
        except:
            continue  
    return valid_smiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reinforcement Learning Stage")
    parser.add_argument("--agent_model_path", type=str,
                        default="1_Pretraining/model_weights/pretrain6_al0_DRD2.pt",
                        help="Path to the agent model weights")
    parser.add_argument("--prior_model_path", type=str,
                        default="5_ActiveLearning/model_weights/pretrain6_al1_DRD2主动学习.pt",
                        help="Path to the prior model weights")
    args = parser.parse_args()

    base_path = os.getcwd()
    config = Configuration.Config(
        base_path=base_path,
        verbose=True,  
        cycle_prefix="pretrain",
        cycle_suffix="DRD2",
        al_iteration=0,
        training_fname="combined_train.csv.gz",
        validation_fname="combined_valid.csv.gz",
    )

    Agent_model, optimizer = train_with_reinforcement_learning(config,
                                                               Agent_model_weight_path=args.agent_model_path,
                                                               Prior_model_weight_path=args.prior_model_path,
                                                               sigma=20,
                                                               n_epoch=1000,
                                                               experience_replay=1,
                                                               save_dir="RL_data",
                                                               dock_file_dir="./ledock",
                                                               work_dir="./ledock_",
                                                               save_work_dir=False)