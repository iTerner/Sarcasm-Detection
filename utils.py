import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch.
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    return correct.sum() / len(correct)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_file(path: str):
    """
    The function load the pickle file and returns him
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_file_pickle(data: object, path: str):
    """
    The function saves the given data in a pickle file with the given path
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_max_sentence_length(d: dict) -> int:
    """
    The function returns the max length sentence
    """
    max_len = 0
    for key, val in d.items():
        if len(val) > max_len:
            max_len = len(val)
    return max_len


def pad_sentences(d: dict) -> list:
    """
    The function pad the sentences with respect to the max length sentence.
    """
    max_len = get_max_sentence_length(d)
    tensor_dict = {}
    len_dict = {}
    for sen_id, sen_vecs in tqdm(d.items()):
        len_dict[sen_id] = len(sen_vecs)
        new_vecs = []
        for i in range(max_len):
            if i < len(sen_vecs):
                new_vecs.append(torch.from_numpy(sen_vecs[i]))
            else:
                new_vecs.append(torch.zeros(200))
        tensor_dict[sen_id] = torch.stack(new_vecs)
    return tensor_dict, len_dict


def plot(x, y, plot_type, save_path):
    """
    The function create a plot and saves it
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = os.path.join(save_path, f"train_{plot_type}.png")
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(plot_type)
    plt.title(f"Training {plot_type} per Epoch")
    plt.grid()
    plt.savefig(path)
    plt.clf()
