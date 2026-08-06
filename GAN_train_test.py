from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from pprint import pprint, pformat

import tqdm
import os
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

class ECGDataset(Dataset):
    def __init__(self, ecgs, diagnoses):
        self.ecgs = torch.tensor(ecgs, dtype=torch.float32)
        self.diagnoses = torch.tensor(diagnoses, dtype=torch.float32)

    def __len__(self):
        return len(self.ecgs)

    def __getitem__(self, idx):
        return self.ecgs[idx], self.diagnoses[idx]


def organise_train_data_multiclass( ecg_dataset, test_size = 0.2, random_state = 42):
    
    #Split the real and generated data
    real_data = []
    real_labels = []
    gen_data = []
    gen_labels = []
    for pair in ecg_dataset:
        real_label = pair.real.label
        
        #Append the data only if real label is not already present in the list of real labels
        if real_label[:3].upper() in real_labels:
            pass
        else:
            real_labels_vect = [real_label[:3].upper()]*len(pair.real.data)#.split("_")[1]
            real_labels = real_labels + real_labels_vect #append(pair.real.label)
            real_data.append(pair.real.data)
        
        gen_data.append(pair.fake.data)
        fake_label = pair.fake.label
        gen_labels_vect = [fake_label.split("_")[1].upper()]*len(pair.fake.data)
        gen_labels = gen_labels + gen_labels_vect #append(pair.fake.label)
    
    # Convert labels to uppercase for consistency with real labels    
    # gen_labels = [s.split("_", 1)[0].upper() for s in gen_labels]
    
    real_data = np.concatenate(real_data, axis=0)
    gen_data = np.concatenate(gen_data, axis=0)
    
    
    #check real-fake ecg shape so that the last dimension is 12 (leads)
    if real_data.shape[-1] == 12:
        real_data =np.transpose(real_data, (0, 2, 1))
    elif real_data.shape[1] == 12:
        pass
    else:
        raise ValueError(f"Expected one dimension to have size 12, got {real_data.shape}")
    
    if gen_data.shape[-1] == 12:
        gen_data =np.transpose(gen_data, (0, 2, 1))
    elif gen_data.shape[1] == 12:
        pass
    else:
        raise ValueError(f"Expected one dimension to have size 12, got {gen_data.shape}")
    
    
    #Encode the labels to integers
    label_encoder = LabelEncoder()
    real_labels = label_encoder.fit_transform(real_labels)
    fake_labels = label_encoder.transform(gen_labels)
    int2class = {index: class_name for index, class_name in enumerate(label_encoder.classes_) } #dict of mapping

    print("Encoding labels dictionary: ", int2class)
    
    fake_labels = label_encoder.transform(gen_labels)
    
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(real_data, real_labels,test_size=test_size, 
                                                        random_state=random_state, stratify=real_labels)
    
    X_fake_train, X_fake_test, y_fake_train, y_fake_test = train_test_split(gen_data, fake_labels, test_size=test_size, 
                                                        random_state=random_state, stratify=fake_labels)
    
    print(f"Real data train-test split: {X_real_train.shape[0]}-{X_real_test.shape[0]}")
    print(f"Fake data train-test split: {X_fake_train.shape[0]}-{X_fake_test.shape[0]}")
    
    return X_real_train, X_real_test, y_real_train, y_real_test, X_fake_train, X_fake_test, y_fake_train, y_fake_test, int2class




def organise_train_data_multilabel( ecg_dataset, test_size = 0.2, random_state = 42):
    
    #Split the real and generated data
    real_data = []
    real_labels = []
    gen_data = []
    gen_labels = []
   
    for pair in ecg_dataset:
        real_label = pair.real.label
        
        #Append the data only if real label is not already present in the list of real labels
        if real_label[:3].upper() in real_labels:
            pass
        else:
            real_labels_vect = [real_label[:3].upper()]*len(pair.real.data)#.split("_")[1]
            real_labels = real_labels + real_labels_vect #append(pair.real.label)
            real_data.append(pair.real.data)
        
        gen_data.append(pair.fake.data)
        fake_label = pair.fake.label
        gen_labels_vect = [fake_label.split("_")[1].upper()]*len(pair.fake.data)
        gen_labels = gen_labels + gen_labels_vect #append(pair.fake.label)
    
    # Convert labels to uppercase for consistency with real labels    
    # gen_labels = [s.split("_", 1)[0].upper() for s in gen_labels]
    
    real_data = np.concatenate(real_data, axis=0)
    gen_data = np.concatenate(gen_data, axis=0)
    
    
    #check real-fake ecg shape so that the last dimension is 12 (leads)
    if real_data.shape[-1] == 12:
        real_data =np.transpose(real_data, (0, 2, 1))
    elif real_data.shape[1] == 12:
        pass
    else:
        raise ValueError(f"Expected one dimension to have size 12, got {real_data.shape}")
    
    if gen_data.shape[-1] == 12:
        gen_data =np.transpose(gen_data, (0, 2, 1))
    elif gen_data.shape[1] == 12:
        pass
    else:
        raise ValueError(f"Expected one dimension to have size 12, got {gen_data.shape}")
    
    #Split stratified my the multilabel labels
    splitter_real = MultilabelStratifiedShuffleSplit(test_size=test_size, random_state=random_state)
    splitter_fake = MultilabelStratifiedShuffleSplit(test_size=test_size, random_state=random_state)
    
    real_index_train, real_index_test = next(splitter_real.split(real_data, real_labels))
    fake_index_train, fake_index_test = next(splitter_fake.split(gen_data, gen_labels))
    
    X_real_train = real_data[real_index_train]
    X_real_test = real_data[real_index_test]
    y_real_train = real_labels[real_index_train]
    y_real_test = real_labels[real_index_test]
    
    X_fake_train = gen_data[fake_index_train]
    X_fake_test = gen_data[fake_index_test]
    y_fake_train = gen_labels[fake_index_train]
    y_fake_test = gen_labels[fake_index_test]
    
    print(f"Real data train-test split: {X_real_train.shape[0]}-{X_real_test.shape[0]}")
    print(f"Fake data train-test split: {X_fake_train.shape[0]}-{X_fake_test.shape[0]}")
    
    return X_real_train, X_real_test, y_real_train, y_real_test, X_fake_train, X_fake_test, y_fake_train, y_fake_test,




def train_aux_classifier(train_loader, val_loader, train_aux_model , epochs = 10, lr = 0.001, multilabel = False, device = None):
    
    if multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else: #MultiClass classification
        criterion = nn.CrossEntropyLoss()

    
    train_aux_model = train_aux_model.to(device)
    optimizer = torch.optim.Adam(train_aux_model.parameters(), lr=lr)
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in tqdm.tqdm(range(epochs), desc="Train AUX Model", unit="epoch"): 
        
        train_aux_model.train()
        train_running_loss = 0.0
        val_running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = train_aux_model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item() * inputs.size(0)
        
        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        
        train_aux_model.eval()
        with torch.no_grad():    
            for inputs, labels in val_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = train_aux_model(inputs)
                loss = criterion(outputs, labels.long())
                val_running_loss += loss.item() * inputs.size(0)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        
    
    return train_aux_model, (train_loss_history, val_loss_history)
  
  
  
def predict_aux_classifier(model, test_loader, device=None, multilabel=False):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)

            if multilabel:
                predicted = (torch.sigmoid(logits) >= 0.5).int()
            else:
                predicted = logits.argmax(dim=1)

            predictions.append(predicted.cpu())

    return torch.cat(predictions)
    
    
def compute_scores_and_save(gan_labels, gan_preds, baseline_labels, baseline_preds, int2label, 
                            name = "", save_path = None, multilabel = False):
    
    gan_accuracy = accuracy_score(gan_labels, gan_preds)
    baseline_accuracy = accuracy_score(baseline_labels, baseline_preds)

    report_kwargs = {
        "digits": 2,
        "zero_division": 0
    }

    if not multilabel:
        class_ids = sorted(int2label)
        report_kwargs["labels"] = class_ids
        report_kwargs["target_names"] = [int2label[i] for i in class_ids]

    gan_report = classification_report( gan_labels, gan_preds, **report_kwargs )
    
    baseline_report = classification_report( baseline_labels, baseline_preds, **report_kwargs )

    output = (
        f"GAN-train accuracy: {gan_accuracy:.4f}\n"
        f"Baseline accuracy: {baseline_accuracy:.4f}\n\n"
        f"--------------------------------\n"
        f"{name} classification report\n"
        f"{gan_report}\n\n"
        f"--------------------------------\n"
        f"Baseline classification report\n"
        f"{baseline_report}"
    )

    if save_path is not None:
        filename = f"{name}_scores.txt" if name else "GAN_scores.txt"
        with open(os.path.join(save_path, filename), "w") as file:
            file.write(output)
    else:
        print(output)

    return gan_accuracy, baseline_accuracy

        
    
def plot_losses(train_loss_gantrain, val_loss_gantrain, train_loss_gantest, val_loss_gantest):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # GAN-train
    axes[0].plot(train_loss_gantrain, label="Train")
    axes[0].plot(val_loss_gantrain, label="Validation")
    axes[0].set_title("GAN-train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # GAN-test
    axes[1].plot(train_loss_gantest, label="Train")
    axes[1].plot(val_loss_gantest, label="Validation")
    axes[1].set_title("GAN-test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    
    return

def compute_gan_train_test_scores(gan_train_model, gan_test_model, ecg_dataset, train_config, 
                                  plot_loss_hist = True, multilabel=False, save_folder = None):

    """
    This is a multiclass classifier, so CE loss is used.
    Input: 
        classifier_model: Pytorch model for ECG classification
        ecg_dataset: ECG dataset with labels
        train_config: dict with the fields I need for the training.
    """
    
    device = train_config.get("device")
    test_size = train_config.get("test_size", 0.2)
    random_state = train_config.get("random_state", 42)
    batch_size = train_config.get("batch_size", 32)
    epochs = train_config.get("epochs", 10)
    lr = train_config.get("lr", 0.001)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if multilabel:
        (X_real_train, X_real_test, 
         y_real_train, y_real_test, 
         X_fake_train, X_fake_test,
         y_fake_train, y_fake_test) = organise_train_data_multilabel(ecg_dataset, test_size=test_size, random_state=random_state)
        int2label = {}
    else:
        (X_real_train, X_real_test, 
        y_real_train, y_real_test, 
        X_fake_train, X_fake_test,
        y_fake_train, y_fake_test, int2label) = organise_train_data_multiclass(ecg_dataset, test_size=test_size, random_state=random_state)
    
    
    
    train_real_dataset = ECGDataset(X_real_train, y_real_train)
    test_real_dataset = ECGDataset(X_real_test, y_real_test)
    
    train_fake_dataset = ECGDataset(X_fake_train, y_fake_train)
    test_fake_dataset = ECGDataset(X_fake_test, y_fake_test)
    
    train_real_loader = DataLoader(train_real_dataset, batch_size=batch_size, shuffle=True)
    test_real_loader = DataLoader(test_real_dataset, batch_size=batch_size, shuffle=False)
    
    train_fake_loader = DataLoader(train_fake_dataset, batch_size=batch_size, shuffle=True)
    test_fake_loader = DataLoader(test_fake_dataset, batch_size=batch_size, shuffle=False)

    
    #Train the GAN train model on FAKE DATA
    print("GAN train model training on FAKE DATA...")
    gan_train_model, (train_loss_gantrain, val_loss_gantrain) = train_aux_classifier(train_fake_loader, test_fake_loader, 
                                                                                     gan_train_model, epochs = epochs, lr = lr, 
                                                                                     device = device, multilabel = multilabel)

    #Train the GAN test model on REAL DATA
    print("\n")
    print("GAN test model training on REAL DATA...")
    gan_test_model, (train_loss_gantest, val_loss_gantest) = train_aux_classifier(train_real_loader, test_real_loader, 
                                                                                     gan_test_model, epochs = epochs, lr = lr, 
                                                                                     device = device, multilabel = multilabel)
    
    if plot_loss_hist:
        plot_losses(train_loss_gantrain, val_loss_gantrain, train_loss_gantest, val_loss_gantest)
        
    if save_folder is not None:
        save_gan_train_model_path = os.path.join(save_folder, "gan_train_model.pth")
        save_gan_test_model_path = os.path.join(save_folder, "gan_test_model.pth")
        torch.save(gan_train_model.state_dict(), save_gan_train_model_path)
        torch.save(gan_test_model.state_dict(), save_gan_test_model_path)
        
    ## GAN TRAIN SCORE
    gan_train_preds = predict_aux_classifier(gan_train_model, test_real_loader, device=device, multilabel=multilabel)
    gan_train_baseline_preds = predict_aux_classifier(gan_train_model, test_fake_loader, device=device, multilabel=multilabel)
    
    gan_train_score, baseline_train_score =  compute_scores_and_save(y_real_test, gan_train_preds, 
                                                                     y_fake_test, gan_train_baseline_preds, 
                                                                     int2label, name = "GAN_train ", save_path = save_folder,
                                                                     multilabel = multilabel)    
    
    
    ## GAN TEST SCORE
    gan_test_preds = predict_aux_classifier(gan_test_model, test_fake_loader, device=device, multilabel=multilabel)
    gan_test_baseline_preds = predict_aux_classifier(gan_test_model, test_real_loader, device=device, multilabel=multilabel)
    gan_test_score, baseline_test_score =  compute_scores_and_save(y_fake_test, gan_test_preds, 
                                                                    y_real_test, gan_test_baseline_preds, 
                                                                    int2label, name = "GAN_test ", save_path = save_folder,
                                                                    multilabel = multilabel)
    
    
    scores = { "GAN-train": {
        "score": gan_train_score,
        "baseline": baseline_train_score, },
    "GAN-test": {
        "score": gan_test_score,
        "baseline": baseline_test_score,
    }, }

    # Pretty-print to console
    pprint(scores)

    # Save pretty-printed dictionary to a text file
    if save_folder is not None :
        save_scores_name = os.path.join(save_folder, "gan_scores.txt") 
        with open(save_scores_name, "w") as file:
            file.write(pformat(scores, sort_dicts=False))

    return