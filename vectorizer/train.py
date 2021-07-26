import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.optim as optim
from online_triplet_loss.losses import batch_hard_triplet_loss

import time, random, os, json, joblib
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
from glob import glob
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from vectorizer.model import Model
from vectorizer.utils import preprocess, AverageMeter, Full_layer

def main():
    model = Model()
    datapath =  config.get("VECTORIZER", "datapath")
    trainsets = json.loads(config.get("DATA", "trainsets"))
    testsets = json.loads(config.get("DATA", "testsets"))
    train_datasets = [torchaudio.datasets.LIBRISPEECH(datapath, url=trainset, download=True) for trainset in trainsets]
    test_datasets = [torchaudio.datasets.LIBRISPEECH(config.get("VECTORIZER", "datapath"), url=testset, download=True) for testset in testsets]
    num_test_speakers = sum([len(glob(os.path.join(datapath, "LibriSpeech", t, "*"))) for t in testsets])

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    kwargs = {'num_workers': int(mp.cpu_count()), 'pin_memory': False} if torch.cuda.is_available() else {}

    train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config.getint("VECTORIZER", "batch_size"),
                                    shuffle=True,
                                    collate_fn=lambda x: preprocess(x),
                                    drop_last=True,
                                    **kwargs)

    test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    drop_last=True,
                                    **kwargs)

    torch.manual_seed(config.get("SEED", "value"))
    device = torch.device("cuda:0")
    print(f'Number of model parameters: {sum([p.data.nelement() for _, p in model.named_parameters() ]):,}')

    speaker_id_dict = joblib.load(config.get("VECTORIZER", "train_labels_dict"))
    num_train_classes = len(speaker_id_dict)
    print(f"Number of speakers in training set: {num_train_classes}")
    fc = Full_layer(config.getint("VECTORIZER", "embeddings_size"), num_train_classes)
    print(f"Number of speakers in test set: {num_test_speakers}")

    optimizer = optim.Adam(
            [{'params': model.parameters()}, {'params': fc.parameters()}], 
            lr=config.getfloat("VECTORIZER", "lr")
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    ce_criterion = torch.nn.CrossEntropyLoss().to(device)
    start_epoch = 0
    model.to(device)
    fc.to(device)
    best_acc = 0
    best_val_prec_path = config.get("VECTORIZER", "trained_model")

    try:
        for epoch in range(start_epoch, config.getint("VECTORIZER", "epochs")):
            # train for one epoch
            now = time.time()
            train(train_loader, model, optimizer, ce_criterion, fc, epoch, device)

            # evaluate on validation set
            val_acc = validate(model, test_loader, device)
            print(f"Current validation accuracy: {val_acc:.2f}%")

            scheduler.step(val_acc)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            print(f'Best accuracy: {best_acc:.2f}%')
            print(f"Ran epoch {epoch+1} in {time.time()-now:.1f} seconds")
            
            # Save model with highest validation accuracy
            if is_best:
                print(f"Saving new best validation accuracy model to {best_val_prec_path}")

                model_state_dict = model.state_dict() 
                torch.save({
                    'best_acc': best_acc,
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'scheduler': scheduler,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, best_val_prec_path)

    except KeyboardInterrupt:
        print("Manual interrupt")


def train(train_loader, model, optimizer, ce_criterion, fc, epoch, device) -> None:
    """Train for one epoch on the training set"""
    triplet_losses = AverageMeter()
    ce_losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    batch_idx = 0
    for x, target in train_loader:
        x = x.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        optimizer.zero_grad()
        features = model(input_var, train=True)
        output = fc(features)
        ce_loss = ce_criterion(output, target_var)
        accuracies.update(accuracy_score(target_var.detach().cpu().numpy(), torch.argmax(output, dim=1).detach().cpu().numpy()))

        triplet_loss = batch_hard_triplet_loss(target_var, features, margin=1, device=device)
        triplet_losses.update(triplet_loss.data.item(), x.size(0))
        ce_losses.update(ce_loss.data.item(), x.size(0))
        loss = config.getfloat("VECTORIZER", "triplet_loss_alpha") * triplet_loss + ce_loss

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if batch_idx % config.getint("VECTORIZER", "log_interval") == 0:
            string = f"Epoch: {epoch+1}\tTriplet Loss: {triplet_losses.value:.3f}\tCE Loss: {ce_losses.value:.3f}\tAccuracy: {accuracies.value*100:.2f}%"
            print(string)

        batch_idx += 1


def validate(model, test_loader, device) -> float:
    # switch to evaluate mode
    model.eval()
    validate_dict = defaultdict(list)
    print(f"Computing enrollment and test vectors")
    for waveform, _, _, speaker_id, _, _ in tqdm(test_loader):
        speaker_id = speaker_id.item()
        audio = np.squeeze(waveform.cpu().numpy())[None, :]
        audio = torch.from_numpy(audio.astype(np.float32)).to(device)
        embeddings = model(audio, train=False)
        embeddings = list(embeddings.cpu().detach().numpy())
        validate_dict[speaker_id].append(embeddings)

    accuracy = 0
    print(f"Running {config.getint('VECTORIZER', 'validation_trials')} tests...")
    for _ in tqdm(range(config.getint("VECTORIZER", "validation_trials"))):
        X = list()
        y = list()
        X_test = list()
        y_test = list()

        for label, embeddings in validate_dict.items():
            enrol_x = embeddings.pop(random.randrange(len(embeddings)))
            X.extend(enrol_x)
            y.extend([label]*len(enrol_x))

            for embs in embeddings:
                X_test.extend(embs) 
                y_test.extend([label]*len(embs))

        neigh = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        neigh.fit(X, y)

        accuracy += neigh.score(X_test, y_test)
    return (accuracy/config.getint("VECTORIZER", "validation_trials")) * 100

   
if __name__ == "__main__": 
    main()