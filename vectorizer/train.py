import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.optim as optim
from online_triplet_loss.losses import batch_hard_triplet_loss

import time, random, os, json
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

from vectorizer.model import Model
from vectorizer.utils import preprocess, AverageMeter

def main():
    model = Model()
    
    trainsets = json.loads(config.get("DATA", "trainsets"))
    train_datasets = [torchaudio.datasets.LIBRISPEECH(config.get("VECTORIZER", "datapath"), url=trainset, download=True) for trainset in trainsets]
    testsets = json.loads(config.get("DATA", "testsets"))
    test_datasets = [torchaudio.datasets.LIBRISPEECH(config.get("VECTORIZER", "datapath"), url=testset, download=True) for testset in testsets]

    kwargs = {'num_workers': int(mp.cpu_count()), 'pin_memory': False} if torch.cuda.is_available() else {}

    train_loaders = [DataLoader(dataset=train_dataset,
                                    batch_size=config.getint("VECTORIZER", "batch_size"),
                                    shuffle=True,
                                    collate_fn=lambda x: preprocess(x),
                                    drop_last=True,
                                    **kwargs) for train_dataset in train_datasets]

    test_loaders = [DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    drop_last=True,
                                    **kwargs) for test_dataset in test_datasets]

    torch.manual_seed(config.get("SEED", "value"))
    device = torch.device("cuda:0")
    print(f'Number of model parameters: {sum([p.data.nelement() for name, p in model.named_parameters() ]):,}')

    optimizer = optim.SGD([{'params': model.parameters()}],
                                lr=config.getfloat("VECTORIZER", "lr"),
                                momentum=config.getfloat("VECTORIZER", "momentum"),
                                weight_decay=config.getfloat("VECTORIZER", "weigth_decay"))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    start_epoch = 0
    model.to(device)
    best_acc = 0

    try:
        for epoch in range(start_epoch, config.getint("VECTORIZER", "epochs")):
            # train for one epoch
            now = time.time()
            train(train_loaders, model, optimizer, epoch, device, best_acc)

            # evaluate on validation set
            val_acc = validate(model, test_loaders, device)
            print(f"Current validation accuracy: {val_acc:.2f}%")

            scheduler.step(val_acc)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            print(f'Best accuracy: {best_acc:.2f}%')
            print("Ran epoch {} in {:.1f} seconds".format(epoch, time.time()-now))
            
            # Save model with highest validation accuracy
            if is_best:
                best_val_prec_path = config.get("VECTORIZER", "trained_model")
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


def train(train_loaders, model, optimizer, epoch, device, best_acc) -> None:
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    triplet_losses = AverageMeter()

    train_batches_num = sum([len(train_loader) for train_loader in train_loaders])

    # switch to train mode
    model.train()

    end = time.time()
    batch_idx = 0
    for train_loader in train_loaders:
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            input_var = torch.autograd.Variable(x)
            target_var = torch.autograd.Variable(target)

            features = model(input_var, train=True)

            triplet_loss = batch_hard_triplet_loss(target_var, features, margin=1, device=device)
            triplet_losses.update(triplet_loss.data.item(), x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            triplet_loss.backward()

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % config.getint("VECTORIZER", "log_interval") == 0:
                string = ('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                        'Triplet Loss {triplet_loss.value:.4f} ({triplet_loss.ave:.4f})\t'.format(
                        epoch, batch_idx+1, train_batches_num, batch_time=batch_time,
                        triplet_loss=triplet_losses))

                print(string)

            batch_idx += 1


def validate(model, test_loaders, device) -> float:
    # switch to evaluate mode
    model.eval()
    validate_dict = defaultdict(list)
    val_batches_num = sum([len(test_loader) for test_loader in test_loaders])
    print(f"Computing enrollment and test vectors")
    pbar = tqdm(total=val_batches_num)
    for test_loader in test_loaders:
        for waveform, _, _, speaker_id, _, _ in test_loader:
            speaker_id = speaker_id.item()
            audio = np.squeeze(waveform.cpu().numpy())[None, :]
            audio = torch.from_numpy(audio.astype(np.float32)).to(device)
            embeddings = model(audio, train=False)
            embeddings = list(embeddings.cpu().detach().numpy())
            validate_dict[speaker_id].append(embeddings)
            pbar.update(1)

    accuracy = 0
    print(f"Running {config.getint('VECTORIZER', 'validation_trials')} tests...")
    for idx, _ in tqdm(enumerate(range(config.getint("VECTORIZER", "validation_trials")))):
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

        neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        neigh.fit(X, y)

        accuracy += neigh.score(X_test, y_test)
    return (accuracy/config.getint("VECTORIZER", "validation_trials")) * 100

   
if __name__ == "__main__": 
    main()