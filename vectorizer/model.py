import torch
import torch.nn as nn
import torchaudio
import torchvision
import torch.nn.functional as F

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

class Normalize(nn.Module):
    """ 
    Scale Audio to be between -1 and 1
    """
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, audio:torch.Tensor):
        if len(audio.shape) != 2:
            raise ValueError("Audio should be 2D: [batch_size X audio_length]")
        if audio.shape[1] < 1:
            raise ValueError("Audio length is zero")
        
        max_value = torch.max(audio, dim=1)[0].detach()
        min_value = torch.min(audio, dim=1)[0].detach()
        max_value = torch.unsqueeze(max_value,1)
        min_value = torch.unsqueeze(min_value,1)
        audio = (audio - min_value) / (max_value - min_value + 1E-10)
        return audio * 2 - 1


class Normalize3D(nn.Module):
    """ 
    Scale Spectrogram to be between 0 and 1
    """
    def __init__(self):
        super(Normalize3D, self).__init__()

    def forward(self, X:torch.Tensor):
        if len(X.shape) != 3:
            raise ValueError("Input should be 3D: [batch_size X num_features X num_steps]")
        
        batch_size, num_features, num_steps = X.shape
        X = X.contiguous().view(batch_size, num_features*num_steps)
        max_value = torch.max(X, dim=1)[0].detach()
        min_value = torch.min(X, dim=1)[0].detach()
        max_value = torch.unsqueeze(max_value,1)
        min_value = torch.unsqueeze(min_value,1)
        X = (X - min_value) / (max_value - min_value + 1E-10)
        return X.view(batch_size, num_features, num_steps)


class TrainableSpectrogram(nn.Module):
    """
    From 1D raw audio to 2D representation
    """
    def __init__(self, config):
        super(TrainableSpectrogram, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=320, stride=160, padding=80, bias=False)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=self.config.getint('AUDIO', 'n_mels'), kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x:torch.Tensor):
        if len(x.shape) != 2:
            raise ValueError("Audio should be 2D: [batch_size X audio_length]")
        if x.shape[1] < 1:
            raise ValueError("Audio length is zero")
        
        x.unsqueeze_(1) # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class Model(nn.Module):
    """
    Feature extractor model
    """
    def __init__(self):
        super(Model, self).__init__()
        self.n_mels = config.getint('AUDIO', 'n_mels')
        self.sr = config.getint('AUDIO', 'sr')

        preprocess_steps = list()
        preprocess_steps.append(Normalize())
        if config.getboolean("VECTORIZER", "use_trainable_mel"):
            preprocess_steps.append(TrainableSpectrogram(config))
        else:
            preprocess_steps.append(torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=400, win_length=400, hop_length=160, n_mels=self.n_mels))
        preprocess_steps.append(Normalize3D())
        
        self.eval_preprocess_steps = nn.Sequential(*tuple(preprocess_steps))

        preprocess_steps.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=10))
        preprocess_steps.append(torchaudio.transforms.TimeMasking(time_mask_param=15))

        self.train_preprocess_steps = nn.Sequential(*tuple(preprocess_steps))

        model = torchvision.models.mobilenet_v3_small(pretrained=False)
        self.net = torch.nn.Sequential(*list(model.children())[:-1])
        # Use single channel input
        self.net[0][0] = nn.Conv2d(1, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.gelu = torch.nn.GELU()
        self.emb_fc = nn.Linear(576, config.getint("VECTORIZER", "embeddings_size"))

    def forward(self, x:torch.Tensor, train:bool=False) -> torch.Tensor:
        batch_size = x.shape[0]
        if train:
            x = self.train_preprocess_steps(x)
        else:
            x = self.eval_preprocess_steps(x)
        x = x.unsqueeze(1) # Add channel dimension
        features = self.net(x).view(batch_size, -1)        
        features = self.gelu(features)
        features = self.emb_fc(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features

if __name__ == "__main__":
    model = Model()
    trainable_spec = TrainableSpectrogram(config)
    print(f'Number of model parameters: {sum([p.data.nelement() for name, p in model.named_parameters() ]):,}')
    print(f'Number of TrainableSpectrogram parameters: {sum([p.data.nelement() for _, p in trainable_spec.named_parameters() ]):,}')
    audio = torch.randn(config.getint("VECTORIZER", "batch_size"), 16000)
    features = model(audio, train=True)
    print(features.shape)

    normalize3D = Normalize3D()
    spec = torch.arange(10, 20).view(1, 2, 5)
    print(normalize3D(spec))

    normalize = Normalize()
    audio = torch.arange(-2, 3).view(1, 5)
    print(normalize(audio))