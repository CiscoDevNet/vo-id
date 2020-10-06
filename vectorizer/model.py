import torch
import torch.nn as nn
import torchaudio
from efficientnet_pytorch import EfficientNet

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
        preprocess_steps.append(torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=400, win_length=400, hop_length=160, n_mels=self.n_mels))
        preprocess_steps.append(Normalize3D())
        
        self.eval_preprocess_steps = nn.Sequential(*tuple(preprocess_steps))

        preprocess_steps.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=10))
        preprocess_steps.append(torchaudio.transforms.TimeMasking(time_mask_param=15))

        self.train_preprocess_steps = nn.Sequential(*tuple(preprocess_steps))

        self.net = EfficientNet.from_name("efficientnet-b0", include_top=False)
        self.net._change_in_channels(in_channels=1)
        self.gelu = torch.nn.GELU()
        self.emb_fc = nn.Linear(1280, config.getint("VECTORIZER", "embeddings_size"))

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
        
        return features

if __name__ == "__main__":
    model = Model()
    audio = torch.randn(config.getint("VECTORIZER", "batch_size"), 32000)
    features = model(audio, train=True)
    print(features.shape)

    normalize3D = Normalize3D()
    spec = torch.arange(10, 20).view(1, 2, 5)
    print(normalize3D(spec))

    normalize = Normalize()
    audio = torch.arange(-2, 3).view(1, 5)
    print(normalize(audio))
    