import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_size, hparams["n_hidden_1"]),
            nn.BatchNorm1d(hparams["n_hidden_1"]),
            nn.LeakyReLU(0.1),

            nn.Linear(hparams["n_hidden_1"], hparams["n_hidden_2"]),
            nn.BatchNorm1d(hparams["n_hidden_2"]),
            nn.LeakyReLU(0.1),

            nn.Linear(hparams["n_hidden_2"], latent_dim)
        )

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams

        self.decoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(latent_dim, hparams["n_hidden_2"]),
            nn.BatchNorm1d(hparams["n_hidden_2"]),
            nn.LeakyReLU(0.1),

            nn.Linear(hparams["n_hidden_2"], hparams["n_hidden_1"]),
            nn.BatchNorm1d(hparams["n_hidden_1"]),
            nn.LeakyReLU(0.1),

            nn.Linear(hparams["n_hidden_1"], output_size)
        )

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        
        latent = self.encoder.forward(x)
        reconstruction = self.decoder.forward(latent)
        
        return reconstruction

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None

        self.train()
        self.optimizer.zero_grad()
        images = batch
        images = images.to(self.device)
        images = images.view(images.shape[0], -1)
        pred = self.forward(images)
        loss = loss_func(pred, images)
        loss.backward()
        self.optimizer.step() 
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        self.eval()
        with torch.no_grad():
            images = batch
            images = images.to(self.device)
            images = images.view(images.shape[0], -1) 
            pred = self.forward(images)
            loss = loss_func(pred, images)

        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = nn.Sequential(
            nn.Linear(self.hparams["latent_dim"], self.hparams["n_hidden_2"]),
            nn.BatchNorm1d(self.hparams["n_hidden_2"]),
            nn.LeakyReLU(0.1),

            nn.Linear(self.hparams["n_hidden_2"], self.hparams["num_classes"])
        )

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = torch.optim.Adam(self.parameters(), self.hparams["lr"])


    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
