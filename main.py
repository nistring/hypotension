import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader
import lightning as L

from model import Model
import yaml
from dataset import VitalDataset

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from easydict import EasyDict as edict
from utils import vis

class LitDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage):
        self.train = VitalDataset(self.cfg, (0.0, 0.7))
        self.val = VitalDataset(self.cfg, (0.7, 0.8))
        self.test = VitalDataset(self.cfg, (0.8, 1.0))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.BATCH_SIZE, shuffle=False, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.BATCH_SIZE, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.cfg.BATCH_SIZE, shuffle=False, num_workers=1)

# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = Model(cfg)
        self.lr = cfg.LEARNING_RATE
        self.loss_fn = nn.BCELoss(reduction="sum")
        self.threshold = 0.5

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        
    def on_test_start(self):
        self.true = np.array([])
        self.pred = np.array([])
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).cpu().numpy()
        self.true = np.append(self.true, y.cpu().numpy())
        self.pred = np.append(self.pred, y_hat)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def train():
    with open("config.yaml", "r") as f:
        cfg = edict(yaml.safe_load(f))

    model = LitModel(cfg)
    datamodule = LitDataModule(cfg)
    checkpoint_cb = ModelCheckpoint(monitor="val_loss")
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=2, verbose=False, mode="min")    

    trainer = L.Trainer(max_epochs=cfg.MAX_EPOCHS, callbacks=[checkpoint_cb, early_stop_cb], log_every_n_steps=10)
    trainer.fit(model=model, datamodule=datamodule)

def test():
    checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=21-step=220.ckpt"
    with open("config.yaml", "r") as f:
        cfg = edict(yaml.safe_load(f))

    model = LitModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
    datamodule = LitDataModule(cfg)
    trainer = L.Trainer()
    trainer.test(model=model, datamodule=datamodule)
    vis(model.true, model.pred)

if __name__ == "__main__":
    # train()
    test()