import pytorch_lightning as L
import torch
import torch.nn.functional as F

class RFClassifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 1e-3
        self.lr_ignored = 1e-3 # Learning rate for ignored layers

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Initialize lists for different parameter groups
        ignored_layers_params = []
        other_layers_params = []

        # Collect parameters of specific layers based on layer names and conditions
        ignored_params_set = set()
        for name, m in self.model.named_modules():
            if any(name.startswith(f'blocks.{i}') for i in range(3)) or (isinstance(m, torch.nn.Linear) and m.out_features == 2):
                ignored_layers_params += list(m.parameters())
                ignored_params_set.update(m.parameters())  # Add to set to avoid duplicates

        # Other parameters: Exclude the ignored layers' parameters
        other_layers_params = [p for p in self.model.parameters() if p not in ignored_params_set]

        # Create the optimizer with different learning rates for different parameter groups
        optimizer = torch.optim.AdamW([
            {'params': ignored_layers_params, 'lr': self.lr_ignored},  # Lower learning rate for ignored layers
            {'params': other_layers_params, 'lr': self.lr}  # Default learning rate for other layers
        ], weight_decay=0)

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)