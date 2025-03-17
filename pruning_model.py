# import os
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import matplotlib.pyplot as plt
# from torchsummary import summary
# import torch_pruning as tp
# from mobilenetv3 import mobilenetv3
# from collections import defaultdict
# import time
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import pytorch_lightning as L

# data_folder = '/data/personal/qoherent2/new_recording/New_Recording/r22_august8_beam_dataset_qualified_original'
# checkpoint_filename = 'pruning'
# original_model_checkpoint = 'august8_beam.ckpt'

# example_length = 4096

# def read_recording(file):
#     with open(file=file, mode='rb') as f:
#         data = np.load(f, allow_pickle=True)
#         metadata = np.load(f, allow_pickle=True)
#         metadata = metadata.tolist()
#     return data, metadata


# def reshape_complex_array(array, example_length):
#     num_examples = array.shape[1] // example_length
#     array = array[:, array.shape[1] - (num_examples * example_length):]
#     chan1 = array[0].real.reshape(-1, example_length)
#     chan2 = array[1].real.reshape(-1, example_length)
#     chan3 = array[2].real.reshape(-1, example_length)
#     chan4 = array[3].real.reshape(-1, example_length)
#     chan5 = array[0].imag.reshape(-1, example_length)
#     chan6 = array[1].imag.reshape(-1, example_length)
#     chan7 = array[2].imag.reshape(-1, example_length)
#     chan8 = array[3].imag.reshape(-1, example_length)
#     reshaped_array = np.stack([chan1, chan2, chan3, chan4, chan5, chan6, chan7, chan8], axis=1)
#     return reshaped_array


# def build_dataset(df, example_length):
#     iq_data = []
#     labels = []
#     for i, row in df.iterrows():
#         data, _ = read_recording(row['recording'])
#         data = reshape_complex_array(data, example_length)
#         iq_data.append(data)
#         label = np.float32(np.array([row['servo_azimuth'], row['servo_elevation']]))
#         label = np.tile(label, (data.shape[0], 1))
#         labels.append(label)
#     iq_data = np.vstack(iq_data)
#     labels = np.vstack(labels)
#     return iq_data, labels


# def test_and_plot(loader, model, title_suffix, file_suffix):
#     device = next(model.parameters()).device  
#     with torch.no_grad():
#         predictions_dict = defaultdict(list)
#         for batch in loader:
#             inputs, true_angles = batch
#             inputs, true_angles = inputs.to(device), true_angles.to(device)
#             predicted = model(inputs).cpu().numpy()
#             true_angles = true_angles.cpu().numpy()
#             for i in range(len(true_angles)):
#                 key = tuple(true_angles[i])
#                 predictions_dict[key].append(predicted[i])

      
#         mean_predictions = {}
#         average_base_angles = []

#         for key, preds in predictions_dict.items():
#             mean_pred = np.mean(preds, axis=0)
#             mean_predictions[key] = mean_pred
#             average_base_angles.append(mean_pred)

      
#         true_azimuths = []
#         true_elevations = []
#         pred_azimuth_diffs = []
#         pred_elevation_diffs = []

#         for (true_azimuth, true_elevation), mean_pred in mean_predictions.items():
#             true_azimuths.append(true_azimuth)
#             true_elevations.append(true_elevation)
#             pred_azimuth_diffs.append(mean_pred[0] - true_azimuth)
#             pred_elevation_diffs.append(mean_pred[1] - true_elevation)

     
    
#         true_azimuths = np.array(true_azimuths)
#         true_elevations = np.array(true_elevations)
#         pred_azimuth_diffs = np.array(pred_azimuth_diffs)
#         pred_elevation_diffs = np.array(pred_elevation_diffs)

        

#     return true_azimuths, true_elevations, pred_azimuth_diffs, pred_elevation_diffs


# def save_model_size(model_path):
#     size_mb = os.path.getsize(model_path) / (1024 * 1024) 
#     return size_mb



# def save_pruning_groups(pruning_method, groups, prune_amount):
#     file_path = f'pruning_groups_{pruning_method}_amount_{prune_amount}.txt'
#     with open(file_path, 'w') as f:
#         f.write(f"Pruning Method: {pruning_method}\n")
#         f.write(f"Pruning Amount: {prune_amount}\n\n")
#         for i, group in enumerate(groups):
#             f.write(f"Group {i + 1}:\n")
#             f.write(group.details())
#             f.write("\n" + "-" * 50 + "\n")


# class RFClassifier(L.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.lr = 3e-3

#     def forward(self, x):
#         return self.model(x)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0)
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
#         return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.mse_loss(y_hat, y)
#         self.log('train_loss', loss, on_epoch=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.mse_loss(y_hat, y)
#         self.log('val_loss', loss, prog_bar=True)



# def prune_model(model, prune_method, prune_amount):
#     model.eval()  

   
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)

   
#     DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 8, example_length).to(device))

#     groups = []

#     if prune_method == 'channel_pruning' or prune_method == 'filter_pruning':
#         for layer in model.modules():
#             if isinstance(layer, torch.nn.Conv1d):
#                 print(f"Pruning {prune_method} on layer: {layer}")
#                 group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=[0, 1, 2])
#                 groups.append(group)

                
#                 if DG.check_pruning_group(group):
#                     group.prune()

#     elif prune_method == 'layer_pruning':
#         for name, layer in model.named_children():
#             if isinstance(layer, torch.nn.Conv1d):
#                 print(f"Pruning {prune_method} on layer: {layer}")
#                 group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=[0, 1, 2])
#                 groups.append(group)

              
            
#                 if DG.check_pruning_group(group):
#                     group.prune()

  
#     save_pruning_groups(prune_method, groups, prune_amount)

   
#     input_size = (8, example_length)
#     try:
#         summary(model, input_size)
#         print(f"Model structure matches the input size {input_size}.")
#     except RuntimeError as e:
#         print(f"Model size mismatch encountered after pruning: {str(e)}")
#         print("Adjusting model...")
#         summary(model, input_size)
#         print("Model is now adjusted and compatible with the input size.")

#     return model


# def train_model_with_pruning(pruning_method, model_name_suffix, train_loader, val_loader, prune_amount):
#     checkpoint_filename = f"model_pruned_{model_name_suffix}_amount_{prune_amount}.ckpt"
#     final_model_path = f'model_final_pruned_{model_name_suffix}_amount_{prune_amount}.pth'

   
#     if os.path.exists(checkpoint_filename) or os.path.exists(final_model_path):
#         print(f"Model {model_name_suffix} at {prune_amount * 100}% pruning already exists. Skipping training.")
#         return


#     model = mobilenetv3(
#         model_size='mobilenetv3_small_050',
#         num_classes=2,
#         drop_rate=0,
#         drop_path_rate=0,
#         in_chans=8
#     )

#     model = prune_model(model, pruning_method, prune_amount)

#     rf_classifier = RFClassifier(model)

#     checkpoint_callback = L.callbacks.ModelCheckpoint(
#         dirpath='.',
#         filename=checkpoint_filename.replace(".ckpt", ""),
#         save_top_k=1,
#         verbose=True,
#         monitor='val_loss',
#         mode='min'
#     )

#     trainer = L.Trainer(
#         max_epochs=1,
#         callbacks=[checkpoint_callback],
#         accelerator='gpu',
#         devices=1,
#         benchmark=True,
#         precision='bf16-mixed',
#     )

#     print(f"Training the model with {pruning_method} applied at {prune_amount * 100}%...")
#     trainer.fit(rf_classifier, train_loader, val_loader)

#     torch.save(model.state_dict(), final_model_path)


#     model_size_mb = save_model_size(final_model_path)


#     rf_classifier.eval()
#     start_time = time.time()
#     test_and_plot(
#         val_loader, model, f'Pruned {pruning_method} ({prune_amount * 100}% Pruning)',
#         f'pruned_{pruning_method}_amount_{prune_amount}'
#     )
#     inference_time = time.time() - start_time

#     analysis_results['model_name'].append(f"model_pruned_{pruning_method}_amount_{prune_amount}")
#     analysis_results['model_size_mb'].append(model_size_mb)
#     analysis_results['validation_loss'].append(trainer.callback_metrics['val_loss'].item())
#     analysis_results['inference_time'].append(inference_time)
#     analysis_results['pruning_percentage'].append(prune_amount * 100)
#     analysis_results['pruning_method'].append(pruning_method)

#     print(
#         f"Model {pruning_method} at {prune_amount * 100}% pruning saved. "
#         f"Size: {model_size_mb:.2f} MB, Validation Loss: {trainer.callback_metrics['val_loss'].item()}, "
#         f"Inference Time: {inference_time:.2f} seconds"
#     )



# print("Creating the baseline validation DataFrame with all modulation schemes...")
# recordings_list = []
# for folder in os.listdir(data_folder):
#     if os.path.isdir(os.path.join(data_folder, folder)):
#         for recording in os.listdir(os.path.join(data_folder, folder)):
#             _, extension = os.path.splitext(recording)
#             if extension == '.npy':
#                 _, metadata = read_recording(os.path.join(data_folder, folder, recording))
#                 data_row = {
#                     'recording': os.path.join(data_folder, folder, recording),
#                     'category': folder,
#                     'servo_azimuth': metadata['servo_azimuth'],
#                     'servo_elevation': metadata['servo_elevation'],
#                 }
#                 recordings_list.append(data_row)

# df = pd.DataFrame(recordings_list)
# df.to_pickle('baseline_val.pkl')
# print("Baseline validation DataFrame saved.")


# baseline_df = pd.read_pickle('baseline_val.pkl')
# baseline_data, baseline_labels = build_dataset(baseline_df, example_length)
# baseline_set = TensorDataset(torch.as_tensor(baseline_data), torch.as_tensor(baseline_labels))
# baseline_loader = DataLoader(dataset=baseline_set, batch_size=128, shuffle=False, num_workers=8)


# train_df, val_df = train_test_split(baseline_df, stratify=baseline_df['category'], test_size=0.2, random_state=2016)
# train_data, train_labels = build_dataset(train_df, example_length)
# val_data, val_labels = build_dataset(val_df, example_length)


# train_loader = DataLoader(
#     dataset=TensorDataset(torch.as_tensor(train_data), torch.as_tensor(train_labels)),
#     batch_size=128, shuffle=True, num_workers=8
# )
# val_loader = DataLoader(
#     dataset=TensorDataset(torch.as_tensor(val_data), torch.as_tensor(val_labels)),
#     batch_size=128, shuffle=False, num_workers=8
# )


# analysis_results = {
#     'model_name': [],
#     'model_size_mb': [],
#     'validation_loss': [],
#     'training_time': [],
#     'inference_time': [],
#     'pruning_percentage': [],
#     'pruning_method': [],
# }


# print("Testing the original unpruned model...")
# model = mobilenetv3(
#     model_size='mobilenetv3_small_050',
#     num_classes=2,
#     drop_rate=0,
#     drop_path_rate=0,
#     in_chans=8
# )
# rf_classifier = RFClassifier.load_from_checkpoint(original_model_checkpoint, model=model)
# rf_classifier.eval()


# rf_classifier.to('cuda' if torch.cuda.is_available() else 'cpu')

# start_time = time.time()
# test_and_plot(baseline_loader, rf_classifier.model, 'Original Model (Unpruned)', 'original_unpruned')
# inference_time = time.time() - start_time


# analysis_results['model_name'].append('original_unpruned')
# analysis_results['model_size_mb'].append(save_model_size(original_model_checkpoint))
# analysis_results['validation_loss'].append(None)  
# analysis_results['training_time'].append(None) 
# analysis_results['inference_time'].append(inference_time)
# analysis_results['pruning_percentage'].append(0)
# analysis_results['pruning_method'].append('None')

# pruning_methods = ['channel_pruning', 'filter_pruning', 'layer_pruning']
# pruning_amounts = [0.05]  

# for method in pruning_methods:
#     for amount in pruning_amounts:
#         train_model_with_pruning(method, method, train_loader, val_loader, amount)

# max_len = max(len(lst) for lst in analysis_results.values())

# for key in analysis_results:
#     if len(analysis_results[key]) < max_len:
     
#         analysis_results[key].extend([None] * (max_len - len(analysis_results[key])))


# results_df = pd.DataFrame(analysis_results)
# print(results_df)

# torch.save(model.state_dict(), 'original_model.pth')

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torch_pruning as tp
from mobilenetv3 import mobilenetv3
from collections import defaultdict
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as L
import h5py
from torch.utils.data import Dataset

data_folder = "/home/mohammad.hallaq/workarea/MobileNet_compression/r22_august22_beam_dataset/beam_recordings" 
checkpoint_filename = 'pruning'
original_model_checkpoint = 'august22_beam.ckpt'

example_length = 4096
def read_recording(file):
    with open(file=file, mode='rb') as f:
        data = np.load(f, allow_pickle=True)
        metadata = np.load(f, allow_pickle=True)
        metadata = metadata.tolist()
    return data, metadata


def reshape_complex_array(array, example_length):
    num_examples = array.shape[1] // example_length
    array = array[:, array.shape[1] - (num_examples * example_length):]
    chan1 = array[0].real.reshape(-1, example_length)
    chan2 = array[1].real.reshape(-1, example_length)
    chan3 = array[2].real.reshape(-1, example_length)
    chan4 = array[3].real.reshape(-1, example_length)
    chan5 = array[0].imag.reshape(-1, example_length)
    chan6 = array[1].imag.reshape(-1, example_length)
    chan7 = array[2].imag.reshape(-1, example_length)
    chan8 = array[3].imag.reshape(-1, example_length)
    reshaped_array = np.stack([chan1, chan2, chan3, chan4, chan5, chan6, chan7, chan8], axis=1)
    return reshaped_array


def build_dataset(df, example_length):
    iq_data = []
    labels = []
    for i, row in df.iterrows():
        data, _ = read_recording(row['recording'])
        data = reshape_complex_array(data, example_length)
        iq_data.append(data)
        label = np.float32(np.array([row['servo_azimuth'], row['servo_elevation']]))
        label = np.tile(label, (data.shape[0], 1))
        labels.append(label)
    iq_data = np.vstack(iq_data)
    labels = np.vstack(labels)
    return iq_data, labels


def test_and_plot(loader, model, title_suffix, file_suffix):
    device = next(model.parameters()).device  
    with torch.no_grad():
        predictions_dict = defaultdict(list)
        for batch in loader:
            inputs, true_angles = batch
            inputs, true_angles = inputs.to(device), true_angles.to(device)
            predicted = model(inputs).cpu().numpy()
            true_angles = true_angles.cpu().numpy()
            for i in range(len(true_angles)):
                key = tuple(true_angles[i])
                predictions_dict[key].append(predicted[i])

      
        mean_predictions = {}
        average_base_angles = []

        for key, preds in predictions_dict.items():
            mean_pred = np.mean(preds, axis=0)
            mean_predictions[key] = mean_pred
            average_base_angles.append(mean_pred)

      
        true_azimuths = []
        true_elevations = []
        pred_azimuth_diffs = []
        pred_elevation_diffs = []

        for (true_azimuth, true_elevation), mean_pred in mean_predictions.items():
            true_azimuths.append(true_azimuth)
            true_elevations.append(true_elevation)
            pred_azimuth_diffs.append(mean_pred[0] - true_azimuth)
            pred_elevation_diffs.append(mean_pred[1] - true_elevation)

     
    
        true_azimuths = np.array(true_azimuths)
        true_elevations = np.array(true_elevations)
        pred_azimuth_diffs = np.array(pred_azimuth_diffs)
        pred_elevation_diffs = np.array(pred_elevation_diffs)

        

    return true_azimuths, true_elevations, pred_azimuth_diffs, pred_elevation_diffs


def save_model_size(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024) 
    return size_mb

def save_pruning_groups(pruning_method, groups, prune_amount):
    file_path = f'pruning_groups_{pruning_method}_amount_{prune_amount}.txt'
    with open(file_path, 'w') as f:
        f.write(f"Pruning Method: {pruning_method}\n")
        f.write(f"Pruning Amount: {prune_amount}\n\n")
        for i, group in enumerate(groups):
            f.write(f"Group {i + 1}:\n")
            f.write(group.details())
            f.write("\n" + "-" * 50 + "\n")

# class RFClassifier(L.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.lr = 3e-3

#     def forward(self, x):
#         return self.model(x)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0)
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
#         return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.mse_loss(y_hat, y)
#         self.log('train_loss', loss, on_epoch=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.mse_loss(y_hat, y)
#         self.log('val_loss', loss, prog_bar=True)


class RFClassifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 3e-3
        self.lr_ignored = 1e-4  # Learning rate for ignored layers

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
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)


class R22_H5_Dataset(Dataset):
    def __init__(self, data_file, label='label', iqlabel='iq_data'):
        self.data_file = data_file
        self.label = label
        self.iqlabel = iqlabel

    def __len__(self):
        with h5py.File(self.data_file, 'r') as f:
            length = len(f[self.label])
        return length

    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as f:
            iq_data = f[self.iqlabel][idx]
            label = f[self.label][idx]
        return iq_data, label
    
    def get_metadata(self, idx):
        with h5py.File(self.data_file, 'r') as f:
            metadata = {
                'recording': f['recording'][idx].decode('utf-8)'),
                'category': f['category'][idx].decode('utf-8)')
            }
        return metadata


def prune_model(model, prune_method, prune_amount, train_loader, val_loader):
    # model.eval()  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    example_inputs=torch.randn(1, 8, example_length).to(device)
    # DG = tp.DependencyGraph().build_dependency(model, example_inputs)

    groups = []

    if prune_method == 'channel_pruning_Taylor_importance':
        imp = tp.importance.TaylorImportance() #GroupNormImportance(p=2)

        ignored_layers = []
        # for m in model.modules():
        #     if isinstance(m, torch.nn.Linear) and m.out_features == 2:
        #         ignored_layers.append(m) # DO NOT prune the final classifier!

        for name, m in model.named_modules():
            # Check if the module is within Sequential(4) or Sequential(5)
            if any(name.startswith(f'blocks.{i}') for i in range(3)) or (isinstance(m, torch.nn.Linear) and m.out_features == 2):
                ignored_layers.append(m)  # DO NOT prune the final classifier

    
        model.to(device)
        # example_inputs.to(device)

        # rf_classifier = RFClassifier(model)#.to(device)
        batch = next(iter(train_loader))
        # x, y = batch
        # x = x.to(device)
       
        iterative_steps= 1

        pruner = tp.pruner.MagnitudePruner( 
            model,
            example_inputs=example_inputs,
            importance=imp,
            pruning_ratio=prune_amount, 
            # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
            ignored_layers=ignored_layers,
            iterative_steps= iterative_steps,
            # global_pruning=True,
            # isomorphic=True
        )

        # prune the model, iteratively if necessary.
        for i in range(iterative_steps):

            # Taylor expansion requires gradients for importance estimation
            if isinstance(imp, tp.importance.TaylorImportance):
                
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = F.mse_loss(y_hat, y)
                loss.backward() # before pruner.step()

            pruner.step()


  
    save_pruning_groups(prune_method, groups, prune_amount)

   
    input_size = (8, example_length)
    try:
        summary(model, input_size)
        print(f"Model structure matches the input size {input_size}.")
    except RuntimeError as e:
        print(f"Model size mismatch encountered after pruning: {str(e)}")
        print("Adjusting model...")
        summary(model, input_size)
        print("Model is now adjusted and compatible with the input size.")

    return model

class CustomCheckpoint(L.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_model = None

    def on_validation_end(self, trainer, pl_module):
        # Access validation loss from the trainer's metrics
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = pl_module.model
            # Save the best model
            torch.save(self.best_model, 'best_model.pth')
            print(f"New best model saved with validation loss {val_loss:.4f}")

def train_model_with_pruning(trained_model, pruning_method, model_name_suffix, train_loader, val_loader, prune_amount):
    checkpoint_filename = f"model_pruned_{model_name_suffix}_amount_{prune_amount}.ckpt"
    final_model_path = f'model_final_pruned_{model_name_suffix}_amount_{prune_amount}.pth'

   
    if os.path.exists(checkpoint_filename) or os.path.exists(final_model_path):
        print(f"Model {model_name_suffix} at {prune_amount * 100}% pruning already exists. Skipping training.")
        return


    # model = mobilenetv3(
    #     model_size='mobilenetv3_small_050',
    #     num_classes=2,
    #     drop_rate=0,
    #     drop_path_rate=0,
    #     in_chans=8
    # )

    pruned_model = prune_model(trained_model, pruning_method, prune_amount, train_loader, val_loader)

    rf_classifier = RFClassifier(pruned_model)
    # rf_classifier.lr = 1e-4

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        dirpath='.',
        filename=checkpoint_filename.replace(".ckpt", ""),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # Create the custom callback
    custom_checkpoint = CustomCheckpoint()

    trainer = L.Trainer(
        max_epochs=30,
        callbacks=[checkpoint_callback, custom_checkpoint],
        accelerator='gpu',
        devices=1,
        benchmark=True,
        precision='32-true',
    )

    print(f"Training the model with {pruning_method} applied at {prune_amount * 100}%...")
    trainer.fit(rf_classifier, train_loader, val_loader)

    # trainer = L.Trainer(
    # accelerator='gpu',
    # callbacks=[checkpoint_callback],
    # devices=1,
    # benchmark=True,
    # precision='bf16-mixed',
    # )

    # print(f"Validating the model with {pruning_method} applied at {prune_amount * 100}%...")
    # trainer.validate(rf_classifier, val_loader)

    torch.save(rf_classifier.model, final_model_path)


    model_size_mb = save_model_size(final_model_path)


    rf_classifier.eval()
    start_time = time.time()
    test_and_plot(
        val_loader, rf_classifier.model, f'Pruned {pruning_method} ({prune_amount * 100}% Pruning)',
        f'pruned_{pruning_method}_amount_{prune_amount}'
    )
    inference_time = time.time() - start_time

    analysis_results['model_name'].append(f"model_pruned_{pruning_method}_amount_{prune_amount}")
    analysis_results['model_size_mb'].append(model_size_mb)
    analysis_results['validation_loss'].append(trainer.callback_metrics['val_loss'].item())
    analysis_results['inference_time'].append(inference_time)
    analysis_results['pruning_percentage'].append(prune_amount * 100)
    analysis_results['pruning_method'].append(pruning_method)

    print(
        f"Model {pruning_method} at {prune_amount * 100}% pruning saved. "
        f"Size: {model_size_mb:.2f} MB, Validation Loss: {trainer.callback_metrics['val_loss'].item()}, "
        f"Inference Time: {inference_time:.2f} seconds"
    )

print("Creating the baseline validation DataFrame with all modulation schemes...")
recordings_list = []
for folder in os.listdir(data_folder):
    if os.path.isdir(os.path.join(data_folder, folder)):
        for recording in os.listdir(os.path.join(data_folder, folder)):
            _, extension = os.path.splitext(recording)
            if extension == '.npy':
                _, metadata = read_recording(os.path.join(data_folder, folder, recording))
                data_row = {
                    'recording': os.path.join(data_folder, folder, recording),
                    'category': folder,
                    'servo_azimuth': metadata['servo_azimuth'],
                    'servo_elevation': metadata['servo_elevation'],
                }
                recordings_list.append(data_row)

df = pd.DataFrame(recordings_list)
df.to_pickle('baseline_val.pkl')
print("Baseline validation DataFrame saved.")


baseline_df = pd.read_pickle('baseline_val.pkl')
baseline_data, baseline_labels = build_dataset(baseline_df, example_length)
baseline_set = TensorDataset(torch.as_tensor(baseline_data), torch.as_tensor(baseline_labels))
baseline_loader = DataLoader(dataset=baseline_set, batch_size=128, shuffle=False, num_workers=8)


train_df, val_df = train_test_split(baseline_df, stratify=baseline_df['category'], test_size=0.2, random_state=2016)
train_data, train_labels = build_dataset(train_df, example_length)
val_data, val_labels = build_dataset(val_df, example_length)


train_loader = DataLoader(
    dataset=TensorDataset(torch.as_tensor(train_data), torch.as_tensor(train_labels)),
    batch_size=128, shuffle=True, num_workers=8
)
val_loader = DataLoader(
    dataset=TensorDataset(torch.as_tensor(val_data), torch.as_tensor(val_labels)),
    batch_size=128, shuffle=False, num_workers=8
)


analysis_results = {
    'model_name': [],
    'model_size_mb': [],
    'validation_loss': [],
    'training_time': [],
    'inference_time': [],
    'pruning_percentage': [],
    'pruning_method': [],
}


print("Testing the original unpruned model...")
model = mobilenetv3(
    model_size='mobilenetv3_small_050',
    num_classes=2,
    drop_rate=0,
    drop_path_rate=0,
    in_chans=8
)
rf_classifier = RFClassifier.load_from_checkpoint(original_model_checkpoint, model=model)
rf_classifier.eval()


rf_classifier.to('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
test_and_plot(baseline_loader, rf_classifier.model, 'Original Model (Unpruned)', 'original_unpruned')
inference_time = time.time() - start_time


analysis_results['model_name'].append('original_unpruned')
analysis_results['model_size_mb'].append(save_model_size(original_model_checkpoint))
analysis_results['validation_loss'].append(None)  
analysis_results['training_time'].append(None) 
analysis_results['inference_time'].append(inference_time)
analysis_results['pruning_percentage'].append(0)
analysis_results['pruning_method'].append('None')

pruning_methods = ['channel_pruning_Taylor_importance'] #, 'filter_pruning', 'layer_pruning']
pruning_amounts = [0.0]  

for method in pruning_methods:
    for amount in pruning_amounts:
        # print("Skipped a pruning method ...")
        train_model_with_pruning(rf_classifier.model, method, method, train_loader, val_loader, amount)

max_len = max(len(lst) for lst in analysis_results.values())

for key in analysis_results:
    if len(analysis_results[key]) < max_len:
     
        analysis_results[key].extend([None] * (max_len - len(analysis_results[key])))


results_df = pd.DataFrame(analysis_results)
print("This is the analysis result: \n", results_df)

torch.save(model.state_dict(), 'original_model.pth')