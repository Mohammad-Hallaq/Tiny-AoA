import argparse
import os
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 25
random.seed(SEED)

# Example usage:
# python3 create_dataset.py -n august_22_beam -d <path to dataset folder> -s <path to save dataset files> -l 4096 -p 15

def read_recording(file):
    with open(file=file, mode='rb') as f:
        data = np.load(f, allow_pickle=True)
        metadata = np.load(f, allow_pickle=True)
        metadata = metadata.tolist()
    return data, metadata

def reshape_complex_array(array, example_length):

    num_examples = array.shape[1] // example_length
    array = array[:,array.shape[1]-(num_examples * example_length):]

    chan1 = array[0].real.reshape(-1,example_length)
    chan2 = array[1].real.reshape(-1,example_length)
    chan3 = array[2].real.reshape(-1,example_length)
    chan4 = array[3].real.reshape(-1,example_length)

    chan5 = array[0].imag.reshape(-1,example_length)
    chan6 = array[1].imag.reshape(-1,example_length)
    chan7 = array[2].imag.reshape(-1,example_length)
    chan8 = array[3].imag.reshape(-1,example_length)

    reshaped_array = np.float32(np.stack([chan1,chan2,chan3,chan4,chan5,chan6,chan7,chan8],axis=1))
    
    return reshaped_array

def write_dataset(args, df, suffix):

    file_started = False
    f = h5py.File(f'{args.save_location}/{args.name}{suffix}.h5', 'a')

    max_length = 0
    for i, row in df.iterrows():

        max_length += row['length'] // args.example_length

    for i, row in df.iterrows():
        
        iq_data, _ = read_recording(row['recording'])
        split_name = row['recording'].split('/')

        threshold = 10
        if np.sum(np.abs(iq_data) > threshold):
            iq_data[np.abs(iq_data) > 10] = 0
        iq_data = iq_data / np.max(np.abs(iq_data))

        iq_data = reshape_complex_array(iq_data, args.example_length)
        label = np.float32(np.array([row['servo_azimuth'], row['servo_elevation']]))
        label = np.tile(label, (iq_data.shape[0],1))

        recording_tiled = np.array([split_name[-1]]*iq_data.shape[0], dtype=h5py.string_dtype())
        category_tiled = np.array([row['category']]*iq_data.shape[0], dtype=h5py.string_dtype())
        
        if file_started:
            f['iq_data'].resize((f['iq_data'].shape[0] + iq_data.shape[0]), axis=0)
            f['iq_data'][-iq_data.shape[0]:] = iq_data
            f['label'].resize((f['label'].shape[0] + label.shape[0]), axis=0)
            f['label'][-label.shape[0]:] = label
            f['recording'].resize((f['recording'].shape[0] + recording_tiled.shape[0]), axis=0)
            f['recording'][-recording_tiled.shape[0]:] = recording_tiled
            f['category'].resize((f['category'].shape[0] + category_tiled.shape[0]), axis=0)
            f['category'][-category_tiled.shape[0]:] = category_tiled
        else:
            file_started=True
            f.create_dataset('iq_data', data=iq_data, maxshape=(max_length,8,args.example_length))
            f.create_dataset('label', data=label, maxshape=(max_length,2))
            f.create_dataset('recording', data=recording_tiled, maxshape=(max_length))
            f.create_dataset('category', data=category_tiled, maxshape=(max_length))

def create_dataset(args):

    data_folder = args.data_folder

    recordings_list = []
    for folder in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, folder)):
            all_recordings = [rec for rec in os.listdir(os.path.join(data_folder, folder)) if rec.endswith('.npy')]
            # Select a random subset based on the percentage specified
            num_samples = max(1, int(len(all_recordings) * args.percentage / 100))
            selected_recordings = random.sample(all_recordings, num_samples)
            
            for recording in selected_recordings:
                iq_data, metadata = read_recording(os.path.join(data_folder, folder, recording))
                if np.sum(np.isinf(iq_data)) > 0:
                    print(f'{folder}/{recording} ({metadata["servo_azimuth"]}, {metadata["servo_elevation"]}) skipped due to Inf values')
                    continue
                if np.sum(np.isnan(iq_data)) > 0:
                    print(f'{folder}/{recording} ({metadata["servo_azimuth"]}, {metadata["servo_elevation"]}) skipped due to NaN values')
                    continue

                threshold = 10
                if np.sum(np.abs(iq_data) > threshold):
                    print(f'{folder}/{recording} ({metadata["servo_azimuth"]}, {metadata["servo_elevation"]}) has {np.sum(np.abs(iq_data) > threshold)} large absolute values >{threshold}, setting to 0')
                    iq_data[np.abs(iq_data) > 10] = 0

                frac_nonzeros = np.count_nonzero(iq_data) / iq_data.size
                if frac_nonzeros < 0.5:
                    print(f'Skipping {folder}/{recording}, not enough nonzeros ({frac_nonzeros*100}%).')
                    continue

                data_row = {
                'recording': os.path.join(data_folder, folder, recording),
                'category': folder,
                'servo_azimuth': metadata['servo_azimuth'],
                'servo_elevation': metadata['servo_elevation'],
                'length': iq_data.shape[1]
                }
                recordings_list.append(data_row)

    print(f"Generated {len(recordings_list)} signals")
    df = pd.DataFrame(recordings_list)

    df_train, df_test = train_test_split(df,
                                        stratify=df['category'],
                                        test_size=1/5,
                                        random_state=SEED,
                                        )
    
    df_train, df_val = train_test_split(df_train,
                                        stratify=df_train['category'],
                                        test_size=1/10,
                                        random_state=SEED,
                                        )
    
    write_dataset(args, df_train, '_train')
    write_dataset(args, df_test, '_test')
    write_dataset(args, df_val, '_val')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="r22 H5 Dataset Builder")
    parser.add_argument('--name', '-n', type=str, default='r22', help="Name of the dataset.")
    parser.add_argument('--data_folder', '-d', type=str, help="Folder the recordings are located.")
    parser.add_argument('--save_location', '-s', type=str, help="Location to save dataset (optional, defaults to current dir)")
    parser.add_argument('--example_length', '-l', type=int, default=4096, help="Example length to use")
    parser.add_argument('--percentage', '-p', type=float, default=100, help="Percentage of files to use from each folder")

    args = parser.parse_args()
    create_dataset(args)
