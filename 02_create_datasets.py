import pathlib

import librosa
import numpy as np
import torch
import pickle
import torchaudio 
import math
import torch.multiprocessing as mp
import torchaudio.functional as F
import json

import sys
sys.path.insert(0, "")
import configs.data.config_ace_clean_01 as config_ace_clean_01



def load_wavs_to_dict(d_config, speaker_file):
    ret_dict = {}
    for s in speaker_file:
        f_path = d_config.data_path + s + ".wav"
        wave, sr = librosa.load(f_path, sr=d_config.sampling_rate)
        ret_dict[s] = wave
    return ret_dict

def load_custom_noise(d_config):
    wave, sr = librosa.load(d_config.noise_file, sr=d_config.sampling_rate)
    return wave

def combine_signals(wav_dict, dataset_dict):
    retVal= []
    for row in dataset_dict:
        s = row["speaker"]
        files = row['files']
        rirs = row['rirs']
        cut_points = row["cut_points"]

        tensor_list = []
        for f, r, c in zip(files, rirs, cut_points):
            signal_name = s+f+"_"+r.replace("/","_")
            signal = wav_dict[signal_name][c[0]:c[1]]
            tensor = np.expand_dims(signal, 0)
            tensor = torch.from_numpy(tensor)
            tensor_list.append(tensor)

        retVal.append(tensor_list)
    return retVal

def concat_waves(wave_list):
    retVal = []
    for wave_pair in wave_list:
        retVal.append(torch.cat(wave_pair, axis=1))
    return retVal

def save_pickle(config, spec_list, process, idx):
    if process == "train":
        save_name = config.dataset_pickle_name_train
    elif process == "val":
        save_name = config.dataset_pickle_name_val
    elif process == "test":
        save_name = config.dataset_pickle_name_test

    save_name = save_name +"_"+str(idx)+".pkl"
    pickle.dump(spec_list, open(config.dataset_root_path+save_name, "wb"))
    print("saving done")

def torch_spectrogram(d_config, wave_list):
    n_fft = d_config.n_fft
    win_length = d_config.window_length
    hop_length = d_config.hop_length
    center=True
    pad_mode = "reflect"
    power=2.0

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length= win_length, 
        hop_length = hop_length,
        pad = 0, 
        power= power, 
        normalized= False, 
        center = center, 
        pad_mode= pad_mode, 
        onesided = True,)

    ret_list = []
    for wave in wave_list:
        spec = spectrogram(wave)
        spec = librosa.power_to_db(spec)
        ###
        spec = np.squeeze(spec)
        spec = np.transpose(spec, (1,0))
        spec = torch.from_numpy(spec)
        ###
        ret_list.append(spec)
    return ret_list

def torch_melspectrogram(d_config, wave_list):
    n_fft = d_config.n_fft
    win_length = d_config.window_length
    hop_length = d_config.hop_length
    center=True
    pad_mode = "reflect"
    power=2.0

    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=d_config.sampling_rate, 
        n_fft=n_fft,
        win_length= win_length, 
        hop_length = hop_length, 
        f_min= 0.0, 
        f_max = None, 
        pad = 0, 
        n_mels = d_config.emb_size, 
        power= power, 
        normalized= False, 
        center = center, 
        pad_mode= pad_mode, 
        onesided = True, 
        norm= None, 
        mel_scale= 'htk')

    ret_list = []
    for wave in wave_list:
        spec = spectrogram(wave)
        spec = librosa.power_to_db(spec)
        ###
        spec = np.squeeze(spec)
        spec = np.transpose(spec, (1,0))
        spec = torch.from_numpy(spec)
        ###
        ret_list.append(spec)
    return ret_list

def torch_mfcc(d_config, wave_list):
    mel_dict = {
        "n_fft":d_config.n_fft,
        "win_length": d_config.window_length, 
        "hop_length": d_config.hop_length, 
        "f_min": 0.0, 
        "f_max": None, 
        "pad": 0, 
        "n_mels": d_config.emb_size, 
        "power": 2.0, 
        "normalized":False, 
        "center":True, 
        "pad_mode": "reflect", 
        "onesided": True, 
        "norm": None, 
        "mel_scale": 'htk'
    }

    spectrogram = torchaudio.transforms.MFCC(
        sample_rate=d_config.sampling_rate, 
        norm= "ortho",
        n_mfcc = 20,
        dct_type= 2,
        log_mels= False,
        melkwargs=mel_dict
        )

    ret_list = []
    for wave in wave_list:
        spec = spectrogram(wave)
        spec = np.squeeze(spec)
        spec = np.transpose(spec, (1,0))
        ret_list.append(spec)
    return ret_list

def torch_spectral_centroids(d_config, wave_list):
    win_length = d_config.window_length
    hop_length = d_config.hop_length
    spectrogram = torchaudio.transforms.SpectralCentroid(
        sample_rate=d_config.sampling_rate,
        n_fft=d_config.n_fft,
        win_length= win_length, 
        hop_length = hop_length,
        pad = 0
        )

    ret_list = []
    for wave in wave_list:
        spec = spectrogram(wave)
        spec = torch.transpose(spec, 0, 1)
        ret_list.append(spec)
    return ret_list

def torch_cqt(d_config, wave_list):
    hop_length = d_config.n_fft//2
    ret_list = []
    i = 0
    for wave in wave_list:
        spec = np.abs(librosa.cqt(wave.numpy(), sr=d_config.sampling_rate, hop_length=hop_length))
        spec = librosa.power_to_db(spec)
        
        spec = np.squeeze(spec)
        spec = np.transpose(spec, (1,0))
        spec = torch.from_numpy(spec)
        
        ret_list.append(spec)
        print(i, "/", len(wave_list), end="\r")
        i += 1
    return ret_list

def torch_vqt(d_config, wave_list):
    hop_length = d_config.n_fft//2
    ret_list = []
    for wave in wave_list:
        spec = np.abs(librosa.vqt(wave.numpy(), sr=d_config.sampling_rate, hop_length=hop_length))
        spec = librosa.power_to_db(spec)
        
        spec = np.squeeze(spec)
        spec = np.transpose(spec, (1,0))
        spec = torch.from_numpy(spec)
        
        ret_list.append(spec)
    return ret_list

def torch_flatness(d_config, wave_list):
    hop_length = d_config.n_fft//2
    ret_list = []
    for wave in wave_list:
        spec = librosa.feature.spectral_flatness(y=wave.numpy(), n_fft=d_config.n_fft, hop_length=hop_length)
        spec = spec.reshape(-1,1)
        spec = torch.from_numpy(spec)
        ret_list.append(spec)
    return ret_list



def apply_noise(wave_list, dataset_dict, d_config):
    if d_config.noise_file is not "whitenoise":
        custom_noise = load_custom_noise(d_config)
    for i in range(len(wave_list)):
        snr_db = dataset_dict[i]["noise"]
        if snr_db is None:
            continue
        if d_config.noise_file is "whitenoise":
            noise = (0.001**0.5)*torch.randn(1, wave_list[i].shape[1])
        else:
            noise = custom_noise[:wave_list[i].shape[1]]
            noise = torch.from_numpy(noise)
        sig_power = wave_list[i].norm(p=2)
        noise_power = noise.norm(p=2)

        
        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / sig_power
        wave_list[i] = (scale * wave_list[i]  + noise)/2

        # Skalierung auf dieselbe lautstärke
        augmented_power = wave_list[i].norm(p=2)
        a_scale = sig_power/augmented_power
        wave_list[i] = wave_list[i] * a_scale
    return wave_list

def apply_codec(wave_list, dataset_dict, d_config, processes=1):
    def worker(w, d, wav_start_index, queue):
        for wav_subset_index in range(len(w)):
            for c in d[wav_subset_index]["compressions"]:
                if c is None:
                    continue
                sr = d_config.sampling_rate
                if c["format"] == "amr-nb":
                    sr = d_config.sampling_rate//2
                w[wav_subset_index] = F.apply_codec(w[wav_subset_index], sample_rate=sr, **c)
        queue.put((w,wav_start_index))
        done.wait()
    
    i = 0
    q = mp.Queue()
    proc_list = []
    rets = []
    wave_len = len(wave_list)
    per_proc_size = int(np.ceil(wave_len/processes))

    for _ in range(processes):
        p_wave_list = wave_list[i:per_proc_size+i]
        p_dataset_dict = dataset_dict[i:per_proc_size+i]

        done = mp.Event()
        p = mp.Process(target=worker, args=(p_wave_list, p_dataset_dict, i, q))
        proc_list.append([p, done])
        p.start()

        i += per_proc_size

    for _ in range(processes):
        ret = q.get()
        rets.append(ret)
        del ret
        

    rets.sort(key=lambda x: x[1])
    retVal = []
    for r in rets:
        retVal += r[0]


    
    for p, done in proc_list:
        done.set()
        p.join()
        p.close()
    wave_list = retVal
    return wave_list


def create_all():
    configs = [
        config_ace_clean_01,
    ]

    print("preparing datasets...")

    for d_config in configs:
        print("processing: ",d_config.config_name)
        pathlib.Path(d_config.dataset_root_path).mkdir(parents=True, exist_ok=True)

        dataset_path = [d_config.dataset_path_train, d_config.dataset_path_val, d_config.dataset_path_test]
        processing = ["train", "val", "test"]

        print("loading data")
        preped_files = []
        for speaker in d_config.speaker_file_all:
            for w in d_config.speaker_file_all[speaker]:
                for r in d_config.rir_file_all:
                    preped_files.append(speaker+w+"_"+r.replace("/", "_"))
        wav_dict = load_wavs_to_dict(d_config, preped_files)

        print("loading testdata")
        preped_files_test = []
        for speaker in d_config.speaker_file_test:
            for w in d_config.speaker_file_test[speaker]:
                for r in d_config.rir_file_test:
                    preped_files_test.append(speaker+w+"_"+r.replace("/", "_"))
        wav_dict_test = load_wavs_to_dict(d_config, preped_files_test)

        for dataset_p, process in zip(dataset_path, processing):
            print("processing: ",process)

            if dataset_p == None:
                print("skipping ",process)
                continue
            
            if process == "test":
                wav_dict = wav_dict_test

            print("loading dataset json")
            with open(dataset_p, "r") as fp:
                dataset_dict = json.load(fp)
            dataset_dict = list(dataset_dict.values())

            package_size = d_config.package_size
            curr_pos = 0

            while curr_pos < len(dataset_dict):
                print("processing rows: "+str(curr_pos)+" - "+str(curr_pos+package_size)+"/"+str(len(dataset_dict)))

                print("generating WAVEs")
                wave_list = combine_signals(wav_dict, dataset_dict[curr_pos:curr_pos+package_size])

                print("concatenating cut parts")
                wave_list = concat_waves(wave_list)

                print("adding noise")
                wave_list = apply_noise(wave_list, dataset_dict[curr_pos:curr_pos+package_size], d_config)
#
                print("applying codecs")
                # WARNING generating multiplie processes, uses a lot of ram
                # set processes or package_size (configs/data/your_config.py) to a smaller value if you run out of memory 
                wave_list = apply_codec(wave_list, dataset_dict[curr_pos:curr_pos+package_size], d_config, processes=8)

                print("calculating spectrograms")

                spec_list = []
                for spec_type in d_config.input_specs_types:
                    if spec_type == "melspec":
                        print("adding melspectrogram...")
                        spec_list.append(torch_melspectrogram(d_config, wave_list))
                    elif spec_type == "mfcc":
                        print("adding MFCC...")
                        spec_list.append(torch_mfcc(d_config, wave_list))
                    elif spec_type == "spec":
                        print("adding spectrogram...")
                        spec_list.append(torch_spectrogram(d_config, wave_list))
                    elif spec_type == "flatness":
                        print("adding flatness...")
                        spec_list.append(torch_flatness(d_config, wave_list))
                    elif spec_type == "cqt":
                        print("adding cqt...")
                        spec_list.append(torch_cqt(d_config, wave_list))
                    elif spec_type == "vqt":
                        print("adding vqt...")
                        spec_list.append(torch_vqt(d_config, wave_list))
                    elif spec_type == "centroid":
                        print("adding spectral centroids...")
                        spec_list.append(torch_spectral_centroids(d_config, wave_list))


                print("saving arrays")
                save_pickle(d_config, spec_list, process, curr_pos)

                curr_pos += package_size



if __name__ == "__main__":
    create_all()