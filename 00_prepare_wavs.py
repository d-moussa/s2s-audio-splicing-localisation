import librosa
import torch
import numpy as np
import torchaudio
from timeit import default_timer as timer
import os

import sys
sys.path.insert(0, "")

# Add additional configs to prepare different samples
import configs.data.config_ace_clean_01 as config_ace_clean_01


data_raw_path = "data/raw_speech/"
wav_save_path = "data/prepared/"
rir_save_path = "data/rir/"

def transform_dict_to_list(wav_dict):
    retVal = []
    speakers = wav_dict.keys()
    for s in speakers:
        for f in wav_dict[s]:
            retVal.append(s+f)
    return retVal

def load_wavs(wav_files, save_path, sr):
    ret_dict = {}
    i = 0
    for k in wav_files:
        print("loading: ",k, "current: ",i, " total: ",len(wav_files))
        i+=1
        f_path = save_path + k + ".wav"
        if k is "none":
            ret_dict[k] = None
            continue
        print(f_path)
        wave, _ = librosa.load(f_path, sr=sr)
        ret_dict[k] = wave
    return ret_dict

def combine_rir_wav(wavs, rirs, sub_dir, sr):
    wav_keys = wavs.keys()
    rir_keys = rirs.keys()

    i = 1
    for w in wav_keys:
        for r in rir_keys:
            w_file = wavs[w]
            r_file = rirs[r]
            new_name = w+"_"+r.replace("/", "_")

            if os.path.isfile(wav_save_path+sub_dir+new_name+'.wav'):
                continue
            
            if r_file is None:
                tensor = np.expand_dims(w_file, 0)
                tensor = torch.from_numpy(tensor)
                torchaudio.save(wav_save_path+new_name+'.wav', tensor, sr)
                i += 1
                continue

            tensor = np.expand_dims(w_file, 0)
            r_file = np.expand_dims(r_file, 0)
            tensor = torch.from_numpy(tensor)
            r_file = torch.from_numpy(r_file)
            sig_w = torch.nn.functional.pad(tensor, (r_file.shape[1]-1, 1))
            agumented_w = torch.nn.functional.conv1d(sig_w[None, ...], r_file[None, ...])[0]
            torchaudio.save(wav_save_path+sub_dir+new_name+'.wav', agumented_w, sr)
            print(i,"/",(len(wav_keys)*len(rir_keys)))
            i +=1


def main():
    print("Adding RIRs to raw audio samples")
    
    # Add additional configs to prepare different samples
    configs = [
        config_ace_clean_01
    ]
    for config in configs:
        print("--- ",config.config_name, " ---")
        start_time = timer()
        wav_files = transform_dict_to_list(config.speaker_file_all)
        wav = load_wavs(wav_files, data_raw_path, config.sampling_rate)
        rir = load_wavs(config.rir_file_all, rir_save_path, config.sampling_rate)


        combine_rir_wav(wav, rir, "", config.sampling_rate)
        end_time = timer()
        total_time = end_time - start_time
        print("Train and Validation created... total time: ",total_time)

        start_time = timer()
        wav_files = transform_dict_to_list(config.speaker_file_test)
        wav = load_wavs(wav_files, data_raw_path, config.sampling_rate)
        rir = load_wavs(config.rir_file_test, rir_save_path, config.sampling_rate)
        
        
        combine_rir_wav(wav, rir, "", config.sampling_rate)
        end_time = timer()
        total_time = end_time - start_time
        print("Test created... total time: ",total_time)

if __name__ == "__main__":
    main()
