import numpy as np
import librosa
import json
import copy
from pathlib import Path

import sys
sys.path.insert(0, "")

# Add additional configs to prepare different dictionaries
import configs.data.config_ace_clean_01 as config_ace_clean_01


def create_dirs(d_config):
    Path(d_config.dataset_root_path).mkdir(parents=True, exist_ok=True)

def energy_calc(wave, window=300):
    cut = len(wave) % window
    x = wave
    if cut != 0:
        x = x[:-cut]
    x = x.reshape(-1,window)
    e = np.sum(x*x, 1)
    return e

def detect_from_energy(e, wave_len, d_config):
    quantile = np.quantile(e, 0.05)
    silence = np.where(e <= quantile)[0]
    e_size = len(e)
    w_size = wave_len
    factor = w_size/e_size
    silence = (silence* factor).astype(np.int64)
    
    start = d_config.ignore_start_sec*d_config.sampling_rate
    end = w_size-d_config.ignore_end_sec*d_config.sampling_rate

    idx_silence = np.where(np.logical_and(silence>=start, silence<=end))
    silence = silence[idx_silence]
    return silence

def select_split_pos(silence_list, max_selects=10):
    np.random.shuffle(silence_list)
    return silence_list[:max_selects].tolist()

def load_wavs_to_dict(speaker_file, d_config):
    speakers = speaker_file.keys()
    ret_dict = {}
    file_sizes = {}
    for s in speakers:
        temp_dict = {}
        temp_size_dict = {}
        for f in speaker_file[s]:
            f_path = d_config.data_path + s + f + "_none.wav"
            wave, sr = librosa.load(f_path, sr=d_config.sampling_rate)
            temp_dict[f] = wave
            temp_size_dict[f] = wave.shape[-1]
        ret_dict[s] = temp_dict
        file_sizes[s] = temp_size_dict
    return ret_dict, file_sizes

def generate_silence_dict(wave_dict, speaker_file, silence_save_path, d_config):
    output_dict = {}
    speakers = speaker_file.keys()
    for s in speakers:
        temp_dict = {}
        for f in speaker_file[s]:
            wave  = wave_dict[s][f]
            energy = energy_calc(wave, window=d_config.energy_window)
            silence_list = detect_from_energy(energy, wave.shape[-1], d_config)
            split_pos = select_split_pos(silence_list, max_selects=-1)
            temp_dict[f] = split_pos
        output_dict[s] = temp_dict

    with open(silence_save_path, 'w') as fp:
        json.dump(output_dict, fp)
    return output_dict

def select_valid_points(silence_pos_list, total_size, min_size=48000):
    # changing min_size allows smaller or bigger splice parts
    s1 = 0
    s2 = total_size

    valid_pos = copy.copy(silence_pos_list)
    valid_pos.append(0)
    valid_pos.append(total_size)
    valid_pos = np.asarray(valid_pos)
    
    new_pos = np.random.choice(valid_pos, size=1)[0]
    min_start = np.max([0, new_pos-min_size])
    min_end = np.min([total_size, new_pos+min_size])
    
    masked_pos = valid_pos[(valid_pos<min_start) | (valid_pos>min_end)]
    masked_pos = masked_pos[masked_pos!= new_pos]
    new_pos2 = np.random.choice(masked_pos, size=1)[0]

    s1 = int(np.min([new_pos, new_pos2]))
    s2 = int(np.max([new_pos, new_pos2]))
    return s1, s2

def generate_signal_combos(silence_dict, size_dict, rir_file, dataset_size, dataset_path, db_snr, compression_param, d_config):
    retVal = {}
    speakers = list(silence_dict.keys())
    rir_list = rir_file
    i = 0
    sampled_ids = []
    sampling_tries = 0
    max_sampling_tries = 0
    
    split_counts = np.random.randint(d_config.min_split_points, d_config.max_split_points+1, size=dataset_size)
    while i < dataset_size:
        try:
            s = np.random.choice(speakers, size=1)[0]
            
            if d_config.use_same_file:
                files_list = [np.random.choice(list(silence_dict[s].keys()))]
            else:
                files_list = list(silence_dict[s].keys())

            file_names = np.random.choice(files_list, size=split_counts[i]+1).tolist()
            rir_names = np.random.choice(rir_list, size=split_counts[i]+1).tolist()
            noise = np.random.choice(db_snr, 1)[0]
            if noise is not None:
                noise = int(noise)
            
            temp_compressions = np.random.choice(compression_param, d_config.max_compressions).tolist()
            compressions = []
            for ii in range(len(temp_compressions)):
                if temp_compressions[ii] is not None:
                    compr = int(np.random.choice(temp_compressions[ii]["compression"], 1)[0])
                    compressions.append({"format": temp_compressions[ii]["format"], "compression": compr})
                else:
                    compressions.append(None)
            
            cut_tuples = []
            cut_time_tuples = []

            total_size = 0
            total_time = 0
            splice_positions = []
            splice_times = []
            splice_bins = []
            sampled_id = s

            
            for fn in file_names:
                cut_start, cut_end = select_valid_points(silence_dict[s][fn], size_dict[s][fn], min_size=48000)
                cut_start_time = cut_start /d_config.sampling_rate
                cut_end_time = cut_end /d_config.sampling_rate
                cut_tuples.append((cut_start, cut_end))
                cut_time_tuples.append((cut_start_time, cut_end_time))
                splice_positions.append(total_size + (cut_end - cut_start))
                t = total_time + (cut_end_time - cut_start_time)
                splice_times.append(t)
                splice_bins.append(np.floor(t*2)/2)
                total_size += (cut_end - cut_start)
                total_time += (cut_end_time - cut_start_time)
                sampled_id += fn+str(cut_start)[:-3]+"_"+str(cut_end)[:-3]
                
            splice_positions.pop()
            splice_times.pop()
            splice_bins.pop()

            if split_counts[i] == 0:
                splice_times = [0]
                splice_bins = [0]
                splice_positions = [0]

            
            if (total_time >= d_config.min_time) and (total_time <= d_config.max_time):
                if (d_config.duplicate_strategy is not "allow") and (sampled_id in sampled_ids):
                    sampling_tries += 1
                    if d_config.duplicate_strategy == "disallow":
                        if sampling_tries < d_config.max_duplicate_retries:
                            continue
                        else:
                            print("Dictionary generation terminated!")
                            break
                    elif d_config.duplicate_strategy == "minimize":
                        if sampling_tries < d_config.max_duplicate_retries:
                            continue
                        else:
                            sampling_tries = 0
                retVal[i] = {
                    "id": i, 
                    "speaker": s, 
                    "files": file_names,
                    "rirs": rir_names,
                    "noise": noise,
                    "compressions": compressions,
                    "cut_points": cut_tuples,
                    "cut_times": cut_time_tuples,
                    "total_size": total_size, 
                    "total_time": total_time,
                    "splice_positions": splice_positions,
                    "splice_times": splice_times,
                    "splice_bins": splice_bins,
                    "sample_id": sampled_id
                }
                
                i+=1
                sampled_ids.append(sampled_id)
                if sampling_tries > max_sampling_tries:
                    max_sampling_tries = sampling_tries
                sampling_tries = 0
                if i%1000==0:
                    print(str(i)+"/"+str(dataset_size)+ " max tries: "+str(max_sampling_tries))

        except Exception as e:
            #print(str(e))
            #print(str(file_names)+ " failed. Trying a different file...")
            pass

    with open(dataset_path, "w") as fp:
        json.dump(retVal, fp)




def main():

    # Add additional configs to prepare different dictionaries
    configs = [
        config_ace_clean_01
    ]

    for d_config in configs:
        print("--- ", d_config.config_name, " ---")
        np.random.seed(d_config.random_seed)
        create_dirs(d_config)
        
        print("generating trainset...")
        speaker_file = d_config.speaker_file_train
        rir_file = d_config.rir_file_train
        dataset_size = d_config.dataset_size_train
        dataset_path = d_config.dataset_path_train

        loaded_wavs, file_sizes = load_wavs_to_dict(speaker_file, d_config)
        silence_dict = generate_silence_dict(loaded_wavs, speaker_file, d_config.silence_path_train, d_config)
        generate_signal_combos(silence_dict, file_sizes, rir_file, dataset_size, dataset_path, d_config.snr_db_train, d_config.compressions_train, d_config)

        print("generating valset...")
        speaker_file = d_config.speaker_file_val
        rir_file = d_config.rir_file_val
        dataset_size = d_config.dataset_size_val
        dataset_path = d_config.dataset_path_val

        loaded_wavs, file_sizes = load_wavs_to_dict(speaker_file, d_config)
        silence_dict = generate_silence_dict(loaded_wavs, speaker_file, d_config.silence_path_val, d_config)
        generate_signal_combos(silence_dict, file_sizes, rir_file, dataset_size, dataset_path, d_config.snr_db_val, d_config.compressions_val, d_config)

        print("generating testset...")
        speaker_file = d_config.speaker_file_test
        rir_file = d_config.rir_file_test
        dataset_size = d_config.dataset_size_test
        dataset_path = d_config.dataset_path_test

        loaded_wavs, file_sizes = load_wavs_to_dict(speaker_file, d_config)
        silence_dict = generate_silence_dict(loaded_wavs, speaker_file, d_config.silence_path_test, d_config)
        generate_signal_combos(silence_dict, file_sizes, rir_file, dataset_size, dataset_path, d_config.snr_db_test, d_config.compressions_test, d_config)

if __name__ == "__main__":
    main()