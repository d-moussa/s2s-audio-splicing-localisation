random_seed = 42
config_name = "ACE clean 0-1 Data Config"
set_name = "ace"
experiment_name = "clean"

data_path = "data/prepared/"     # Path to processed speech audio

min_split_points = 0                        # Minimal number of splices per sample
max_split_points = 1                        # Maximal number of splices per sample
dataset_size_train = 50000                  # Number of samples in train set
dataset_size_val = 3000                     # Number of samples in validation set
dataset_size_test = 3000                    # Number of samples in test set
package_size = 5000                         # Splits the datasets in smaller packages of this size

duplicate_strategy = "allow"                # Duplicate handling "disallow" = no duplicates allowed (slowest), "minimize" = try to minimize duplicate count, "allow" = do nothing (fastest)
max_duplicate_retries = 5                   # if disallow - finishes datageneration after N tries, if minimize - tries N times to generate a new sample generation will continue
dataset_root_path = "datasets/ace/clean/"               # Path to the actual datasets and to the dataset dictionaries                 
dataset_pickle_name_train = "train_clean_single"                                # Filenames
dataset_pickle_name_val = "val_clean_single"
dataset_pickle_name_test = "test_clean_single"
dataset_path_train = "datasets/ace/clean/dataset_train_ace_01.json"
dataset_path_val = "datasets/ace/clean/dataset_val_ace_01.json"
dataset_path_test = "datasets/ace/clean/dataset_test_ace_01.json"

silence_path_train = "datasets/ace/clean/silence_train_ace_01.json"
silence_path_val = "datasets/ace/clean/silence_val_ace_01.json"
silence_path_test = "datasets/ace/clean/silence_test_ace_01.json"
use_same_file = False                       # Use same file for datageneration (usefull for intersplicing)
#
rir_file_all = [                                                        # Used rirs needed for generation of processed samples
    "none",
    "raw_others_pra/pos_2/pos_2_room_11_1.16_pra",
    "raw_others_pra/pos_2/pos_2_room_7_0.74_pra",
    "raw_others_pra/pos_2/pos_2_room_4_0.53_pra",
    "raw_others_pra/pos_1/pos_1_room_12_1.41_pra",
    "raw_others_pra/pos_1/pos_1_room_1_0.33_pra",
    "paper_pra/pos_2/pos_2_room_1_0.31_pra",
    "paper_pra/pos_2/pos_2_room_5_0.52_pra",
    "paper_pra/pos_1/pos_1_room_2_0.35_pra",
    "paper_ACE/pos_2/Lecture_Room_2_Single_403a_1_7_1.25_ace",
    "paper_ACE/pos_2/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_2/Office_2_Single_803_1_3_0.39_ace",
    "paper_ACE/pos_1/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_1/Meeting_Room_2_Single_611_1_2_0.37_ace",
    "paper_ACE/pos_1/Office_1_Single_502_1_1_0.34_ace",
    ]
rir_file_train = ["none",                                               # Used rirs for the train set (needed for dataset generation)
    "raw_others_pra/pos_2/pos_2_room_11_1.16_pra",
    "raw_others_pra/pos_2/pos_2_room_7_0.74_pra",
    "raw_others_pra/pos_2/pos_2_room_4_0.53_pra",
    "raw_others_pra/pos_1/pos_1_room_12_1.41_pra",
    "raw_others_pra/pos_1/pos_1_room_1_0.33_pra",
    "paper_pra/pos_2/pos_2_room_1_0.31_pra",
    "paper_pra/pos_2/pos_2_room_5_0.52_pra",
    "paper_pra/pos_1/pos_1_room_2_0.35_pra",
    "paper_ACE/pos_2/Lecture_Room_2_Single_403a_1_7_1.25_ace",
    "paper_ACE/pos_2/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_2/Office_2_Single_803_1_3_0.39_ace",
    "paper_ACE/pos_1/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_1/Meeting_Room_2_Single_611_1_2_0.37_ace",
    "paper_ACE/pos_1/Office_1_Single_502_1_1_0.34_ace",
    ]
rir_file_val = ["none",                                                 # Used rirs for the validation set (needed for dataset generation)
    "raw_others_pra/pos_2/pos_2_room_11_1.16_pra",
    "raw_others_pra/pos_2/pos_2_room_7_0.74_pra",
    "raw_others_pra/pos_2/pos_2_room_4_0.53_pra",
    "raw_others_pra/pos_1/pos_1_room_12_1.41_pra",
    "raw_others_pra/pos_1/pos_1_room_1_0.33_pra",
    "paper_pra/pos_2/pos_2_room_1_0.31_pra",
    "paper_pra/pos_2/pos_2_room_5_0.52_pra",
    "paper_pra/pos_1/pos_1_room_2_0.35_pra",
    "paper_ACE/pos_2/Lecture_Room_2_Single_403a_1_7_1.25_ace",
    "paper_ACE/pos_2/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_2/Office_2_Single_803_1_3_0.39_ace",
    "paper_ACE/pos_1/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_1/Meeting_Room_2_Single_611_1_2_0.37_ace",
    "paper_ACE/pos_1/Office_1_Single_502_1_1_0.34_ace",
    ]
rir_file_test = ["none",                                                # Used rirs for the test set (needed for dataset generation)
    "raw_others_pra/pos_2/pos_2_room_11_1.16_pra",
    "raw_others_pra/pos_2/pos_2_room_7_0.74_pra",
    "raw_others_pra/pos_2/pos_2_room_4_0.53_pra",
    "raw_others_pra/pos_1/pos_1_room_12_1.41_pra",
    "raw_others_pra/pos_1/pos_1_room_1_0.33_pra",
    "paper_pra/pos_2/pos_2_room_1_0.31_pra",
    "paper_pra/pos_2/pos_2_room_5_0.52_pra",
    "paper_pra/pos_1/pos_1_room_2_0.35_pra",
    "paper_ACE/pos_2/Lecture_Room_2_Single_403a_1_7_1.25_ace",
    "paper_ACE/pos_2/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_2/Office_2_Single_803_1_3_0.39_ace",
    "paper_ACE/pos_1/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_1/Meeting_Room_2_Single_611_1_2_0.37_ace",
    "paper_ACE/pos_1/Office_1_Single_502_1_1_0.34_ace",
    ]

speaker_file_all = {                                                    # Used speakerfiles needed for generation of processed samples
    "F1":["s1", "s2", "s3", "s4", "s5"],
    "F2":["s1", "s2", "s3", "s4", "s5"],
    "F3":["s1", "s2", "s3", "s4", "s5"],
    "F4":["s1", "s2", "s3", "s4", "s5"],
    "F5":["s1", "s2", "s3", "s4", "s5"],
    "M1":["s1", "s2", "s3", "s4", "s5"],
    "M2":["s1", "s2", "s3", "s4", "s5"],
    "M3":["s1", "s2", "s3", "s4", "s5"],
    "M4":["s1", "s2", "s3", "s4", "s5"],
    "M5":["s1", "s2", "s3", "s4", "s5"],
    "M6":["s3", "s4"],
    "M7":["s3", "s4"],
    "M8":["s3", "s4"],
    "M9":["s3", "s4"]
}
speaker_file_train = {                                              # Used speakerfiles for the train set (needed for dataset generation)
    "F1":["s1", "s2", "s3", "s4", "s5"],
    "F2":["s1", "s2", "s3", "s4", "s5"],
    "F3":["s1", "s2", "s3", "s4", "s5"],
    "F4":["s1", "s2", "s3", "s4", "s5"],
    "M2":["s1", "s2", "s3", "s4", "s5"],
    "M3":["s1", "s2", "s3", "s4", "s5"],
    "M4":["s1", "s2", "s3", "s4", "s5"],
    "M5":["s1", "s2", "s3", "s4", "s5"],
    "M6":["s3", "s4"],
    "M7":["s3", "s4"],
}
speaker_file_val = {                                                # Used speakerfiles for the validation set (needed for dataset generation)
    "M8":["s3", "s4"],
    "M9":["s3", "s4"]
}
speaker_file_test = {                                               # Used speakerfiles for the test set (needed for dataset generation)
    "F5":["s1", "s2", "s3", "s4", "s5"],
    "M1":["s1", "s2", "s3", "s4", "s5"],
}

noise_file = "whitenoise"                                           # Noise file whitenoise - generates whitenoise or path (e.g. "data/custom_noise/airport.wav") - use custom noise
snr_db_train = [None]                                               # SNR range for training None or list of integers
snr_db_val = [None]                                                 # SNR range for validation None or list of integers
snr_db_test = [None]                                                # SNR range for testing None or list of integers

max_compressions = 1                                                # Number of iterative compressions
compressions_train = [None]                                         # Type of compressions for training None or list of dicts
#compressions_train = [                                             # Example
#    {"format":"mp3", "compression":list(range(10,129))}, 
#    {"format":"amr-nb", "compression":[0,1,2,3,4,5,6,7]}
#    ]

compressions_val = [None]                                           # Type of compressions for validation None or list of dicts
compressions_test = [None]                                          # Type of compressions for testing None or list of dicts

sampling_rate = 16000                                               # Sampling rate of audio files


energy_window = 300                                                 # Window size for silence detection
ignore_start_sec = 0.0                                              # Ignore first seconds of the audio file
ignore_end_sec = 0.0                                                # Ignore last seconds of the audio file
min_time = 5.0                                                      # Minimal spliced audio length
max_time = 45.0                                                     # Maximal spliced audio length

n_fft = 16000                                                       # Size of FFT (fast fourier transform)
window_length = None                                                # Window size (None or int)
hop_length = None                                                   # Length of hop between STFT windows (None or int)

emb_size = 256                                                      # Melspectrogram bins
input_specs_dims = [256, 20, 1]                                     # Height of multiinput representations or None for single input
input_specs_types = ["melspec", "mfcc", "centroid"]                 # Types of multiinput representations or None for single input