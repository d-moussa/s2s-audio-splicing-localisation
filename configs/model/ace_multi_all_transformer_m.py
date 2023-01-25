num_encoder_layers = 5              # Number of encoder layers
num_decoder_layers = 5              # Number of decoder layers
emb_size = 279                      # Embedding size of the encoder
nhead= 9                            # Number of attention heads encoder and decoder for single input or decoder for multi input
ffn_hid_dim = 512                   # Sublayer feedforward dimension
batch_size = 384                    # Batch size
num_epochs = 10                     # Number of training epochs
lr = 0.0001                         # Learning rate
betas=(0.9, 0.98)                   # Adam optimizer beta parameters
eps=1e-9                            # Adam optimizer eps parameter
dropout = 0.2                       # Dropout


save_model_dir = "models/transformer/ace/"                  # Model save location
early_stopping_wait = 3                                     # Wait for N epochs for improvement
early_stopping_delta = 0.                                   # Minimal accepted improvement

load_checkpoint_path = None                                 # Path for retraining/finetuning or None for a new training

model_name = "transformer_m"
train_set_name = "ace"
splice_name = "multi"
experiment_name = "allMultiinput"

use_single_encoder = True                                  # Uses single input Transformer if True or multi input Transformer if False
encoder_memory_transformation = "concatenate"               # Encoder strategy: projection, concatenate, None
#memory_layer_args = {}
enc_args_list = [                                           # List of dicts for each encoder input or empty list for single input
            {"d_model": 256, "nhead":8, "dim_feedforward": 256, "num_layers":5},
            {"d_model": 20, "nhead":4, "dim_feedforward": 256, "num_layers":5},
            {"d_model": 3, "nhead":3, "dim_feedforward": 64, "num_layers":5},
        ]
