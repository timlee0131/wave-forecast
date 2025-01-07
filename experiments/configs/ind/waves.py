import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./experiments/data/pt/ind"
        if config.computer == "local"
        else "/data"
    )
    
    # dataset info
    config.dataset = 'waves'
    config.num_features = 6
    
    # MLP model
    config.loss_fn = 'l1'
    config.mlp_hidden1 = 32
    config.mlp_hidden2 = 48
    
    # CNN model
    config.is_cnn = True
    config.cnn_hidden1 = 24
    config.cnn_hidden2 = 48
    config.fc_hidden = 24
    config.output_channels = 24
    config.kernel_size = 6
    config.stride = 1
    
    # training settings
    config.runs = 10
    config.epochs = 200
    config.lr = 5e-3
    config.min_lr = 1e-4
    
    # misc.
    config.look_ahead = 1
    config.n_step = 6
    config.seq_len = 12   # look-back steps (k)
    config.train_ratio = 0.8
    config.verbose = True
    config.epoch_verbose = False
    config.time_verbose = True
    
    return config