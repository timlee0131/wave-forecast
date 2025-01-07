import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./experiments/data/pt"
        if config.computer == "local"
        else "/data"
    )
    
    # dataset info
    config.dataset = 'waves-2024'
    config.num_features = 9

    # MLP model
    config.loss_fn = 'l1'
    config.mlp_hidden1 = 32
    config.mlp_hidden2 = 8
    
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
    config.epochs = 1000
    config.lr = 0.001
    config.min_lr = 1e-3
    
    # misc.
    config.n_step = [6 * 12, 6 * 24, 6 * 36, 6 * 48, 6 * 72, 6 * 96]    # look-ahead steps
    # config.n_step = [6 * 12]
    config.seq_len = 6 * 72   # look-back steps
    config.train_ratio = 0.8
    config.verbose = True
    config.epoch_verbose = False
    config.time_verbose = False

# LEGACY 
    # # MLP model
    # config.loss_fn = 'l1'
    # config.hidden_channels = 32
    
    # # CNN model
    # config.is_cnn = True
    # config.cnn_hidden = 32
    # config.output_channels = 16
    # config.kernel_size = 6
    # config.stride = 2
    
    # # training settings
    # config.runs = 10
    # config.epochs = 1000
    # config.lr = 0.001
    # config.min_lr = 1e-4
    
    # # misc.
    # config.n_step = 72
    # config.seq_len = 24
    # config.train_ratio = 0.8
    # config.verbose = True
    
    return config