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
    config.hidden_channels = 32
    
    # CNN model
    config.is_cnn = True
    config.cnn_hidden = 32
    config.output_channels = 16
    config.kernel_size = 6
    config.stride = 2
    
    # training settings
    config.runs = 10
    config.epochs = 1000
    config.lr = 0.001
    config.min_lr = 1e-4
    
    # misc.
    config.n_step = 72
    config.seq_len = 24
    config.train_ratio = 0.8
    config.verbose = True
    
    return config