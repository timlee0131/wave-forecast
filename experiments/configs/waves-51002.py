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
    config.dataset = 'waves-51002'
    config.num_features = 10
    
    # MLP model
    config.loss_fn = 'l1'
    config.hidden_channels = 480
    
    # CNN model
    config.is_cnn = True
    config.cnn_hidden = 24
    config.output_channels = 48
    config.kernel_size = 6
    config.stride = 1
    
    # training settings
    config.runs = 10
    config.epochs =500
    config.lr = 0.005
    config.min_lr = 1e-4
    
    # misc.
    config.n_step = 24
    config.seq_len = 6 * 48
    config.train_ratio = 0.8
    config.verbose = True
    
    return config