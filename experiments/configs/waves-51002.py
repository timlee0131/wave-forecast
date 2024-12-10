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
    config.hidden_channels = 24
    
    # CNN model
    config.is_cnn = True
    config.cnn_hidden1 = 24
    config.cnn_hidden2 = 36
    config.fc_hidden = 12
    config.output_channels = 48
    config.kernel_size = 6
    config.stride = 1
    
    # training settings
    config.runs = 10
    config.epochs =200
    config.lr = 0.01
    config.min_lr = 1e-3
    
    # misc.
    config.n_step = [6 * 12, 6 * 24, 6 * 48, 6 * 72, 6 * 96]
    # config.n_step = [6 * 12]
    config.seq_len = 6 * 24
    config.train_ratio = 0.8
    config.verbose = True
    config.epoch_verbose = False
    
    return config