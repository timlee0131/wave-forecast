import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./experiments/data/pt/group"
        if config.computer == "local"
        else "/data"
    )
    
    # dataset info
    config.dataset = 'labn'
    config.num_nodes = 5
    config.num_features = 16

    # Time then Space
    config.time_hidden = 240
    config.time_out = 60
    config.space_hidden = 32
    config.space_out = 1
    
    # training settings
    config.runs = 10
    config.epochs = 1000
    config.lr = 0.005
    config.min_lr = 5e-4
    
    # misc.
    config.look_back = 24
    config.horizon = 6 * 72
    
    config.verbose = True
    config.epoch_verbose = False
    config.time_verbose = False
    
    return config