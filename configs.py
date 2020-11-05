import os
import numpy as np

from settings import BASE_DIR


def create_config(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3,
                    max_agents=40, n_input_channels=(10 + 1)*2 + 3, n_output=50*3, history_step_size=1):

    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network, 
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },
        
        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },
        
        'train_data_loader': {
            'key': 'scenes/train.zarr',
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },
        
        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,
            
            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
        'key': 'scenes/test.zarr',
        'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'samples_per_epoch': 10E+10
        },
    
        'val_data_loader': {
            'key': 'scenes/validate_chopped_100/validate.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10000
        }
    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


def create_config_train_full(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3,
                    max_agents=40, n_input_channels=(10 + 1)*2 + 3, n_output=50*3, history_step_size=1):

    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network, 
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },
        
        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },
        
        'train_data_loader': {
            'key': 'scenes/train_full.zarr',
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },
        
        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,
            
            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
        'key': 'scenes/test.zarr',
        'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'samples_per_epoch': 10E+10
        },
    
        'val_data_loader': {
            'key': 'scenes/validate_chopped_100/validate.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10000
        }
    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


def create_prep_config(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5],
                  batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10,
                  history_num_frames=10, future_num_frames=50, n_modes=3,
                  max_agents=40, n_input_channels=(10 + 1) * 2 + 3, n_output=50 * 3, history_step_size=1):
    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network,
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },

        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },

        'train_data_loader': {
            'key': 'scenes/train_full.zarr',
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,

            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
            'key': 'scenes/test.zarr',
            'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10E+10
        },

        'val_data_loader': {
            'key': 'scenes/validate.zarr',
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10000
        }
    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


def create_config_multi_train_chopped_lite(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3,
                    max_agents=40, n_input_channels=(10 + 1)*2 + 3, n_output=50*3, history_step_size=1):

    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network, 
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },
        
        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },
        
        'train_data_loader_10': {
            'key': 'scenes/train_full_chopped_10_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_10_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_10_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_20': {
            'key': 'scenes/train_full_chopped_20_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_20_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_20_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_30': {
            'key': 'scenes/train_full_chopped_30_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_30_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_30_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_40': {
            'key': 'scenes/train_full_chopped_40_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_40_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_40_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_50': {
            'key': 'scenes/train_full_chopped_50_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_50_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_50_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_60': {
            'key': 'scenes/train_full_chopped_60_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_60_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_60_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_70': {
            'key': 'scenes/train_full_chopped_70_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_70_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_70_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_80': {
            'key': 'scenes/train_full_chopped_80_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_80_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_80_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_90': {
            'key': 'scenes/train_full_chopped_90_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_90_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_90_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_100': {
            'key': 'scenes/train_full_chopped_100_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_100_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_100_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_110': {
            'key': 'scenes/train_full_chopped_110_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_110_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_110_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_120': {
            'key': 'scenes/train_full_chopped_120_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_120_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_120_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_130': {
            'key': 'scenes/train_full_chopped_130_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_130_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_130_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_140': {
            'key': 'scenes/train_full_chopped_140_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_140_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_140_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_150': {
            'key': 'scenes/train_full_chopped_150_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_150_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_150_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_160': {
            'key': 'scenes/train_full_chopped_160_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_160_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_160_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_170': {
            'key': 'scenes/train_full_chopped_170_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_170_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_170_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_180': {
            'key': 'scenes/train_full_chopped_180_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_180_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_180_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_190': {
            'key': 'scenes/train_full_chopped_190_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_190_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_190_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_200': {
            'key': 'scenes/train_full_chopped_200_lite/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_200_lite/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_200_lite/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },
        
        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,
            
            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
            'key': 'scenes/test.zarr',
            'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10E+10
        },
    
        'val_data_loader': {
            'key': 'scenes/validate_chopped_100/validate.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10000
        },

    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


def create_config_multi_train_chopped(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3,
                    max_agents=40, n_input_channels=(10 + 1)*2 + 3, n_output=50*3, history_step_size=1):

    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network, 
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },
        
        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },
        
        'train_data_loader_10': {
            'key': 'scenes/train_full_chopped_10_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_10_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_10_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_30': {
            'key': 'scenes/train_full_chopped_30_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_30_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_30_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_50': {
            'key': 'scenes/train_full_chopped_50/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_50/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_50/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_70': {
            'key': 'scenes/train_full_chopped_70_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_70_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_70_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_90': {
            'key': 'scenes/train_full_chopped_90_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_90_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_90_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_100': {
            'key': 'scenes/train_full_chopped_100/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_100/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_100/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_110': {
            'key': 'scenes/train_full_chopped_110_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_110_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_110_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_130': {
            'key': 'scenes/train_full_chopped_130_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_130_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_130_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_150': {
            'key': 'scenes/train_full_chopped_150_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_150_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_150_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_180': {
            'key': 'scenes/train_full_chopped_180_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_180_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_180_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_200': {
            'key': 'scenes/train_full_chopped_200_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_200_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_200_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },
        
        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,
            
            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
            'key': 'scenes/test.zarr',
            'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10E+10
        },
    
        'val_data_loader': {
            'key': 'scenes/validate_chopped_100/validate.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10000
        },

    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


def create_config_orig(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3,
                    max_agents=40, n_input_channels=(10 + 1)*2 + 3, n_output=50*3, history_step_size=1):
    """
    val_data_loader not based on chopped dataset
    train.zarr
    """
    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network, 
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },
        
        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },
        
        'train_data_loader': {
            'key': 'scenes/train.zarr',
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },
        
        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,
            
            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
        'key': 'scenes/test.zarr',
        'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'samples_per_epoch': 10E+10
        },
    
        'val_data_loader': {
            'key': 'scenes/validate.zarr',
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 20000
        }
    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


def create_config_tl_future(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3,
                    max_agents=40, n_input_channels=(10 + 1)*2 + 3, n_output=50*3, history_step_size=1):

    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network, 
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },
        
        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'tl_future': True,
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },
        
        'train_data_loader': {
            'key': 'scenes/train.zarr',
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },
        
        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,
            
            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
        'key': 'scenes/test.zarr',
        'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'samples_per_epoch': 10E+10
        },
    
        'val_data_loader': {
            'key': 'scenes/validate.zarr',
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 20000
        }
    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


def create_config_tl(str_network='', cfg_model_params=None, input_size=224, pixel_size=[0.5, 0.5], ego_center=[0.25, 0.5], batch_size=12, num_workers=16, samples_per_epoch=0, sample_history_num_frames=10, history_num_frames=10, future_num_frames=50, n_modes=3,
                    max_agents=40, n_input_channels=(10 + 1)*2 + 3, n_output=50*3, history_step_size=1):

    cfg = {
        'format_version': 4,
        'model_params': {
            'model_architecture': str_network, 
            'n_input_channels': n_input_channels,
            'n_output': n_output,
            'sample_history_num_frames': sample_history_num_frames,
            'history_num_frames': history_num_frames,
            'history_step_size': history_step_size,
            'history_delta_time': 0.1,
            'future_num_frames': future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
            'n_modes': n_modes
        },
        
        'raster_params': {
            'raster_size': [input_size, input_size],
            'pixel_size': pixel_size,
            'ego_center': ego_center,
            'map_type': 'py_semantic',
            'tl_persistence': True,
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5
        },
        
        'train_data_loader_10': {
            'key': 'scenes/train_full_chopped_10_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_10_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_10_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_30': {
            'key': 'scenes/train_full_chopped_30_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_30_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_30_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_50': {
            'key': 'scenes/train_full_chopped_50/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_50/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_50/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_70': {
            'key': 'scenes/train_full_chopped_70_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_70_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_70_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_90': {
            'key': 'scenes/train_full_chopped_90_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_90_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_90_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_100': {
            'key': 'scenes/train_full_chopped_100/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_100/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_100/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },

        'train_data_loader_110': {
            'key': 'scenes/train_full_chopped_110_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_110_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_110_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_130': {
            'key': 'scenes/train_full_chopped_130_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_130_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_130_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_150': {
            'key': 'scenes/train_full_chopped_150_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_150_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_150_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_180': {
            'key': 'scenes/train_full_chopped_180_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_180_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_180_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },


        'train_data_loader_200': {
            'key': 'scenes/train_full_chopped_200_CF/train_full.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_200_CF/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/train_full_chopped_200_CF/mask.npz'),
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'samples_per_epoch': samples_per_epoch
        },
        
        'train_params': {
            'max_num_steps': 10000,
            'checkpoint_every_n_steps': 5000,
            
            # 'eval_every_n_steps': -1
        },

        'test_data_loader': {
            'key': 'scenes/test.zarr',
            'mask_path': os.path.join(BASE_DIR, 'scenes/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10E+10
        },
    
        'val_data_loader': {
            'key': 'scenes/validate_chopped_100/validate.zarr',
            'gt_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/gt.csv'),
            'mask_path': os.path.join(BASE_DIR, 'scenes/validate_chopped_100/mask.npz'),
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'samples_per_epoch': 10000
        },

    }

    if isinstance(cfg_model_params, dict):
        for k, v in cfg_model_params.items():
            cfg['model_params'][k] = v

    return cfg


if __name__ == '__main__':

    pass
