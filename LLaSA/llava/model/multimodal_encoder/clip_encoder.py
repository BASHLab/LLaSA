import torch
import torch.nn as nn
from copy import deepcopy
from .models import LIMUBertModel4Pretrain, MinimalConfig
from .utils_new import get_device \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer #no_need
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch') #no_need

        self.hidden_size = 72  # TODO: Remove this after debugging
        self.device = torch.device("cuda")
        self.dtype = torch.float16  # TODO: Investigate how to properly retrieve
        # TODO: Currently dtype needs to be manually changed to float16 during inference
        # TODO: And needs to be changed to bfloat16 during training
        
        if not delay_load:
            # print("JUST LOADING MODEL")
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = MinimalConfig()
            # self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        mode = "base"
        args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
        data, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
        # pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
        model = LIMUBertModel4Pretrain(model_cfg)
        model.load_state_dict(torch.load("../model_4.pt", map_location=self.device))
        # model = model.to(device=self.device, dtype=self.dtype)
        self.image_processor = torch.nn.Identity()
        self.vision_tower = nn.Sequential(list(model.children())[0])
        self.vision_tower = self.vision_tower.to(device=self.device, dtype=self.dtype)
        # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name) ## TODO: Write an Indentity function to return the input
        # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map) ## TODO: load limubert (only encoder)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                # image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype))
                image_features.append(image_feature)
        else:
            # image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # image_features = self.feature_select(image_forward_outs).to(images.dtype) ## TODO: not required
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    # @property
    # def hidden_size(self):
    #     return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
