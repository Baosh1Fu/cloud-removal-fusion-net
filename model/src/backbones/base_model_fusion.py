
from src.backbones.base_model import BaseModel, S2_BANDS
from src.loss_fusion import FusionCloudLoss

class FusionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.criterion = FusionCloudLoss(config)

    def forward(self):
        self.net_out = self.netG(self.real_A, batch_positions=self.dates)
        self.out_un  = {k: self.net_out[f"{k}_un"]   for k in ("delta","gamma","alpha","beta")}
        self.out_mamba = {k: self.net_out[f"{k}_ma"]   for k in ("delta","gamma","alpha","beta")}
        self.out_fused  = {k: self.net_out[f"{k}_fuse"] for k in ("delta","gamma","alpha","beta")}
        self.y_img = self.real_B  # Tensor[B, S2_BANDS, H, W]
        self.fake_B = self.net_out['delta_fuse']
    def get_loss_G(self):
        total_loss, aux = self.criterion(
            self.out_un,
            self.out_mamba,
            self.out_fused,
            self.y_img,
            #self.masks
        )
        self.loss_G = total_loss
        self.aux_info = aux
        if self.log_vars is None:
            self.log_vars = {}
        self.log_vars.update(aux)