import torch
def create_dmdl_color_ess(albedo, specular, metallic, skybox_mask=None):
    dmdl_color = albedo + specular * 0.08 * (1-metallic)
    if skybox_mask is not None:
        dmdl_color = torch.ones_like(dmdl_color) * skybox_mask + dmdl_color * (1 - skybox_mask)
    return dmdl_color