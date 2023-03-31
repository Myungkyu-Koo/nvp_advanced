from skimage.metrics import structural_similarity as ssim

def image_mse(mask, model_output, gt, loss_type):
    if loss_type == 'L2':
        if mask is None:
            return {'img_loss': ((model_output['model_out']- gt['img']) ** 2).mean()}
        else:
            return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        alpha = 0.7
        if mask is None:
            return {'img_loss': alpha * abs(model_output['model_out']- gt['img']).mean() + (1-alpha) * ssim(model_output['model_out'], gt['img'], multichannel=True)}
        else:
            return {'img_loss': alpha * (mask * abs(model_output['model_out']- gt['img'])).mean() + (1-alpha) * ssim(model_output['model_out'], gt['img'], multichannel=True)}