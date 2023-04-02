import numpy as np
import ssim_loss

def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out']- gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}
            
def image_ssim(mask, model_output, gt, height, width, channel=3):
    alpha = 0.7
    output = model_output['model_out'].squeeze().view(-1, height, width, channel).permute(0, 3, 1, 2)      # Shape: [5, channel, height, width]
    print(f"*******gt['img'] shape: {gt['img'].shape}*******")
    target = gt['img'].squeeze().view(-1, height, width, channel).permute(0, 3, 1, 2)
    
    if mask is None:
        total_loss = 0
        for i in range(output.shape[0]):
            output_frame = output[i:i+1, :, :, :].squeeze()
            target_frame = target[i:i+1, :, :, :].squeeze()
            total_loss += alpha * (output_frame - target_frame).abs().mean() \
                + (1-alpha) * (1 - ssim_loss.ssim(output_frame, target_frame))
                
        return {'img_loss': total_loss/5}
    else:
        return {'img_loss': alpha * (mask * abs(output - target)).mean() \
                + (1-alpha) * ssim_loss.ssim(output, target)}