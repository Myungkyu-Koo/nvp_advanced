import numpy as np
import ssim_loss

def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out']- gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}
            
def image_ssim(mask, model_output, gt, height, width, channel=3):
    alpha = 0.7
    output = model_output['model_out'].view([-1, channel, height, width])
    target = gt['img'].view([-1, channel, height, width])
    
    print(f"*******height: {height}, width: {width}, channel: {channel}*******")
    print(f"*******output shape: {output.shape}*******")
    print(f"*******target shape: {target.shape}*******")
    
    if mask is None:
        total_loss = 0
        for i in range(output.shape[0]):
            output_frame = output[i:i+1, :, :, :].squeeze()
            print(f"*******output_frame shape: {output_frame.shape}*******")
            target_frame = target[i:i+1, :, :, :].squeeze()
            print(f"*******target_frame shape: {target_frame.shape}*******")
            total_loss += alpha * (output_frame - target_frame).abs().mean() \
                + (1-alpha) * ssim_loss.ssim(output_frame, target_frame)
                
        return {'img_loss': total_loss/5}
    else:
        return {'img_loss': alpha * (mask * abs(output - target)).mean() \
                + (1-alpha) * ssim_loss.ssim(output, target)}