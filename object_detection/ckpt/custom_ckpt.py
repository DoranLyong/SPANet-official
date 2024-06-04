""" 
SpectralFilter(nn.Module) 의 사이즈가 
- ImageNet 224 
- ADE20K 512 
로 사이즈가 달라 이를 맞춰주기 위해 삭제. 

근거: 
- 학습하는 파라미터는 아니기 때문에 삭제해도 됨. 
"""
import torch 

name = 'spanet_s24-224_k7_r1100'
ckpt = torch.load(f'./model_best.pth.tar')


print(ckpt['arch'])
print(ckpt['epoch'])
print(ckpt['metric'])
print(type(ckpt))

for key in list(ckpt['state_dict'].keys()):
    if '.filter' in key:
        del ckpt['state_dict'][key]
        print(f'{key}: deleted')

# -- save -- # 
torch.save(ckpt, f'./mmseg_init.pth.tar')
