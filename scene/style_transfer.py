import torch
import torch.nn as nn
from scene.gaussian_conv import GaussianConv
from utils.loss_utils import calc_mean_std
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()
        # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128,64,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64,matrixSize,3,1,1))
        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)


class MulLayer(nn.Module):
    def __init__(self, matrixSize=32, adain=True, use_local_align=True):
        super(MulLayer,self).__init__()
        self.adain = adain
        self.use_local_align = use_local_align
        if adain:
            return

        self.snet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, matrixSize)
        )
        self.unzip = nn.Sequential(
            nn.Linear(matrixSize, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )

    def compute_local_statistics(self, features, kernel_size=3):
        """计算局部特征统计量"""
        padding = kernel_size // 2
        mean = F.avg_pool2d(features, kernel_size, stride=1, padding=padding)
        var = F.avg_pool2d(features ** 2, kernel_size, stride=1, padding=padding) - mean ** 2
        std = torch.sqrt(var + 1e-6)
        return mean, std

    def compute_local_alignment(self, content_features, style_features, temperature=0.1):
        """计算局部对齐矩阵"""
        B, C, H, W = style_features.shape
        
        # 将特征图展平为向量
        content_flat = content_features.view(C, -1)  # [C, N]
        style_flat = style_features.view(B, C, -1)   # [B, C, H*W]
        
        # 计算余弦相似度
        content_norm = F.normalize(content_flat, dim=0)  # [C, N]
        style_norm = F.normalize(style_flat, dim=1)      # [B, C, H*W]
        similarity = torch.mm(content_norm.t(), style_norm.squeeze(0))  # [N, H*W]
        
        # 归一化对齐矩阵
        alignment_matrix = F.softmax(similarity / temperature, dim=-1)  # [N, H*W]
        
        return alignment_matrix

    def apply_local_style_transfer(self, content_features, style_features, alignment_matrix):
        """应用局部风格迁移"""
        B, C, H, W = style_features.shape
        
        # 计算局部统计量
        content_mean, content_std = self.compute_local_statistics(content_features.unsqueeze(0))
        style_mean, style_std = self.compute_local_statistics(style_features)
        
        # 将特征图展平为向量
        content_flat = content_features.view(C, -1)  # [C, N]
        content_mean_flat = content_mean.squeeze(0).view(C, -1)  # [C, N]
        content_std_flat = content_std.squeeze(0).view(C, -1)    # [C, N]
        style_mean_flat = style_mean.view(B, C, -1)      # [B, C, H*W]
        style_std_flat = style_std.view(B, C, -1)        # [B, C, H*W]
        
        # 计算局部风格迁移
        aligned_style_mean = torch.mm(style_mean_flat.squeeze(0), alignment_matrix.t())  # [C, N]
        aligned_style_std = torch.mm(style_std_flat.squeeze(0), alignment_matrix.t())    # [C, N]
        
        # 应用局部风格迁移
        stylized_flat = aligned_style_std * (content_flat - content_mean_flat) / (content_std_flat + 1e-6) + aligned_style_mean
        
        return stylized_flat

    def forward(self, cF, sF, trans=True, use_sgm=False, semantic_mask=None, iter=0):
        '''
        input:
            point cloud features: [N, C]
            style image features: [1, C, H, W]
            D: matrixSize
            use_sgm: bool, whether to use semantic guidance module
            semantic_mask: [H, W] or [H*W] binary mask for foreground/background
            iter: int, current iteration number
        '''
        if self.adain:
            cF = cF.T # [C, N]
            style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
            content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

            style_mean = style_mean.squeeze(0)
            style_std = style_std.squeeze(0)
            content_mean = content_mean.squeeze(0)
            content_std = content_std.squeeze(0)

            cF = (cF - content_mean) / content_std
            cF = cF * style_std + style_mean
            return cF.T
      
        assert cF.size(1) == sF.size(1), 'cF and sF must have the same channel size'
        assert sF.size(0) == 1, 'sF must have batch size 1'
        N, C = cF.size()
        B, C, H, W = sF.size()

        # normalize point cloud features
        cF = cF.T # [C, N]
        style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
        content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

        content_mean = content_mean.squeeze(0)
        content_std = content_std.squeeze(0)

        cF = (cF - content_mean) / content_std # [C, N]

        if self.use_local_align:
            # 计算局部对齐矩阵
            alignment_matrix = self.compute_local_alignment(cF, sF)
            
            # 应用语义引导模块 (SGM)
            if use_sgm and semantic_mask is not None and iter > 50000 and torch.rand(1) < 0.3:
                # 处理掩码：先阈值化，再展平为1D
                # 无论输入是 [H,W]、[1,H,W] 还是 [B,1,H,W] 都能正确处理
                binary_mask = (semantic_mask > 0.5).reshape(-1)
                
                # 生成每个像素的权重，确保类型和设备一致性
                w_per_pixel = torch.where(
                    binary_mask,
                    torch.full((binary_mask.shape[0],), 1.5, 
                             dtype=alignment_matrix.dtype, 
                             device=alignment_matrix.device),
                    torch.full((binary_mask.shape[0],), 0.8, 
                             dtype=alignment_matrix.dtype, 
                             device=alignment_matrix.device)
                )
                
                # 应用权重并重新归一化
                alignment_matrix *= w_per_pixel.unsqueeze(0)  # [N, H*W]
                alignment_matrix = F.softmax(alignment_matrix / 0.1, dim=-1)
            
            # 应用局部风格迁移
            out = self.apply_local_style_transfer(cF, sF, alignment_matrix)
            return out.T  # [N, C]

        # compress point cloud features
        compress_content = self.compress(cF.T).T # [D, N]

        # normalize style image features
        sF = sF.view(B,C,-1)
        sF = (sF - style_mean) / style_std  # [1, C, H*W]

        if(trans):
            # get style transformation matrix
            sMatrix = self.snet(sF.reshape(B,C,H,W)) # [B=1, D*D]
            sMatrix = sMatrix.view(self.matrixSize,self.matrixSize) # [D, D]

            transfeature = torch.mm(sMatrix, compress_content).T # [N, D]
            out = self.unzip(transfeature).T # [C, N]

            style_mean = style_mean.squeeze(0) # [C, 1]
            style_std = style_std.squeeze(0) # [C, 1]

            out = out * style_std + style_mean
            return out.T # [N, C]
        else:
            out = self.unzip(compress_content.T) # [N, C]
            out = out * content_std + content_mean
            return out

def local_align(content_features, style_features, K=5, temperature=0.1):
    """对每个内容特征，找到风格特征中最近的K个邻居，做加权平均
    
    Args:
        content_features: [N, C] 内容特征
        style_features: [M, C] 风格特征
        K: int, 邻居数量
        temperature: float, 温度参数，控制软度
        
    Returns:
        [N, C] 对齐后的特征
    """
    N, C = content_features.shape
    M = style_features.shape[0]
    
    # 计算余弦相似度
    content_norm = F.normalize(content_features, dim=1)  # [N, C]
    style_norm = F.normalize(style_features, dim=1)      # [M, C]
    similarity = torch.mm(content_norm, style_norm.t())  # [N, M]
    
    # 对每个内容特征找到K个最近邻
    topk_values, topk_indices = torch.topk(similarity, K, dim=1)  # [N, K]
    
    # 计算注意力权重
    attention_weights = F.softmax(topk_values / temperature, dim=1)  # [N, K]
    
    # 获取对应的风格特征
    aligned_features = []
    for i in range(N):
        # 获取当前内容特征的K个最近邻
        neighbors = style_features[topk_indices[i]]  # [K, C]
        # 加权平均
        weighted_sum = torch.sum(attention_weights[i].unsqueeze(1) * neighbors, dim=0)  # [C]
        aligned_features.append(weighted_sum)
    
    return torch.stack(aligned_features, dim=0)  # [N, C]
