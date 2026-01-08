from src.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatELoss(nn.Module):
    def __init__(self, args, kg, model):
        super(RotatELoss, self).__init__()
        self.args = args
        self.kg = kg
        self.model = model

    def forward(self, input, target, subsampling_weight=None, confidence=1.0, neg_ratio=10):
        p_score, n_score, p_indices, n_indices = self.split_pn_score(input, target, neg_ratio)
        
        # 提取正样本对应的confidence和subsampling_weight
        if isinstance(confidence, torch.Tensor) and confidence.numel() > 1:
            if confidence.dim() > 1:
                confidence = confidence.reshape(-1)
            # 使用正样本的索引提取confidence
            p_confidence = confidence[p_indices] if len(confidence) > max(p_indices) else confidence[:len(p_score)]
        else:
            p_confidence = confidence
        
        if subsampling_weight is not None:
            if subsampling_weight.dim() > 1:
                subsampling_weight = subsampling_weight.reshape(-1)
            p_subsampling_weight = subsampling_weight[p_indices] if len(subsampling_weight) > max(p_indices) else subsampling_weight[:len(p_score)]
        else:
            p_subsampling_weight = None
        
        p_score = F.logsigmoid(p_score) * p_confidence
        if p_subsampling_weight is not None:
            p_loss = -(p_subsampling_weight * p_score).sum(dim=-1)/p_subsampling_weight.sum()
        else:
            p_loss = -p_score.mean()
            
        if neg_ratio > 0:
            # instance-level
            # 负样本的confidence应该和正样本一样，因为每个正样本的负样本共享同一个confidence
            # n_score的形状是[batch_size, neg_ratio]，sum(dim=-1)后是[batch_size]
            # 所以n_confidence应该是[batch_size]，和p_confidence一样
            n_confidence = p_confidence  # 负样本使用对应正样本的confidence
            
            # 负样本的subsampling_weight也应该和正样本一样
            n_subsampling_weight = p_subsampling_weight
            
            n_score = (F.softmax(n_score, dim=-1).detach()
                              * F.logsigmoid(-n_score)).sum(dim=-1) * n_confidence
            
            if n_subsampling_weight is not None:
                n_loss = -(n_subsampling_weight * n_score).sum(dim=-1) / n_subsampling_weight.sum()
            else:
                n_loss = -n_score.mean()
            loss = (p_loss + n_loss)/2
        else:
            # schema-level
            loss = p_loss
        return loss


    def split_pn_score(self, score, label, neg_ratio):
        '''
        Get the scores of positive and negative facts
        :param score: scores of all facts
        :param label: positive facts: 1, negative facts: -1
        :return: p_score, n_score, p_indices, n_indices
        '''
        if neg_ratio <= 0:
            return score, 0, torch.arange(len(score), device=score.device), None
        
        # 获取正样本和负样本的索引
        p_mask = label > 0
        n_mask = label < 0
        
        if p_mask.dim() > 1:
            # label是二维的，需要flatten
            p_mask = p_mask.reshape(-1)
            n_mask = n_mask.reshape(-1)
        
        p_indices = torch.where(p_mask)[0]
        n_indices = torch.where(n_mask)[0]
        
        p_score = score[p_indices]
        n_score = score[n_indices].reshape(-1, neg_ratio)
        
        return p_score, n_score, p_indices, n_indices


class EntityAlignmentLoss(nn.Module):
    """
    实体对齐损失：使用余弦相似度
    """
    def __init__(self, args, kg):
        super(EntityAlignmentLoss, self).__init__()
        self.args = args
        self.kg = kg
        
    def forward(self, ent_embeddings, entity_pairs):
        """
        计算实体对齐的余弦相似度损失
        :param ent_embeddings: [num_ent, emb_dim] 所有实体的嵌入
        :param entity_pairs: [(e1, e2), ...] 对齐实体对列表
        :return: 对齐损失（1 - 余弦相似度的均值）
        """
        if len(entity_pairs) == 0:
            return torch.tensor(0.0, device=ent_embeddings.device, requires_grad=True)
        
        pairs = torch.LongTensor(entity_pairs).to(ent_embeddings.device)
        e1_emb = ent_embeddings[pairs[:, 0]]  # [num_pairs, emb_dim]
        e2_emb = ent_embeddings[pairs[:, 1]]  # [num_pairs, emb_dim]
        
        # 归一化
        e1_emb_norm = F.normalize(e1_emb, p=2, dim=1)
        e2_emb_norm = F.normalize(e2_emb, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_sim = (e1_emb_norm * e2_emb_norm).sum(dim=1)  # [num_pairs]
        
        # 损失：1 - 余弦相似度（我们希望对齐的实体相似度接近1）
        align_loss = (1 - cosine_sim).mean()
        
        return align_loss


class TripleInfoNCELoss(nn.Module):
    """
    基于扩展三元组的 InfoNCE 损失
    - 查询：h+r (head + relation)
    - 正样本：t (正确的尾实体)
    - 负样本：批次内其他样本的 t' (不正确的尾实体)
    """
    def __init__(self, args, kg, temperature=0.07):
        super(TripleInfoNCELoss, self).__init__()
        self.args = args
        self.kg = kg
        self.temperature = temperature
    
    def compute_query_emb(self, h_emb, r_emb):
        """
        计算查询表示 h+r
        :param h_emb: [batch_size, emb_dim] 头实体嵌入
        :param r_emb: [batch_size, emb_dim] 关系嵌入
        :return: [batch_size, emb_dim] 查询表示
        """
        # 对于 TransE: h+r
        query_emb = h_emb + r_emb
        return F.normalize(query_emb, p=2, dim=1)
    
    def get_in_batch_negatives(self, t_emb, batch_size):
        """
        获取批次内负样本（其他样本的尾实体）
        :param t_emb: [batch_size, emb_dim] 尾实体嵌入
        :param batch_size: 批次大小
        :return: [batch_size, batch_size-1, emb_dim] 批次内负样本嵌入
        """
        # 对于每个样本，同一批次中其他样本的尾实体就是负样本
        # t_emb: [batch_size, emb_dim]
        # 扩展为 [batch_size, batch_size, emb_dim]
        in_batch_negs = t_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 移除对角线（正样本自己）
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=t_emb.device)
        in_batch_negs = in_batch_negs[mask].reshape(batch_size, batch_size - 1, -1)
        
        return in_batch_negs
    
    def forward(self, h_emb, r_emb, t_emb):
        """
        计算三元组 InfoNCE 损失
        :param h_emb: [batch_size, emb_dim] 头实体嵌入
        :param r_emb: [batch_size, emb_dim] 关系嵌入
        :param t_emb: [batch_size, emb_dim] 尾实体嵌入（正样本）
        :return: InfoNCE 损失
        """
        batch_size = h_emb.size(0)
        
        # InfoNCE需要至少2个样本才能有批次内负样本
        if batch_size < 2:
            return torch.tensor(0.0, device=h_emb.device, requires_grad=True)
        
        # 计算查询表示
        query_emb = self.compute_query_emb(h_emb, r_emb)  # [batch_size, emb_dim]
        
        # 归一化正样本（尾实体）
        positive_emb = F.normalize(t_emb, p=2, dim=1)  # [batch_size, emb_dim]
        
        # 计算正样本相似度
        pos_sim = (query_emb * positive_emb).sum(dim=1, keepdim=True) / self.temperature  # [batch_size, 1]
        
        # 获取批次内负样本（其他样本的尾实体）
        in_batch_negs = self.get_in_batch_negatives(t_emb, batch_size)  # [batch_size, batch_size-1, emb_dim]
        in_batch_negs_norm = F.normalize(in_batch_negs, p=2, dim=2)
        
        # 计算负样本相似度
        query_emb_expanded = query_emb.unsqueeze(1)  # [batch_size, 1, emb_dim]
        neg_sim = torch.bmm(query_emb_expanded, in_batch_negs_norm.transpose(1, 2)).squeeze(1) / self.temperature
        # [batch_size, batch_size-1]
        
        # 拼接正负样本相似度
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, batch_size]
        
        # 标签：正样本在位置 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # 交叉熵损失（等价于 InfoNCE）
        loss = F.cross_entropy(logits, labels)
        
        return loss



