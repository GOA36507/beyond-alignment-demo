from ..data_load.data_loader import *
from torch.utils.data import DataLoader
import math
import torch
import torch.nn.functional as F
from src.model.loss_func import EntityAlignmentLoss, TripleInfoNCELoss, TripleInfoNCELoss


class TrainEvalBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg

    def _create_dataset(self):
        pass


'''Cross-Link Prediction'''
class CrossLinkPredictionTrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        '''prepare data'''
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.head_dataset = CrossLinkPredictionTrainDatasetMarginLoss(self.args, self.kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionTrainDatasetMarginLoss(self.args, self.kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      batch_sampler=self.head_sampler,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(self.args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        # [RPG 已禁用] RPG 训练处理器
        # if self.args.RPG:
        #     self.RPG_train_processor = RPGTrainBatchProcessor(self.args, self.kg)
        
        # 初始化实体对齐损失函数
        self.alignment_loss_fn = EntityAlignmentLoss(self.args, self.kg)
        
        # 初始化三元组 InfoNCE 损失函数
        if getattr(self.args, 'use_triple_infonce', False):
            infonce_temperature = getattr(self.args, 'triple_infonce_temperature', 0.07)
            self.triple_infonce_loss_fn = TripleInfoNCELoss(self.args, self.kg, temperature=infonce_temperature)
        else:
            self.triple_infonce_loss_fn = None

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        
        # 获取当前epoch（需要在前面定义，因为后面会用到）
        current_epoch = getattr(self.args, 'epoch', 0)
        
        # 在每个epoch开始时，先计算伪对齐对（如果使用伪对齐）
        if self.args.use_pseudo_alignment:
            self.args.logger.info(f'[DEBUG] Epoch {current_epoch}: Starting compute_pseudo_alignment_pairs')
            self.compute_pseudo_alignment_pairs(model)
            num_pseudo_pairs = len(self.kg.pseudo_ent2align) // 2 if hasattr(self.kg, 'pseudo_ent2align') else 0
            self.args.logger.info(f'[DEBUG] Epoch {current_epoch}: Found {num_pseudo_pairs} pseudo alignment pairs')
            # 如果有伪对齐对，添加伪扩展三元组到训练数据中
            if hasattr(self.kg, 'pseudo_ent2align') and len(self.kg.pseudo_ent2align) > 0:
                replace_ratio = 1.0  # 默认使用所有伪实体对
                self.args.logger.info(f'[DEBUG] Epoch {current_epoch}: Starting add_facts_using_pseudo_entities (replace_ratio={replace_ratio})')
                num_added = self.add_facts_using_pseudo_entities(model, replace_ratio)
                self.args.logger.info(f'[DEBUG] Epoch {current_epoch}: Added {num_added} pseudo expand triples')
        else:
            self.args.logger.info(f'[DEBUG] Epoch {current_epoch}: use_pseudo_alignment is False, skipping compute_pseudo_alignment_pairs')

        # 构建图间三元组结构（图结构是固定的，只会在第一次构建，后续使用缓存）
        use_inter_triples = getattr(self.args, 'use_inter_triples', False)
        
        # 用于保存图结构统计信息
        graph_stats = None
        
        # 如果使用图间三元组，在第一个epoch构建图结构；否则只在需要统计信息时构建
        # 由于图结构是固定的（只基于已知对齐对），后续epoch会直接使用缓存
        should_compute_stats = not use_inter_triples or (current_epoch == 0)
        
        if should_compute_stats:
            # 传入训练数据集，确保图结构使用与训练数据一致的扩展三元组
            graph_stats = self.kg.build_inter_graph_structure(
                train_datasets=(self.head_dataset, self.tail_dataset)
            )
            if use_inter_triples and current_epoch == 0:
                self.args.logger.info(
                    f'Epoch {current_epoch}: Graph structure built - '
                    f'Original: {graph_stats["num_original_triples"]}, '
                    f'Known Expand: {graph_stats["num_known_expand_triples"]}'
                )
        
        # 保存图结构统计信息到实例变量，以便在返回时使用
        if graph_stats is not None:
            self._last_graph_stats = graph_stats

        head_data_loader = iter(self.head_data_loader)
        tail_data_loader = iter(self.tail_data_loader)
        success = False
        total_loss = 0.0
        # 用于记录每个epoch的对齐对数量（记录平均值）
        num_known_pairs_total = 0
        num_pseudo_pairs_total = 0
        num_batches_with_known_align = 0
        num_batches_with_pseudo_align = 0
        # 用于统计三元组数量
        num_original_triples = 0
        num_known_expand_triples = 0
        num_pseudo_expand_triples = 0
        while not success:
            total_loss = 0.0
            count_num = 0
            count_loss = None  # 初始化为None，第一次使用时创建
            count_batch = 0
            # 重置对齐对数量累计（避免重试时重复累计）
            num_known_pairs_total = 0
            num_pseudo_pairs_total = 0
            num_batches_with_known_align = 0
            num_batches_with_pseudo_align = 0
            # 重置三元组统计
            num_original_triples = 0
            num_known_expand_triples = 0
            num_pseudo_expand_triples = 0
            try:
                for idx_b, batch in enumerate(tqdm(range(2 * len(self.head_data_loader)))):
                    '''get loss'''
                    if idx_b % 2 == 0:
                        try:
                            iter_ = next(head_data_loader)
                        except:
                            continue
                        if iter_ is None:
                            continue
                        bh, br, bt, by, bc, mode, subsampling_weight, RPG_facts, is_expand, is_ea = iter_
                        mode = 'head-batch'
                    else:
                        try:
                            iter_ = next(tail_data_loader)
                        except:
                            continue
                        if iter_ is None:
                            continue
                        bh, br, bt, by, bc, mode, subsampling_weight, RPG_facts, is_expand, is_ea = iter_
                        mode = 'tail-batch'
                    current_samples_num = subsampling_weight.size(0)
                    bh = bh.to(self.args.device)
                    br = br.to(self.args.device)
                    bt = bt.to(self.args.device)
                    by = by.to(self.args.device)
                    bc = bc.to(self.args.device)
                    subsampling_weight = subsampling_weight.to(self.args.device)
                    if count_loss is None:
                        optimizer.zero_grad()
                        count_loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)

                    jobs = {
                        'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": bh}, 'mode': mode},
                        'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": br}, 'mode': mode},
                        'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": bt}, 'mode': mode},
                    }
                    pred = model.forward(jobs=jobs, stage='train', mode=mode, margin=self.args.margin)
                    
                    # 分离伪三元组损失和原始+已知扩展三元组损失
                    if is_expand is not None and is_ea is not None:
                        # 确保is_expand和is_ea是tensor格式，并移到正确的设备上
                        if not isinstance(is_expand, torch.Tensor):
                            is_expand = torch.tensor(is_expand, device=bh.device, dtype=torch.bool)
                        else:
                            is_expand = is_expand.to(bh.device).bool()
                        
                        if not isinstance(is_ea, torch.Tensor):
                            is_ea = torch.tensor(is_ea, device=bh.device, dtype=torch.bool)
                        else:
                            is_ea = is_ea.to(bh.device).bool()
                        
                        # 确保is_expand和is_ea的长度与by匹配（by包含正负样本）
                        by_flat = by.reshape(-1) if by.dim() > 1 else by
                        by_len = len(by_flat)
                        
                        if len(is_expand) != by_len:
                            min_len = min(len(is_expand), by_len)
                            is_expand = is_expand[:min_len]
                            is_ea = is_ea[:min_len]
                            by_flat = by_flat[:min_len]
                        
                        # 分离伪三元组mask（包括正负样本）
                        # 伪三元组：is_expand=True 且 is_ea=True
                        pseudo_mask = is_expand & is_ea
                        # 原始+已知扩展三元组：非伪三元组
                        original_known_mask = ~pseudo_mask
                        
                        # 只统计正样本（label > 0）
                        positive_mask = (by_flat > 0).bool()
                        
                        if len(is_expand) == len(positive_mask):
                            # 只统计正样本（label > 0）
                            # 原始三元组：is_expand=False 且为正样本
                            original_mask = (~is_expand) & positive_mask
                            num_original_triples += original_mask.sum().item()
                            
                            # 基于已知实体扩展的三元组：is_expand=True 且 is_ea=False 且为正样本
                            known_expand_mask = is_expand & (~is_ea) & positive_mask
                            num_known_expand_triples += known_expand_mask.sum().item()
                            
                            # 基于伪实体扩展的三元组：is_expand=True 且 is_ea=True 且为正样本
                            pseudo_expand_mask = is_expand & is_ea & positive_mask
                            num_pseudo_expand_triples += pseudo_expand_mask.sum().item()
                            
                            # 计算原始+已知扩展三元组损失
                            if original_known_mask.any():
                                original_known_bh = bh[original_known_mask]
                                original_known_br = br[original_known_mask]
                                original_known_bt = bt[original_known_mask]
                                original_known_by = by[original_known_mask]
                                original_known_bc = bc[original_known_mask]
                                original_known_subsampling_weight = subsampling_weight[original_known_mask]
                                original_known_pred = pred[original_known_mask]
                                
                                original_known_loss = model.loss(original_known_pred, original_known_by, original_known_subsampling_weight, original_known_bc, neg_ratio=self.args.neg_ratio)
                                if isinstance(original_known_loss, torch.Tensor) and original_known_loss.dim() > 0:
                                    original_known_loss = original_known_loss.mean()
                            else:
                                original_known_loss = torch.tensor(0.0, device=bh.device, requires_grad=True)
                            
                            # 根据use_pseudo_triple_loss决定是否计算伪三元组损失
                            if self.args.use_pseudo_triple_loss:
                                # 使用分离的伪三元组损失
                                pseudo_triple_weight = getattr(self.args, 'pseudo_triple_weight', 1.0)
                                
                                if pseudo_mask.any():
                                    # 计算伪三元组损失
                                    pseudo_bh = bh[pseudo_mask]
                                    pseudo_br = br[pseudo_mask]
                                    pseudo_bt = bt[pseudo_mask]
                                    pseudo_by = by[pseudo_mask]
                                    pseudo_bc = bc[pseudo_mask]
                                    pseudo_subsampling_weight = subsampling_weight[pseudo_mask]
                                    pseudo_pred = pred[pseudo_mask]
                                    
                                    pseudo_loss = model.loss(pseudo_pred, pseudo_by, pseudo_subsampling_weight, pseudo_bc, neg_ratio=self.args.neg_ratio)
                                    if isinstance(pseudo_loss, torch.Tensor) and pseudo_loss.dim() > 0:
                                        pseudo_loss = pseudo_loss.mean()
                                else:
                                    pseudo_loss = torch.tensor(0.0, device=bh.device, requires_grad=True)
                                
                                # 加权组合损失
                                batch_loss = original_known_loss + pseudo_triple_weight * pseudo_loss
                            else:
                                # 不使用伪三元组损失，只计算原始+已知扩展三元组损失
                                batch_loss = original_known_loss
                            
                            # 计算基于扩展三元组的 InfoNCE 损失
                            if getattr(self.args, 'use_triple_infonce', False) and self.triple_infonce_loss_fn is not None:
                                # 1. 已知对齐对的扩展三元组 InfoNCE 损失（保持不变）
                                known_expand_mask = is_expand & (~is_ea) & positive_mask
                                if known_expand_mask.any():
                                    known_infonce_loss = self.compute_triple_infonce_loss(
                                        model, bh, br, bt, known_expand_mask, mode
                                    )
                                    if isinstance(known_infonce_loss, torch.Tensor) and known_infonce_loss.dim() > 0:
                                        known_infonce_loss = known_infonce_loss.mean()
                                    triple_infonce_weight = getattr(self.args, 'triple_infonce_weight', 0.1)
                                    batch_loss = batch_loss + triple_infonce_weight * known_infonce_loss
                                
                                # 2. 伪对齐对的扩展三元组 InfoNCE 损失（新增，可控制）
                                use_pseudo_triple_infonce = getattr(self.args, 'use_pseudo_triple_infonce', False)
                                if use_pseudo_triple_infonce:
                                    pseudo_expand_mask = is_expand & is_ea & positive_mask
                                    if pseudo_expand_mask.any():
                                        pseudo_infonce_loss = self.compute_triple_infonce_loss(
                                            model, bh, br, bt, pseudo_expand_mask, mode
                                        )
                                        if isinstance(pseudo_infonce_loss, torch.Tensor) and pseudo_infonce_loss.dim() > 0:
                                            pseudo_infonce_loss = pseudo_infonce_loss.mean()
                                        pseudo_triple_infonce_weight = getattr(self.args, 'pseudo_triple_infonce_weight', 0.1)
                                        batch_loss = batch_loss + pseudo_triple_infonce_weight * pseudo_infonce_loss
                        else:
                            # 如果无法分离，使用原始方法
                            batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.args.neg_ratio)
                            if isinstance(batch_loss, torch.Tensor) and batch_loss.dim() > 0:
                                batch_loss = batch_loss.mean()
                    else:
                        # 如果没有is_expand和is_ea信息，使用原始方法
                        batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.args.neg_ratio)
                        if isinstance(batch_loss, torch.Tensor) and batch_loss.dim() > 0:
                            batch_loss = batch_loss.mean()
                    
                    # [RPG 已禁用] RPG triple 损失计算
                    # if self.args.use_RPG_triple and len(RPG_facts) > 0:
                    #     self.RPG_train_processor.set_RPG_edges(RPG_facts)
                    #     RPG_loss = self.RPG_train_processor.loss(model, optimizer)
                    #     batch_loss = self.args.alpha * RPG_loss + batch_loss
                    
                    # 添加实体对齐损失
                    if self.args.use_known_alignment and hasattr(self.kg, 'entity_pairs') and len(self.kg.entity_pairs) > 0:
                        known_align_loss, num_known_pairs = self.compute_known_alignment_loss(model)
                        batch_loss = batch_loss + self.args.known_align_weight * known_align_loss
                        # 累计已知对齐对数量
                        num_known_pairs_total += num_known_pairs
                        num_batches_with_known_align += 1
                    
                    # 添加伪对齐损失（使用epoch开始时计算的伪对齐对）
                    if self.args.use_pseudo_alignment:
                        pseudo_align_loss, num_pseudo_pairs = self.compute_pseudo_alignment_loss(model)
                        batch_loss = batch_loss + self.args.pseudo_align_weight * pseudo_align_loss
                        # 累计伪对齐对数量
                        num_pseudo_pairs_total += num_pseudo_pairs
                        num_batches_with_pseudo_align += 1

                    '''update'''
                    # 使用非原地操作避免对requires_grad=True的tensor进行原地操作
                    if count_loss is None:
                        count_loss = batch_loss * current_samples_num
                    else:
                        count_loss = count_loss + batch_loss * current_samples_num
                    count_num += current_samples_num
                    count_batch += 1
                    if count_num >= self.args.batch_size or self.args.scorer == 'RotatE':
                        count_loss = count_loss / count_num
                        count_loss.backward()
                        optimizer.step()
                        # 使编码器缓存失效（对于lookup_gat等使用缓存的编码器）
                        if hasattr(model.encoder, 'invalidate_cache'):
                            model.encoder.invalidate_cache()
                        total_loss += count_loss.item()
                        count_loss = None  # 重置为None
                        count_num = 0
                        count_batch = 0
                    '''post processing'''
                success = True
            # except:
            #     import sys
            #     e = sys.exc_info()[0]
            #     # 如果是内存超出的错误，减小batch_size，否则直接报错并退出
            #     if 'CUDA out of memory' in str(e):
            #         self.batch_size = self.batch_size // 2
            #         self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.batch_size)
            #         self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.batch_size)
            #         self.head_data_loader = DataLoader(self.head_dataset,
            #                                             batch_sampler=self.head_sampler,
            #                                             collate_fn=self.head_dataset.collate_fn,
            #                                             generator=torch.Generator().manual_seed(int(self.args.seed)),
            #                                             pin_memory=True)
            #         self.tail_data_loader = DataLoader(self.tail_dataset,
            #                                             batch_sampler=self.tail_sampler,
            #                                             collate_fn=self.tail_dataset.collate_fn,
            #                                             generator=torch.Generator().manual_seed(int(self.args.seed)),
            #                                             pin_memory=True)
            #         print('Batch size is too large, reduce to', self.batch_size)
            #     else:
            #         print('Error:', e)
            #         break
            except:
                import sys, traceback
                e = sys.exc_info()[1]
                tb = traceback.format_exc()
                # 如果是内存超出的错误，减小batch_size，否则打印完整异常并退出
                if 'CUDA out of memory' in str(e):
                    self.batch_size = self.batch_size // 2
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                        batch_sampler=self.head_sampler,
                                                        collate_fn=self.head_dataset.collate_fn,
                                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                        pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                        batch_sampler=self.tail_sampler,
                                                        collate_fn=self.tail_dataset.collate_fn,
                                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                        pin_memory=True)
                    print('Batch size is too large, reduce to', self.batch_size)
                else:
                    print('Error:', repr(e))
                    print(tb)
                    break

        # 计算平均对齐对数量
        avg_known_pairs = num_known_pairs_total // max(num_batches_with_known_align, 1) if num_batches_with_known_align > 0 else 0
        avg_pseudo_pairs = num_pseudo_pairs_total // max(num_batches_with_pseudo_align, 1) if num_batches_with_pseudo_align > 0 else 0
        
        # 构建三元组统计信息
        triple_stats = {
            'num_original': num_original_triples,
            'num_known_expand': num_known_expand_triples,
            'num_pseudo_expand': num_pseudo_expand_triples
        }
        
        # 添加图结构统计信息（无论是否使用图间三元组，都输出用于调试）
        use_inter_triples = getattr(self.args, 'use_inter_triples', False)
        if hasattr(self, '_last_graph_stats'):
            triple_stats['graph_original'] = self._last_graph_stats.get('num_original_triples', 0)
            triple_stats['graph_known_expand'] = self._last_graph_stats.get('num_known_expand_triples', 0)
            triple_stats['graph_pseudo_expand'] = self._last_graph_stats.get('num_pseudo_expand_triples', 0)
            
            # 添加调试信息：比较 Original_Triples 和 Graph_Original
            self.args.logger.info(
                f'[DEBUG] Epoch {getattr(self.args, "epoch", 0)}: Triple stats comparison - '
                f'Original_Triples={num_original_triples}, '
                f'Graph_Original={triple_stats["graph_original"]}, '
                f'Ratio={triple_stats["graph_original"]/num_original_triples if num_original_triples > 0 else 0:.2f}'
            )
        else:
            # 如果没有图结构统计信息，设置为0
            triple_stats['graph_original'] = 0
            triple_stats['graph_known_expand'] = 0
            triple_stats['graph_pseudo_expand'] = 0
        
        return total_loss, avg_known_pairs, avg_pseudo_pairs, triple_stats

    # [RPG 已禁用] 基于关系对齐添加训练集三元组的函数
    # def add_facts_using_relations(self, same, inverse):
    #     self.head_dataset.add_facts_using_relations(same, inverse)
    #     self.tail_dataset.add_facts_using_relations(same, inverse)
    #     self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
    #     self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
    #     self.head_data_loader = DataLoader(self.head_dataset,
    #                                        batch_sampler=self.head_sampler,
    #                                        collate_fn=self.head_dataset.collate_fn,
    #                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
    #                                        pin_memory=True)
    #     self.tail_data_loader = DataLoader(self.tail_dataset,
    #                                        batch_sampler=self.tail_sampler,
    #                                        collate_fn=self.tail_dataset.collate_fn,
    #                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
    #                                        pin_memory=True)
    
    def compute_known_alignment_loss(self, model):
        """
        计算已知实体对的对齐损失（余弦相似度）
        :param model: 模型
        :return: 对齐损失, 实体对数量
        """
        # 获取所有实体嵌入
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)
        
        # 获取已知实体对数量
        num_known_pairs = len(self.kg.entity_pairs) if hasattr(self.kg, 'entity_pairs') and len(self.kg.entity_pairs) > 0 else 0
        
        # 使用已知实体对计算对齐损失
        align_loss = self.alignment_loss_fn(ent_embeddings, self.kg.entity_pairs)
        
        return align_loss, num_known_pairs
    
    def compute_pseudo_alignment_pairs(self, model):
        """
        在每个epoch开始时计算伪对齐对（不计算损失）
        公式：
        - c_ei = S_ij - max(S_ij')  # 从 e_i 的角度
        - c_ej = S_ij - max(S_i'j)  # 从 e_j 的角度
        - c(e_i, e_j) = (c_ei + c_ej) / 2  # 平均置信度
        - 如果 c(e_i, e_j) > λ，则作为伪对齐对
        :param model: 模型
        :return: 无，结果存储在 self.kg.pseudo_ent2align 中
        """
        self.args.logger.info('[DEBUG] compute_pseudo_alignment_pairs: Starting computation')
        # 获取所有实体嵌入
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)
        ent_embeddings = F.normalize(ent_embeddings, p=2, dim=1)  # 归一化
        self.args.logger.info(f'[DEBUG] compute_pseudo_alignment_pairs: Got embeddings, shape={ent_embeddings.shape}')
        
        # 分离两个KG的实体
        num_ent_kg1 = self.kg.source[0].num_ent
        ent_emb_kg1 = ent_embeddings[:num_ent_kg1]  # [num_ent_kg1, emb_dim]
        ent_emb_kg2 = ent_embeddings[num_ent_kg1:]  # [num_ent_kg2, emb_dim]
        
        # 计算相似度矩阵 S_ij = ent_emb_kg1 @ ent_emb_kg2.T
        similarity_matrix = torch.matmul(ent_emb_kg1, ent_emb_kg2.t())  # [num_ent_kg1, num_ent_kg2]
        
        # 计算 c_ei = S_ij - max(S_ij')，其中 S_ij' 是除了 e_j 之外的其他实体在 G2 中的相似度
        # 对于每一行（每个 e_i），找到除了当前列之外的最大值
        # 使用向量化实现，避免双重循环
        num_ent_kg2 = ent_emb_kg2.size(0)
        # 对每一行，找到 top-2 最大值
        k = min(2, num_ent_kg2)
        top2_values, top2_indices = torch.topk(similarity_matrix, k=k, dim=1)  # [num_ent_kg1, k]
        
        # 向量化实现：对于每一行 i 和每一列 j
        # 如果 top2_indices[i, 0] == j，则用 top2_values[i, 1]（如果存在），否则用 top2_values[i, 0]
        # 创建列索引矩阵 [num_ent_kg1, num_ent_kg2]，每列都是 [0, 1, 2, ..., num_ent_kg2-1]
        col_indices = torch.arange(num_ent_kg2, device=similarity_matrix.device).unsqueeze(0).expand(num_ent_kg1, -1)  # [num_ent_kg1, num_ent_kg2]
        # 创建 mask：如果当前列 j 是最大值（top2_indices[i, 0] == j）
        max_col_mask = (top2_indices[:, 0:1] == col_indices)  # [num_ent_kg1, num_ent_kg2]
        
        # 如果 k > 1，当最大值是当前列时用第二大的值，否则用最大值
        if k > 1:
            max_other_kg2 = torch.where(max_col_mask, 
                                       top2_values[:, 1:2].expand(-1, num_ent_kg2),  # 用第二大的值
                                       top2_values[:, 0:1].expand(-1, num_ent_kg2))  # 用最大值
        else:
            # 如果只有一列，直接用最大值
            max_other_kg2 = top2_values[:, 0:1].expand(-1, num_ent_kg2)
        
        c_ei = similarity_matrix - max_other_kg2
        
        # 计算 c_ej = S_ij - max(S_i'j)，其中 S_i'j 是除了 e_i 之外的其他实体在 G1 中的相似度
        # 对于每一列（每个 e_j），找到除了当前行之外的最大值
        # 使用向量化实现
        k_col = min(2, num_ent_kg1)
        top2_values_col, top2_indices_col = torch.topk(similarity_matrix, k=k_col, dim=0)  # [k_col, num_ent_kg2]
        
        # 向量化实现：对于每一列 j 和每一行 i
        # 如果 top2_indices_col[0, j] == i，则用 top2_values_col[1, j]（如果存在），否则用 top2_values_col[0, j]
        # 创建行索引矩阵 [num_ent_kg1, num_ent_kg2]，每行都是 [0, 1, 2, ..., num_ent_kg1-1]
        row_indices = torch.arange(num_ent_kg1, device=similarity_matrix.device).unsqueeze(1).expand(-1, num_ent_kg2)  # [num_ent_kg1, num_ent_kg2]
        # 创建 mask：如果当前行 i 是最大值（top2_indices_col[0, :] == i）
        max_row_mask = (top2_indices_col[0:1, :] == row_indices)  # [num_ent_kg1, num_ent_kg2]
        
        # 如果 k_col > 1，当最大值是当前行时用第二大的值，否则用最大值
        if k_col > 1:
            max_other_kg1 = torch.where(max_row_mask,
                                       top2_values_col[1:2, :].expand(num_ent_kg1, -1),  # 用第二大的值
                                       top2_values_col[0:1, :].expand(num_ent_kg1, -1))  # 用最大值
        else:
            # 如果只有一行，直接用最大值
            max_other_kg1 = top2_values_col[0:1, :].expand(num_ent_kg1, -1)
        
        c_ej = similarity_matrix - max_other_kg1
        
        # 计算平均置信度 c(e_i, e_j) = (c_ei + c_ej) / 2
        confidence = (c_ei + c_ej) / 2.0  # [num_ent_kg1, num_ent_kg2]
        
        # 优化1: 排除已知对齐对中的实体（这些实体已经对齐，不再寻找伪对齐对）
        # 如果实体 i 或实体 j 已经在已知对齐对中，则排除该实体所在的行或列
        if hasattr(self.kg, 'entity_pairs') and len(self.kg.entity_pairs) > 0:
            # 收集所有在已知对齐对中的实体
            known_kg1_entities = set()  # kg1 中已对齐的实体
            known_kg2_entities = set()  # kg2 中已对齐的实体（需要减去偏移）
            
            for e1, e2 in self.kg.entity_pairs:
                # e1 在 kg1 中，e2 在 kg2 中（需要减去偏移）
                if e1 < num_ent_kg1 and e2 >= num_ent_kg1:
                    known_kg1_entities.add(e1)
                    known_kg2_entities.add(e2 - num_ent_kg1)  # 减去偏移得到 kg2 中的索引
                elif e2 < num_ent_kg1 and e1 >= num_ent_kg1:
                    known_kg1_entities.add(e2)
                    known_kg2_entities.add(e1 - num_ent_kg1)  # 减去偏移得到 kg2 中的索引
            
            # 排除已知对齐实体所在的行（kg1 中的实体）和列（kg2 中的实体）
            if known_kg1_entities:
                idx1 = torch.tensor(list(known_kg1_entities), device=confidence.device)
                confidence[idx1, :] = -1e9  # 比 -inf 更稳定一点

            if known_kg2_entities:
                idx2 = torch.tensor(list(known_kg2_entities), device=confidence.device)
                confidence[:, idx2] = -1e9

        
        # 找到置信度大于阈值 λ 的候选对齐对
        threshold = getattr(self.args, 'pseudo_align_threshold', 0.1)
        
        # 优化2: 每个实体只对应一个最佳匹配（一对一映射）
        # 使用贪心策略：从最高置信度开始选择，确保每个实体只出现一次
        pseudo_pairs = []
        used_kg1_entities = set()
        used_kg2_entities = set()
        
        # 将所有超过阈值的对齐对按置信度排序
        valid_mask = confidence > threshold
        if valid_mask.any():
            valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # [num_valid, 2]
            valid_confidences = confidence[valid_indices[:, 0], valid_indices[:, 1]]
            
            # 按置信度从高到低排序
            sorted_indices = torch.argsort(valid_confidences, descending=True)
            
            # 贪心选择：每个实体只能出现一次
            for idx in sorted_indices:
                i = valid_indices[idx, 0].item()  # kg1 中的实体索引
                j = valid_indices[idx, 1].item()  # kg2 中的实体索引
                
                # 如果两个实体都未被使用，则添加这个对齐对
                if i not in used_kg1_entities and j not in used_kg2_entities:
                    pseudo_pairs.append((i, j + num_ent_kg1))  # 加上 kg2 的偏移
                    used_kg1_entities.add(i)
                    used_kg2_entities.add(j)
        
        num_pseudo_pairs = len(pseudo_pairs)
        
        self.args.logger.info(
            f'[DEBUG] compute_pseudo_alignment_pairs: Found {num_pseudo_pairs} pseudo pairs '
            f'(threshold={threshold}, valid_mask_count={valid_mask.sum().item() if valid_mask.any() else 0})'
        )
        
        if num_pseudo_pairs == 0:
            # 清空伪实体对齐字典
            self.kg.pseudo_ent2align = dict()
            self.args.logger.info('[DEBUG] compute_pseudo_alignment_pairs: No pseudo pairs found, clearing pseudo_ent2align')
            return
        
        # 根据topk_ratio选择前K%的伪实体对
        topk_ratio = getattr(self.args, 'pseudo_expand_topk_ratio', 1.0)
        if topk_ratio < 1.0:
            # 按置信度排序（已经在sorted_indices中按置信度排序了）
            # 只取前topk_ratio比例的伪实体对
            num_select = max(1, int(num_pseudo_pairs * topk_ratio))
            pseudo_pairs = pseudo_pairs[:num_select]
            num_pseudo_pairs = len(pseudo_pairs)
        
        # 更新kg中的伪实体对齐字典
        self.kg.pseudo_ent2align = dict()
        for e1, e2 in pseudo_pairs:
            self.kg.pseudo_ent2align[e1] = e2
            self.kg.pseudo_ent2align[e2] = e1
        
        self.args.logger.info(
            f'[DEBUG] compute_pseudo_alignment_pairs: Updated pseudo_ent2align with {len(self.kg.pseudo_ent2align)//2} pairs'
        )
    
    def compute_pseudo_alignment_loss(self, model):
        """
        使用已计算的伪对齐对计算对齐损失
        :param model: 模型
        :return: 伪对齐损失, 伪对齐对数量
        """
        # 检查是否有伪对齐对
        if not hasattr(self.kg, 'pseudo_ent2align') or len(self.kg.pseudo_ent2align) == 0:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True), 0
        
        # 从pseudo_ent2align构建伪对齐对列表
        pseudo_pairs = []
        used_entities = set()
        for e1, e2 in self.kg.pseudo_ent2align.items():
            if e1 not in used_entities and e2 not in used_entities:
                pseudo_pairs.append((e1, e2))
                used_entities.add(e1)
                used_entities.add(e2)
        
        num_pseudo_pairs = len(pseudo_pairs)
        if num_pseudo_pairs == 0:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True), 0
        
        # 获取所有实体嵌入
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)
        
        # 计算伪对齐损失
        pseudo_align_loss = self.alignment_loss_fn(ent_embeddings, pseudo_pairs)
        
        return pseudo_align_loss, num_pseudo_pairs
    
    def compute_triple_infonce_loss(self, model, bh, br, bt, expand_mask, mode):
        """
        计算基于扩展三元组的 InfoNCE 损失
        :param model: 模型
        :param bh, br, bt: batch中的subject, relation, object
        :param expand_mask: 扩展三元组的mask（只对扩展三元组计算）
        :param mode: 'head-batch' 或 'tail-batch'
        :return: InfoNCE 损失
        """
        if not expand_mask.any() or self.triple_infonce_loss_fn is None:
            return torch.tensor(0.0, device=bh.device, requires_grad=True)
        
        # 提取扩展三元组（只取正样本）
        expand_bh = bh[expand_mask]
        expand_br = br[expand_mask]
        expand_bt = bt[expand_mask]
        
        # 获取实体和关系嵌入
        h_emb = model.encoder.embed_ent(expand_bh, scorer=model.decoder.scorer, mode=mode)
        r_emb = model.encoder.embed_rel(expand_br)
        t_emb = model.encoder.embed_ent(expand_bt, scorer=model.decoder.scorer, mode=mode)
        
        # 计算 InfoNCE 损失
        infonce_loss = self.triple_infonce_loss_fn(h_emb, r_emb, t_emb)
        
        return infonce_loss
    
    def add_facts_using_pseudo_entities(self, model, replace_ratio=1.0):
        """
        基于伪实体对构造扩展三元组
        :param model: 模型，用于计算三元组得分
        :param replace_ratio: 实体替换的比例（0.0-1.0），控制部分实体替换
        :return: 添加的伪扩展三元组数量
        """
        num_added_head = self.head_dataset.add_facts_using_pseudo_entities(model, replace_ratio)
        num_added_tail = self.tail_dataset.add_facts_using_pseudo_entities(model, replace_ratio)
        
        # 重新创建sampler和dataloader
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                           batch_sampler=self.head_sampler,
                                           collate_fn=self.head_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        
        return num_added_head + num_added_tail


class CrossLinkPredictionValidBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = self.args.test_batch_size
        '''prepare data'''
        self.head_dataset = CrossLinkPredictionValidDataset(args, kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionValidDataset(args, kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.valid, self.args.test_batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.valid, self.args.test_batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      # shuffle=False,
                                      batch_sampler=self.head_sampler,
                                      # batch_size=self.batch_size,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           # shuffle=False,
                                           batch_sampler=self.tail_sampler,
                                           # batch_size=self.batch_size,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(args.seed)),
                                           pin_memory=True)

    def process_epoch(self, model):
        model.eval()

        '''start evaluation'''
        success = False
        while not success:
            try:
                num = 0
                results = dict()
                for data_loader in [self.head_data_loader, self.tail_data_loader]:
                    for batch in data_loader:
                        sub, rel, obj, label, mode = batch
                        sub = sub.to(self.args.device)
                        rel = rel.to(self.args.device)
                        obj = obj.to(self.args.device)
                        label = label.to(self.args.device)
                        num += len(sub)

                        '''link prediction'''
                        if mode == 'tail-batch':
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = obj
                        else:
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = sub
                        pred = model(jobs=jobs, mode=mode, margin=self.args.margin)



                        b_range = torch.arange(pred.size()[0], device=self.args.device)
                        target_pred = pred[b_range, target]
                        pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)

                        pred[b_range, target] = target_pred

                        '''rank all candidate entities'''
                        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, target]

                        '''get results'''
                        ranks = ranks.float()
                        results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                        results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                        results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)

                        for k in range(10):
                            results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get('hits{}'.format(k + 1), 0.0)
                success = True
            except:
                import sys
                e = sys.exc_info()[0]
                if 'CUDA out of memory' in str(e):
                    print('CUDA out of memory, try to reduce batch size by half')
                    self.batch_size = self.batch_size // 2
                    self.head_dataset = CrossLinkPredictionValidDataset(self.args, self.kg, mode='head-batch')
                    self.tail_dataset = CrossLinkPredictionValidDataset(self.args, self.kg, mode='tail-batch')
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.valid,
                                                             self.args.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.valid,
                                                             self.args.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.head_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.head_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.tail_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.tail_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                else:
                    print('Unexpected error:', e)
                    break


        count = float(results['count'])
        for key, val in results.items():
            if key != 'count':
                results[key] = round(val / count, 4)
        return results

    # [RPG 已禁用] 基于关系对齐添加验证集三元组的函数
    # def add_facts_using_relations(self, same, inverse):
    #     self.head_dataset.add_facts_using_relations(same, inverse)
    #     self.tail_dataset.add_facts_using_relations(same, inverse)
    #     self.head_data_loader = DataLoader(self.head_dataset,
    #                                        shuffle=False,
    #                                        batch_size=self.batch_size,
    #                                        collate_fn=self.head_dataset.collate_fn,
    #                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
    #                                        pin_memory=True)
    #     self.tail_data_loader = DataLoader(self.tail_dataset,
    #                                        shuffle=False,
    #                                        batch_size=self.batch_size,
    #                                        collate_fn=self.tail_dataset.collate_fn,
    #                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
    #                                        pin_memory=True)


class CrossLinkPredictionTestBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = self.args.test_batch_size
        '''prepare data'''
        self.head_dataset = CrossLinkPredictionTestDataset(args, kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionTestDataset(args, kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.test, self.args.test_batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.test, self.args.test_batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      # shuffle=False,
                                      batch_sampler=self.head_sampler,
                                      # batch_size=self.batch_size,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           # shuffle=False,
                                           batch_sampler=self.tail_sampler,
                                           # batch_size=self.batch_size,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(args.seed)),
                                           pin_memory=True)

    def process_epoch(self, model):
        def update_results(results_, ranks_):
            results_['count'] = torch.numel(ranks_) + results_.get('count', 0.0)
            results_['mr'] = torch.sum(ranks_).item() + results_.get('mr', 0.0)
            results_['mrr'] = torch.sum(1.0 / ranks_).item() + results_.get('mrr', 0.0)
            for k in range(10):
                results_['hits{}'.format(k + 1)] = torch.numel(ranks_[ranks_ <= (k + 1)]) + results_.get(
                    'hits{}'.format(k + 1), 0.0)
            return results_
        # def ave_results(results_):
        #     count = float(results_['count'])
        #     for key, val in results_.items():
        #         if key != 'count':
        #             results_[key] = round(val / count, 4)
        #     return results_

        def ave_results(results_):
        # 没有任何样本时，count 为 0，直接返回，避免除以 0
            count = float(results_.get('count', 0.0))
            if count == 0.0:
                # 确保需要的 key 至少存在（后面打印时不会 KeyError）
                for k in ['mr', 'mrr'] + ['hits{}'.format(i + 1) for i in range(10)]:
                    results_.setdefault(k, 0.0)
                return results_
            for key, val in results_.items():
                if key != 'count':
                    results_[key] = round(val / count, 4)
            return results_

        model.eval()
        success = False
        while not success:
            try:
                num = 0
                results_inner, results_cross, results_cross_relation, results_cross_entity = dict(), dict(), dict(), dict()
                '''start evaluation'''
                for data_loader in [self.head_data_loader, self.tail_data_loader]:
                    for batch in tqdm(data_loader):
                        sub, rel, obj, label, mode, types = batch
                        sub = sub.to(self.args.device)
                        rel = rel.to(self.args.device)
                        obj = obj.to(self.args.device)
                        label = label.to(self.args.device)
                        types = types.to(self.args.device)

                        num += len(sub)
                        '''link prediction'''
                        if mode == 'tail-batch':
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = obj
                        else:
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = sub
                        pred = model(jobs=jobs, mode=mode, margin=self.args.margin)

                        # pred = model(jobs=jobs, mode=mode, margin=self.args.margin)
                        b_range = torch.arange(pred.size()[0], device=self.args.device)

                        # # filter
                        target_pred = pred[b_range, target]
                        pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)

                        pred[b_range, target] = target_pred

                        '''rank all candidate entities'''
                        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, target]
                        '''get results'''
                        ranks = ranks.float()

                        inner_idx = (types == -1).reshape(-1)
                        cross_idx = (types == 1).reshape(-1)
                        cross_relation_idx = ((sub<self.kg.source[0].num_ent) & (obj<self.kg.source[0].num_ent) & (rel>=self.kg.source[0].num_rel)) \
                                             + ((sub>=self.kg.source[0].num_ent)&(obj>=self.kg.source[0].num_ent)&(rel<self.kg.source[0].num_rel))
                        cross_entity_idx = (~cross_relation_idx) & cross_idx

                        ranks_inner = ranks[inner_idx]
                        ranks_cross = ranks[cross_idx]
                        ranks_cross_relation = ranks[cross_relation_idx]
                        ranks_cross_entity = ranks[cross_entity_idx]

                        results_inner = update_results(results_inner, ranks_inner)
                        results_cross = update_results(results_cross, ranks_cross)
                        results_cross_relation = update_results(results_cross_relation, ranks_cross_relation)
                        results_cross_entity = update_results(results_cross_entity, ranks_cross_entity)

                success = True
            except Exception as e:
                # 如果是内存不足的问题，减小batch_size，否则直接报错
                if 'CUDA out of memory' in str(e):
                    self.args.logger.info('Error: {}'.format(e))
                    self.args.logger.info('Retry...')
                    self.batch_size = int(self.batch_size / 2)
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.test,
                                                             self.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.test,
                                                             self.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.head_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.head_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.tail_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.tail_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                else:
                    self.args.logger.info('Error: {}'.format(e))
                    break



        results_inner = ave_results(results_inner)
        results_cross = ave_results(results_cross)
        results_cross_relation = ave_results(results_cross_relation)
        results_cross_entity = ave_results(results_cross_entity)

        self.args.logger.info('cross_relation_results:{}'.format(results_cross_relation))
        self.args.logger.info('cross__entity_results:{}'.format(results_cross_entity))
        self.args.logger.info('inner_results:{}'.format(results_inner))
        self.args.logger.info('cross_results:{}'.format(results_cross))

        rr = results_inner
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        rr = results_cross
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        rr = results_cross_relation
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        rr = results_cross_entity
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        return results_cross


class RPG_filler():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        '''prepare data'''

    def process_epoch(self, model):
        model.eval()
        '''start'''
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)

        final_ent_embeddings = ent_embeddings
        A = final_ent_embeddings[:self.kg.source[0].num_ent]
        B = final_ent_embeddings[self.kg.source[0].num_ent:]

        A = torch.nn.functional.normalize(A, dim=1)
        B = torch.nn.functional.normalize(B, dim=1)

        r2s_A, r2s_B = {r: s for r, s in self.kg.r2s.items() if r < self.kg.source[0].num_rel}, {r: s for r, s in
                                                                                                 self.kg.r2s.items() if
                                                                                                 r >= self.kg.source[
                                                                                                     0].num_rel}

        A2B_topk, B2A_topk = find_topk_similar_entities_cross(A, B, self.args.topk)
        A2B_cover, B2A_cover = compute_coverage(r2s_A, r2s_B, A2B_topk, B2A_topk, A_start=0, B_start=self.kg.source[0].num_ent)

        self.kg.RPG_cover[:self.kg.source[0].num_rel, self.kg.source[0].num_rel:] = A2B_cover*2.0   # top-k will miss many alignments, multiplying by 2 to adjust.
        self.kg.RPG_cover[self.kg.source[0].num_rel:, :self.kg.source[0].num_rel] = B2A_cover*2.0
        self.kg.RPG_cover[self.kg.RPG_cover>1] = 1
        self.kg.RPG_adj_1 = torch.matmul(self.kg.RPG_cover, self.kg.RPG_adj_0)

        A_relation = {i for i in range(self.kg.source[0].num_rel)}
        B_relation = {i+self.kg.source[0].num_rel for i in range(self.kg.num_rel - self.kg.source[0].num_rel)}
        A2B_relation = self.get_relation(A2B_cover, B2A_cover, A_relation, B_relation, threshold=self.args.lambda_2)
        B2A_relation = self.get_relation(B2A_cover, A2B_cover, B_relation, A_relation, threshold=self.args.lambda_2)

        self.kg.RPG_edge = add_RPG_edge_using_cover(self.kg.RPG_adj_1, self.args.lambda_1)
        self.kg.RPG_r2e()
        self.kg.update_attention_weight()

        same_prototypes = self.get_co_relation(A2B_relation, B2A_relation)
        same_relations_bi = self.get_corelation_bidirection(same_prototypes)
        same, inverse, scores_same, scores_inverse = dict(), dict(), dict(), dict()
        for r1, r2 in same_relations_bi:
            if r1 % 2 == 0 and r2 % 2 == 0:
                min_ = 100000
                if r1 in same:
                    min_ = min(min_, same[r1])
                if r2 in same:
                    min_ = min(min_, same[r2])
                if r2 >= self.kg.source[0].num_rel:
                    if (r1 not in scores_same and r2 not in scores_same) or (min(A2B_cover[r1, r2-len(A_relation)], B2A_cover[r2-len(A_relation), r1]) > min_):
                        same[r1] = r2
                        same[r2] = r1
                        scores_same[r1] = A2B_cover[r1, r2-len(A_relation)]
                        scores_same[r2] = B2A_cover[r2 - len(A_relation), r1]
                else:
                    if (r1 not in scores_same and r2 not in scores_same) or (B2A_cover[r1-len(A_relation), r2]+A2B_cover[r2, r1-len(A_relation)] > min_):
                        same[r1] = r2
                        same[r2] = r1
                        scores_same[r1] = B2A_cover[r1-len(A_relation), r2]
                        scores_same[r2] = A2B_cover[r2, r1 - len(A_relation)]
            elif r1 % 2 == 0 and r2 % 2 == 1:
                min_ = 10000
                if r1 in scores_inverse:
                    min_ = min(min_, scores_inverse[r1])
                if r2 in scores_inverse:
                    min_ = min(min_, scores_inverse[r2])
                if r2 >= self.kg.source[0].num_rel:
                    if (r1 not in scores_inverse and self.kg.relation2inv[r2] not in scores_inverse) or min(A2B_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)], B2A_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]) > min_:
                        inverse[r1] = self.kg.relation2inv[r2]
                        inverse[self.kg.relation2inv[r2]] = r1
                        scores_inverse[r1] = A2B_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)]
                        scores_inverse[self.kg.relation2inv[r2]] = B2A_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]
                else:
                    if (r1 not in scores_inverse and self.kg.relation2inv[r2] not in scores_inverse) or min(B2A_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)], A2B_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]) > min_:
                        inverse[r1] = self.kg.relation2inv[r2]
                        inverse[self.kg.relation2inv[r2]] = r1
                        scores_inverse[r1] = B2A_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)]
                        scores_inverse[self.kg.relation2inv[r2]] = A2B_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]


        filtered_same, filtered_inverse = dict(), dict()
        for r1, r2 in same.items():
            if same[r2] == r1:
                filtered_same[r1] = r2
                filtered_same[r2] = r1

        for r1, r2 in inverse.items():
            if inverse[r2] == r1:
                filtered_inverse[r1] = r2
                filtered_inverse[r2] = r1

        print('same', filtered_same)
        print('inverse', filtered_inverse)

        return filtered_same, filtered_inverse

    def find_topk_similar_entities(self, embeddings, topk):
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        similarity_matrix.fill_diagonal_(float('-inf'))
        topk_similar_entities = torch.topk(similarity_matrix, topk, dim=1)[1]
        return topk_similar_entities

    def get_relation(self, A2B_cover, B2A_cover, A_relations, B_relations, threshold=0.5):
        result_dict = {}
        follow = dict()

        for i, a_relation in enumerate(A_relations):
            result_dict[a_relation] = set()

            for j, b_relation in enumerate(B_relations):
                cover_rate = A2B_cover[i][j]

                if cover_rate >= threshold/2:
                    result_dict[a_relation].add(b_relation)
        return result_dict

    def get_co_relation(self, A2B_relation, B2A_relation):
        same_relations = set()
        for rA, rBs in A2B_relation.items():
            for rB in rBs:
                if rA in B2A_relation[rB]:
                    same_relations.add((rA, rB))
                    same_relations.add((rB, rA))
        for rB, rAs in B2A_relation.items():
            for rA in rAs:
                if rB in A2B_relation[rA]:
                    same_relations.add((rA, rB))
                    same_relations.add((rB, rA))
        return same_relations

    def get_corelation_bidirection(self, same_prototypes):
        def rel2inv(r):
            if r % 2 == 0:
                return r+1
            else:
                return r-1
        same_relations_bi = set()
        for r1, r2 in same_prototypes:
            if (rel2inv(r1), rel2inv(r2)) in same_prototypes:
                same_relations_bi.add((r1, r2))
                same_relations_bi.add((r2, r1))

        return same_relations_bi

class RPGTrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        self.neg_ratio = 0
        '''prepare data'''
        self.head_dataset = RPGTrainDataset(self.args, self.kg, mode='head-batch', neg_ratio=self.neg_ratio)
        self.tail_dataset = RPGTrainDataset(self.args, self.kg, mode='tail-batch', neg_ratio=self.neg_ratio)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        head_data_loader = iter(self.head_data_loader)
        tail_data_loader = iter(self.tail_data_loader)
        count_num = 0
        count_loss = 0
        count_batch = 0
        for idx_b, batch in enumerate(tqdm(range(2 * len(self.head_data_loader)))):
            '''get loss'''
            if idx_b % 2 == 0:
                bh, br, bt, by, bc, mode, subsampling_weight = next(head_data_loader)
                mode = 'head-batch'
            else:
                bh, br, bt, by, bc, mode, subsampling_weight = next(tail_data_loader)
                mode = 'tail-batch'

            bh = bh.to(self.args.device)
            br = br.to(self.args.device)
            bt = bt.to(self.args.device)
            by = by.to(self.args.device)
            bc = bc.to(self.args.device)

            subsampling_weight = subsampling_weight.to(self.args.device)
            current_samples_num = subsampling_weight.size(0)
            optimizer.zero_grad()

            jobs = {
                'sub_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bh}, 'mode': mode},
                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": br}, 'mode': mode},
                'obj_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bt}, 'mode': mode},
            }
            pred = model.forward(jobs=jobs, stage='train', mode=mode, margin=2.0)
            batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.neg_ratio)

            batch_loss = batch_loss

            '''update'''
            count_loss += batch_loss * current_samples_num
            count_num += current_samples_num
            count_batch += 1
            if count_num >= self.args.batch_size or self.args.scorer == 'RotatE':
                count_loss = count_loss / count_num
                count_loss.backward()
                optimizer.step()
                # 使编码器缓存失效（对于lookup_gat等使用缓存的编码器）
                if hasattr(model.encoder, 'invalidate_cache'):
                    model.encoder.invalidate_cache()
                total_loss += count_loss.item()
                count_loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
                count_num = 0
                count_batch = 0
        return total_loss

    def loss(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        losses = []
        nums = []
        head_data_loader = iter(self.head_data_loader)
        tail_data_loader = iter(self.tail_data_loader)
        for idx_b, batch in enumerate(range(2*math.ceil(len(self.head_dataset.facts)/self.args.batch_size))):
            '''get loss'''
            if idx_b % 2 == 0:
                bh, br, bt, by, bc, mode, subsampling_weight, mapping_weight = next(head_data_loader)
                mode = 'head-batch'
            else:
                bh, br, bt, by, bc, mode, subsampling_weight, mapping_weight = next(tail_data_loader)
                mode = 'tail-batch'

            bh = bh.to(self.args.device)
            br = br.to(self.args.device)
            bt = bt.to(self.args.device)
            by = by.to(self.args.device)
            bc = bc.to(self.args.device)

            subsampling_weight = subsampling_weight.to(self.args.device)
            mapping_weight = mapping_weight.to(self.args.device)

            jobs = {
                'sub_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bh}, 'mode': mode},
                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": br}, 'mode': mode},
                'obj_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bt}, 'mode': mode},
            }
            pred = model.forward(jobs=jobs, stage='train', mode=mode, margin=self.args.margin)
            batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.neg_ratio)/torch.mean(mapping_weight)
            losses.append(batch_loss * bh.reshape(-1, 1).size(0))
            nums.append(bh.reshape(-1, 1).size(0))
            '''post processing'''
        try:
            total_loss = torch.sum(torch.stack(losses))/sum(nums)
        except:
            total_loss = 0
        return total_loss
    # def loss(self, model, optimizer):
    #     # 使用已知共享实体对齐的 L1 损失
    #     model.train()
    #     # 若没有共享实体对，返回 0
    #     if not hasattr(self.kg, 'entity_pairs') or len(self.kg.entity_pairs) == 0:
    #         return torch.tensor(0.0, device=self.args.device)

    #     # 取实体嵌入（若使用 lookup_attn，会返回注意力聚合后的实体嵌入；否则为查表嵌入）
    #     ent_emb = model.encoder.embed_ent_all(scorer=getattr(model.decoder, 'scorer', None), mode='tail-batch')
    #     pairs = torch.LongTensor(self.kg.entity_pairs).to(self.args.device)

    #     e1 = ent_emb[pairs[:, 0]]
    #     e2 = ent_emb[pairs[:, 1]]

    #     # L1 对齐损失（取均值即可）
    #     align_l1 = torch.mean(torch.abs(e1 - e2))
    #     return align_l1

    def set_RPG_edges(self, facts):
        self.head_dataset.update_facts(facts)
        self.tail_dataset.update_facts(facts)
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                           batch_sampler=self.head_sampler,
                                           collate_fn=self.head_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)









