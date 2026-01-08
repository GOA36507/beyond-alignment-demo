from src.utils import *
from torch_scatter import scatter


class KgeEmbedder(torch.nn.Module):
    def __init__(self, args, kg):
        super().__init__()
        self.args = args
        self.kg = kg
        self.num_ent = kg.num_ent
        self.num_rel = kg.num_rel

        self.ent_dim = args.emb_dim
        self.rel_dim = args.emb_dim

    def embed_ent(self, indexes):
        pass

    def embed_rel(self, indexes):
        pass

    def embed_ent_all(self, indexes=None):
        pass

    def embed_rel_all(self, indexes=None):
        pass


class LookupEmbedder(KgeEmbedder):
    def __init__(self, args, kg):
        super().__init__(args, kg)

    def _create_embedding(self):
        self.ent_embeddings = nn.Embedding(self.num_ent, self.ent_dim).to(self.args.device)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.rel_dim).to(self.args.device)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.args.margin + 2.0) / self.ent_dim]),
            requires_grad=False
        )
        uniform_(self.ent_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        uniform_(self.rel_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        return self.ent_embeddings(indexes)

    def embed_rel(self, indexes):
        return self.rel_embeddings(indexes)

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None):
        return self.ent_embeddings.weight

    def embed_rel_all(self, indexes=None):
        rel_embeddings = self.rel_embeddings.weight
        return torch.cat([rel_embeddings[0::2], rel_embeddings[0::2]], dim=-1).reshape(-1, rel_embeddings.size(-1))

    def embed_ent_prototype(self, indexes, kg=None, scores=None, scorer=None, query_rel=None, mode=None):
        '''
        将每个关系的头实体集合的mean作为prototype
        :param indexes: 需要编码的关系
        :param kg: 指示（p, r, p)中r所属的kg，方便llemapping
        :return: 各关系的prototype
        '''
        if kg is None or kg.all() or not kg.any():
            ent_embeddings = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
            edge_s = self.kg.edge_s.to(self.args.device)
            edge_r = self.kg.edge_r.to(self.args.device)

            s_embeddings = torch.index_select(ent_embeddings, 0, edge_s)
            proto_embeddings = scatter(src=s_embeddings, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')
            return proto_embeddings[indexes]
        else:  # for llemapping
            edge_s = self.kg.edge_s.to(self.args.device)
            edge_r = self.kg.edge_r.to(self.args.device)
            ent_embeddings_1, ent_embeddings_2 = self.embed_ent_all_double(kg=kg)
            # for kg 1
            s_embeddings_1 = torch.index_select(ent_embeddings_1, 0, edge_s)
            proto_embeddings_1 = scatter(src=s_embeddings_1, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')
            # for kg 2
            s_embeddings_2 = torch.index_select(ent_embeddings_2, 0, edge_s)
            proto_embeddings_2 = scatter(src=s_embeddings_2, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')

            # fill prototype embeddings
            proto = torch.zeros([indexes.size(0), ent_embeddings_1.size(-1)], dtype=torch.float).to(self.args.device)
            proto[~kg] = proto_embeddings_1[indexes[~kg]]
            proto[kg] = proto_embeddings_2[indexes[kg]]
            return proto

    def forward(self, **kwargs):
        jobs = kwargs['jobs']
        res = dict()
        for job, value in jobs.items():
            opt_name = value['opt']
            if opt_name == 'ent_embedding':
                opt = self.embed_ent
            elif opt_name == 'rel_embedding':
                opt = self.embed_rel
            elif opt_name == 'ent_embedding_all':
                opt = self.embed_ent_all
            elif opt_name == 'rel_embedding_all':
                opt = self.embed_rel_all
            elif opt_name == 'ent_embedding_prototype':
                opt = self.embed_ent_prototype
            else:
                raise('Invalid embedding opration!')
            input = value['input']
            res[job] = opt(input['indexes'])
        return res


class LookupEmbedderAttn(LookupEmbedder):
    def __init__(self, args, kg):
        super().__init__(args, kg)
        self.drop = torch.nn.Dropout(p=0.0, inplace=False)

    def forward(self, **kwargs):
        jobs = kwargs['jobs']
        scorer = kwargs['scorer']
        mode = kwargs['mode']
        res = dict()
        for job, value in jobs.items():
            opt_name = value['opt']
            if opt_name == 'rel_embedding':
                query_rel = value['input']['indexes']
        for job, value in jobs.items():
            opt_name = value['opt']
            if opt_name == 'ent_embedding':
                opt = self.embed_ent
            elif opt_name == 'rel_embedding':
                opt = self.embed_rel
            elif opt_name == 'ent_embedding_all':
                opt = self.embed_ent_all
            elif opt_name == 'rel_embedding_all':
                opt = self.embed_rel_all
            elif opt_name == 'ent_embedding_prototype':
                opt = self.embed_ent_prototype
            else:
                raise('Invalid embedding opration!')
            input = value['input']
            if 'ent' in opt_name:
                res[job] = opt(indexes=input['indexes'], scorer=scorer, query_rel=query_rel, mode=mode)
            else:
                res[job] = opt(indexes=input['indexes'])
        return res

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        ent_embeddings_attn = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
        return ent_embeddings_attn[indexes]

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None):
        ent_embeddings = super().embed_ent_all(mode=mode)
        rel_embeddings = super().embed_rel_all()
        edge_s = self.kg.edge_s.to(self.args.device)
        edge_r = self.kg.edge_r.to(self.args.device)
        edge_o = self.kg.edge_o.to(self.args.device)

        s_embeddings = torch.index_select(ent_embeddings, 0, edge_s)
        r_embeddings = torch.index_select(rel_embeddings, 0, edge_r)

        if query_rel is not None:
            if mode == 'head-batch':
                qr = query_rel[0] + 1
                qr_inv = query_rel[0]
            else:
                qr = query_rel[0]
                qr_inv = query_rel[0] + 1
            if self.args.valid:
                B = self.kg.attention_weight[qr]
            else:
                B = self.kg.best_attention_weight[qr]
            if B.size(0) < self.kg.num_rel:
                B = torch.cat([B.unsqueeze(1), B.unsqueeze(1)], dim=-1).reshape(-1)
            B[qr] = 0.5
            B[qr_inv] = 0.5
            B = torch.repeat_interleave(torch.max(B[0::2], B[1::2]), 2)
        else:
            B = torch.ones_like(edge_r, dtype=torch.float).to(self.args.device)
        B += 0.1
        co_relation = torch.index_select(B, 0, edge_r).reshape(-1,1)
        ent_embeddings_attn_sum = scatter(src=self.drop(scorer.decode(s=s_embeddings, r=r_embeddings, modes=(edge_r % 2) == 1)*co_relation),
                                      index=edge_o, dim=0, dim_size=ent_embeddings.size(0), reduce='sum')
        weights_attn_sum = scatter(src=co_relation, index=edge_o, dim=0, reduce='sum', dim_size=ent_embeddings.size(0))

        ent_embeddings_attn = ent_embeddings_attn_sum / (weights_attn_sum+1e-10)
        ent_embeddings_attn = ent_embeddings_attn / 2 + ent_embeddings / 2
        return ent_embeddings_attn

class LookupEmbedderGAT(LookupEmbedder):
    def __init__(self, args, kg):
        super().__init__(args, kg)
        self.drop = torch.nn.Dropout(p=0.0, inplace=False)
        
        # 三个消息变换矩阵
        # 1. 图内三元组消息变换：W_intra
        self.W_intra = nn.Linear(args.emb_dim, args.emb_dim)
        # 2. 对齐实体消息变换：W_align
        self.W_align = nn.Linear(args.emb_dim, args.emb_dim)
        # 3. 图间三元组消息变换：W_inter
        self.W_inter = nn.Linear(args.emb_dim, args.emb_dim)
        
        # 缓存机制：每个batch只计算一次全图嵌入
        self.cached_emb = None          # 缓存的全图嵌入 [N, d]
        self.cache_version = -1         # 缓存的版本号
        self.emb_version = 0            # 当前参数版本号
        
        # 预处理的边数据（避免重复to(device)）
        self.edge_s_intra = None
        self.edge_r_intra = None
        self.edge_o_intra = None
        self.edge_s_inter = None
        self.edge_r_inter = None
        self.edge_o_inter = None
        
        # 预处理的对齐实体数据（向量化，避免for循环）
        self.align_src_indices = None   # 有对齐的源实体索引
        self.align_tgt_indices = None   # 对应的目标对齐实体索引
    
    def _preprocess_edges(self):
        """预处理边数据，只在第一次调用时执行一次"""
        if self.edge_s_intra is None:
            self.edge_s_intra = self.kg.edge_s.to(self.args.device).long()
            self.edge_r_intra = self.kg.edge_r.to(self.args.device).long()
            self.edge_o_intra = self.kg.edge_o.to(self.args.device).long()
    
    def _preprocess_alignment(self):
        """预处理对齐实体数据，向量化以避免for循环"""
        if self.align_src_indices is None and hasattr(self.kg, 'ent2align') and len(self.kg.ent2align) > 0:
            src_list = []
            tgt_list = []
            for src, tgt in self.kg.ent2align.items():
                src_list.append(src)
                tgt_list.append(tgt)
            if len(src_list) > 0:
                self.align_src_indices = torch.LongTensor(src_list).to(self.args.device)
                self.align_tgt_indices = torch.LongTensor(tgt_list).to(self.args.device)
    
    def _preprocess_inter_edges(self):
        """预处理图间边数据"""
        use_inter_triples = getattr(self.args, 'use_inter_triples', False)
        if use_inter_triples and self.edge_s_inter is None:
            if hasattr(self.kg, 'inter_edge_s') and self.kg.inter_edge_s is not None and len(self.kg.inter_edge_s) > 0:
                self.edge_s_inter = self.kg.inter_edge_s.to(self.args.device).long()
                self.edge_r_inter = self.kg.inter_edge_r.to(self.args.device).long()
                self.edge_o_inter = self.kg.inter_edge_o.to(self.args.device).long()
    
    def invalidate_cache(self):
        """使缓存失效，在optimizer.step()之后调用"""
        self.emb_version += 1

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        """
        获取指定实体的嵌入，使用缓存机制
        每个batch第一次调用时计算全图嵌入并缓存，后续调用直接从缓存读取
        """
        # 检查缓存是否有效
        if self.cached_emb is None or self.cache_version != self.emb_version:
            # 缓存失效，重新计算全图嵌入（使用最新参数）
            self.cached_emb = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
            self.cache_version = self.emb_version
        # 缓存命中，直接返回所需实体的嵌入
        return self.cached_emb[indexes]

    def _compute_message(self, X, R, edge_s, edge_r, edge_o, W_transform):
        """
        计算消息：Me(u,r) = W · φ(h_u, h_r)
        :param X: 实体嵌入 [N, d]
        :param R: 关系嵌入 [num_rel, d]
        :param edge_s: 头实体索引 [E]
        :param edge_r: 关系索引 [E]
        :param edge_o: 尾实体索引 [E]
        :param W_transform: 变换矩阵（W_intra 或 W_inter）
        :return: 聚合后的消息 [N, d]
        """
        N, d = X.size(0), X.size(1)
        E = edge_s.size(0)
        
        if E == 0:
            # 如果没有边，返回零向量
            return torch.zeros(N, d, device=X.device)
        
        # 获取尾实体和关系嵌入
        h_u = X[edge_o]  # [E, d] - 尾实体嵌入（消息来源）
        h_r = R[edge_r]  # [E, d] - 关系嵌入
        
        
        phi_ur = h_u - h_r  # [E, d]
        
        # 应用变换：W · φ(h_u, h_r)
        msg = W_transform(phi_ur)  # [E, d]
        msg = self.drop(msg)
        
        # 对每个头实体聚合所有消息：Σ_{r∈N(v)} Me(u,r)
        aggregated_msg = scatter(msg, index=edge_s, dim=0, dim_size=N, reduce='sum')  # [N, d]
        
        return aggregated_msg

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None):
        """
        计算实体嵌入（优化版）
        更新公式：h_v^r = h_v^{r-1} + ReLU(Σ_{r∈N(v)} Me(u,r))
        其中 Me(u,r) 包含三个部分：
        1. 图内三元组：W_intra · φ(h_u, h_r)
        2. 对齐实体：W_align · h_v_align（如果v有已知对齐实体）
        3. 图间三元组：W_inter · φ(h_u, h_r)
        
        优化点：
        - 预处理边数据，避免重复to(device)
        - 向量化对齐实体计算，避免for循环
        """
        # 预处理（只在第一次调用时执行）
        self._preprocess_edges()
        self._preprocess_alignment()
        self._preprocess_inter_edges()
        
        # 基础嵌入（第r-1层）
        X = super().embed_ent_all()  # [N, d]
        R = self.rel_embeddings.weight  # [num_rel, d]
        
        # 1. 图内三元组聚合（使用预处理的边数据）
        msg_intra = self._compute_message(X, R, self.edge_s_intra, self.edge_r_intra, 
                                          self.edge_o_intra, self.W_intra)  # [N, d]
        
        # 2. 对齐实体聚合（向量化版本，避免for循环）
        msg_align = torch.zeros_like(X)  # [N, d]
        if self.align_src_indices is not None:
            # 向量化：批量获取对齐实体的嵌入并变换
            h_align = X[self.align_tgt_indices]  # [num_align, d] - 对齐实体的嵌入
            msg_align_values = self.W_align(h_align)  # [num_align, d] - W_align · h_v_align
            # 使用index_add_将消息累加到对应位置（支持多个实体对齐到同一个实体）
            msg_align.index_add_(0, self.align_src_indices, msg_align_values)
        
        # 3. 图间三元组聚合（如果使用图间三元组）
        msg_inter = torch.zeros_like(X)  # [N, d]
        if self.edge_s_inter is not None:
            msg_inter = self._compute_message(X, R, self.edge_s_inter, self.edge_r_inter, 
                                             self.edge_o_inter, self.W_inter)  # [N, d]
        
        # 聚合所有消息：Σ_{r∈N(v)} Me(u,r) = msg_intra + msg_align + msg_inter
        aggregated_msg = msg_intra + msg_align + msg_inter  # [N, d]
        
        # 应用激活函数并更新：h_v^r = h_v^{r-1} + ReLU(Σ_{r∈N(v)} Me(u,r))
        X_new = X + torch.relu(aggregated_msg)  # [N, d]
        
        return X_new
