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
        # 用于计算注意力权重的可学习参数（公式7中的α）
        self.attention_weight = nn.Parameter(torch.randn(1, args.emb_dim * 3))
        # 将三元组嵌入（3*d）映射回实体嵌入维度（d）
        self.triple_proj = nn.Linear(args.emb_dim * 3, args.emb_dim)
        # Gate 融合机制参数（如果使用图间三元组）
        use_inter_triples = getattr(args, 'use_inter_triples', False)
        if use_inter_triples:
            self.gate_weight = nn.Linear(args.emb_dim * 2, args.emb_dim)
            self.gate_bias = nn.Parameter(torch.zeros(args.emb_dim))

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        emb_all = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
        return emb_all[indexes]

    def _triple_wise_aggregation(self, X, R, edge_s, edge_r, edge_o):
        """
        Triple-Wise Aggregation Layer (公式7和8)
        :param X: 实体嵌入 [N, d]
        :param R: 关系嵌入 [num_rel, d]
        :param edge_s: 头实体索引 [E]
        :param edge_r: 关系索引 [E]
        :param edge_o: 尾实体索引 [E]
        :return: 关系聚合后的实体嵌入 [N, d]
        """
        N, d = X.size(0), X.size(1)
        E = edge_s.size(0)
        
        # 获取三元组嵌入
        xh = X[edge_s]  # [E, d] - head实体嵌入
        rr = R[edge_r]  # [E, d] - 关系嵌入
        xt = X[edge_o]  # [E, d] - tail实体嵌入
        
        # 拼接三元组表示 [x_i^head || r_j || x_k^tail] (公式7)
        triple_emb = torch.cat([xh, rr, xt], dim=1)  # [E, 3*d]
        
        # 计算注意力权重（公式7）
        # α_ijk^t = exp(α^T [x_i^head || r_j || x_k^tail])
        attention_scores = torch.matmul(triple_emb, self.attention_weight.t())  # [E, 1]
        attention_scores = attention_scores.squeeze(-1)  # [E]
        
        # 对每个实体e_i的所有三元组进行softmax归一化
        # 使用scatter进行分组softmax
        max_scores = scatter(attention_scores, index=edge_s, dim=0, dim_size=N, reduce='max')  # [N]
        exp_scores = torch.exp(attention_scores - max_scores[edge_s])  # [E]
        sum_exp_scores = scatter(exp_scores, index=edge_s, dim=0, dim_size=N, reduce='sum')  # [N]
        alpha_ijk = exp_scores / (sum_exp_scores[edge_s] + 1e-10)  # [E] - 归一化权重
        
        # 计算关系嵌入（公式8）
        # x_i^rel = ReLU(Σ_{e_k ∈ T_e_i} Σ_{r_j ∈ R_{e_i e_k}} α_ijk^t [x_i^head || r_j || x_k^tail])
        # 注意：这里使用拼接后的三元组嵌入，但最终输出维度需要是d
        # 为了保持维度一致，我们需要将3*d映射回d
        triple_msg = self.triple_proj(triple_emb)  # [E, d]
        triple_msg = self.drop(triple_msg)
        
        # 加权聚合
        weighted_msg = alpha_ijk.unsqueeze(-1) * triple_msg  # [E, d]
        x_rel = scatter(weighted_msg, index=edge_s, dim=0, dim_size=N, reduce='sum')  # [N, d]
        x_rel = torch.relu(x_rel)
        
        return x_rel

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None, graph_mode='intra'):
        """
        计算实体嵌入
        :param graph_mode: 'intra' 使用图内三元组，'inter' 使用图间三元组，'final' 使用gate融合
        """
        use_inter_triples = getattr(self.args, 'use_inter_triples', False)
        
        # 基础嵌入
        X = super().embed_ent_all()  # [N, d]
        R = self.rel_embeddings.weight  # [num_rel, d]
        
        # 计算图内表示 x_intra
        edge_s_intra = self.kg.edge_s.to(self.args.device).long()
        edge_r_intra = self.kg.edge_r.to(self.args.device).long()
        edge_o_intra = self.kg.edge_o.to(self.args.device).long()
        x_rel_intra = self._triple_wise_aggregation(X, R, edge_s_intra, edge_r_intra, edge_o_intra)
        x_intra = X + x_rel_intra  # [N, d]
        
        # 如果不需要使用图间三元组，直接返回图内表示
        if not use_inter_triples or graph_mode == 'intra':
            return x_intra
        
        # 如果只需要图间表示
        if graph_mode == 'inter':
            if not hasattr(self.kg, 'inter_edge_s') or self.kg.inter_edge_s is None or len(self.kg.inter_edge_s) == 0:
                return X
            edge_s_inter = self.kg.inter_edge_s.to(self.args.device).long()
            edge_r_inter = self.kg.inter_edge_r.to(self.args.device).long()
            edge_o_inter = self.kg.inter_edge_o.to(self.args.device).long()
            x_rel_inter = self._triple_wise_aggregation(X, R, edge_s_inter, edge_r_inter, edge_o_inter)
            x_inter = X + x_rel_inter  # [N, d]
            return x_inter
        
        # Gate 融合机制（graph_mode == 'final' 或默认）
        if not hasattr(self.kg, 'inter_edge_s') or self.kg.inter_edge_s is None or len(self.kg.inter_edge_s) == 0:
            # 如果没有图间三元组，返回图内表示
            return x_intra
        
        # 计算图间表示 x_inter
        edge_s_inter = self.kg.inter_edge_s.to(self.args.device).long()
        edge_r_inter = self.kg.inter_edge_r.to(self.args.device).long()
        edge_o_inter = self.kg.inter_edge_o.to(self.args.device).long()
        x_rel_inter = self._triple_wise_aggregation(X, R, edge_s_inter, edge_r_inter, edge_o_inter)
        x_inter = X + x_rel_inter  # [N, d]
        
        # Gate 融合：gate = sigmoid(W_gate * [x_intra || x_inter] + b_gate)
        x_concat = torch.cat([x_intra, x_inter], dim=1)  # [N, 2*d]
        gate_logit = self.gate_weight(x_concat) + self.gate_bias  # [N, d]
        gate = torch.sigmoid(gate_logit)  # [N, d]
        
        # x_final = gate * x_intra + (1 - gate) * x_inter
        x_final = gate * x_intra + (1 - gate) * x_inter  # [N, d]
        
        return x_final
