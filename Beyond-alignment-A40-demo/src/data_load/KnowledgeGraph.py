from src.utils import *
from ..utils import *
from copy import deepcopy as dcopy
from datasets import load_dataset


class KnowledgeGraph():
    def __init__(self, args):
        self.args = args

        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()
        self.entity_pairs = list()

        self.train, self.valid, self.test = list(), list(), list()
        self.edge_s, self.edge_r, self.edge_o = list(), list(), list()
        # 图间三元组结构（用于跨图聚合）
        self.inter_edge_s, self.inter_edge_r, self.inter_edge_o = None, None, None
        self.source = {i: Source(self.args) for i in range(len(self.args.source_list))}
        self.data_path = self.args.root_dir + self.args.source_list[0] + '-' + self.args.source_list[1] + '/'

        self.RPG_rel2edges = dict()
        self.expand_train = list()
        # 伪实体对齐字典（训练过程中动态更新）
        self.pseudo_ent2align = dict()
        self.load_data()
        self.prepare_data()

    @timing_decorator
    def load_data(self):
        '''
        load data from all source file
        '''

        sr2o_all = dict()
        sr2o_valid = dict()
        self.r2s, self.r2o = dict(), dict()

        data_path = self.args.root_dir + self.args.source_list[0] + '-' + self.args.source_list[1]+'/'

        for ss_id, source in enumerate(self.args.source_list):
            edge_s, edge_r, edge_o = [], [], []
            '''load facts'''
            order = 'hrt'
            ds = load_dataset(self.data_path + source)
            train_facts = load_fact(ds['train'], order)
            valid_facts = load_fact(ds['validation'], order)

            test_facts = load_fact(ds['test'], order)
            
            '''extract entities & relations from all facts'''
            # 合并所有数据集的事实
            all_facts = train_facts + valid_facts + test_facts
            entities, relations = self.expand_entity_relation(all_facts)

            # '''extract entities & relations from facts'''
            # entities, relations = self.expand_entity_relation(train_facts)

            entities_id = self.ent_rel2id(entities, 'ent')
            relations_id = self.ent_rel2id(relations, 'rel')

            '''read train/valid data'''
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts)

            self.train += train
            self.valid += valid

            edge_s, edge_o, edge_r = self.expand_kg(train, 'train', edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, self.r2s, self.r2o)
            self.edge_s = self.edge_s + edge_s
            self.edge_r = self.edge_r + edge_r
            self.edge_o = self.edge_o + edge_o
            _, _, _ = self.expand_kg(valid, 'valid', [], [], [], sr2o_all, sr2o_valid, None, None)

            '''store source'''
            self.store_source(ss_id, train, valid, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, entities_id, relations_id)
        
        # 更新 sr2o_valid，添加基于已知实体对齐的扩展三元组
        self.sr2o_valid = sr2o_valid

        '''read shared entities'''
        align_pairs = load_align_pair(data_path, 'known_shared_entities.txt')
        print('Alignment: ', len(align_pairs))
        self.ent2align = dict()
        for pair in align_pairs:
            e1, e2 = pair
            e1, e2 = self.entity2id[e1], self.entity2id[e2]
            self.ent2align[e1] = e2
            self.ent2align[e2] = e1
        self.expand_align(align_pairs)

        # 获取expand的集合
        for source_id, source in self.source.items():
            for fact in source.train:
                s, r, o = fact
                if s in self.ent2align.keys():
                    self.expand_train.append((self.ent2align[s], r, o))
                    self.expand_train.append((o, r+1, self.ent2align[s]))
                if o in self.ent2align.keys():
                    self.expand_train.append((s, r, self.ent2align[o]))
                    self.expand_train.append((self.ent2align[o], r+1, s))
        self.expand_train = set(self.expand_train)

        # for pyg scatter
        self.edge_s = torch.LongTensor(self.edge_s)
        self.edge_r = torch.LongTensor(self.edge_r)
        self.edge_o = torch.LongTensor(self.edge_o)

        # sr2o_valid 已经在 load_data 中更新，这里不需要再次赋值
        self.sr2o_all = sr2o_all

        self.r2s, self.r2o = self.get_r2e(self.train)

        # for RPG
        if self.args.RPG:
            self.get_RPG()
            self.RPG_r2e()
            self.update_attention_weight()


    def get_RPG(self):
        self.ori_RPG_adj_0 = torch.zeros([self.num_rel, self.num_rel], dtype=torch.float)
        self.ori_RPG_adj_1 = torch.zeros([self.num_rel, self.num_rel], dtype=torch.float)
        self.ori_RPG_cover = torch.eye(self.num_rel, dtype=torch.float)
        self.RPG_node = {r for r in range(self.num_rel)}
        self.RPG_node2entity = deepcopy(self.r2s)

        # get cover edge
        for i in range(self.num_rel):
            for j in range(self.num_rel):
                if i == j:
                    self.ori_RPG_adj_0[i, self.relation2inv[j]] = 1.0
                    self.ori_RPG_adj_0[j, self.relation2inv[i]] = 1.0
                    continue
                coverage1 = len(self.RPG_node2entity[i].intersection(self.RPG_node2entity[j]))/len(self.RPG_node2entity[i])
                coverage2 = len(self.RPG_node2entity[i].intersection(self.RPG_node2entity[j]))/len(self.RPG_node2entity[j])
                self.ori_RPG_cover[i, j] = coverage1
                self.ori_RPG_cover[j, i] = coverage2

        self.ori_RPG_adj_1 = torch.matmul(self.ori_RPG_cover, self.ori_RPG_adj_0)
        new_RPG_edge = add_RPG_edge_using_cover(self.ori_RPG_adj_1, self.args.lambda_1)
        self.ori_RPG_edge = new_RPG_edge

        self.RPG_edge = deepcopy(self.ori_RPG_edge)
        self.RPG_adj_0 = deepcopy(self.ori_RPG_adj_0)
        self.RPG_adj_1 = deepcopy(self.ori_RPG_adj_1)
        self.RPG_cover = deepcopy(self.ori_RPG_cover)
        print('Total RPG edges:', len(self.RPG_edge), '\tnature RPG edges:', self.num_rel)

    def RPG_r2e(self):
        self.args.logger.info('RPG graph edges: {}, new edges: {}'.format(len(self.RPG_edge), len(self.RPG_edge)-len(self.ori_RPG_edge)))
        self.RPG_r2s, self.RPG_r2o = self.get_r2e(self.RPG_edge)
        self.RPG_sr2o = dict()
        self.RPG_rel2edges = dict()
        for edge in self.RPG_edge:
            s, r, o = edge[0], edge[1], edge[2]
            item = self.RPG_sr2o.get((s, r), set())
            item.add(o)
            self.RPG_sr2o[(s, r)] = item

            item = self.RPG_sr2o.get((o, self.relation2inv[r]), set())
            item.add(s)
            self.RPG_sr2o[(o, self.relation2inv[r])] = item

            item = self.RPG_rel2edges.get(s, set())
            item.add(edge)
            self.RPG_rel2edges[s] = item
            item = self.RPG_rel2edges.get(o, set())
            item.add(edge)
            self.RPG_rel2edges[o] = item

        self.RPG_count = self.count_frequency(self.RPG_edge)

    def update_attention_weight(self):
        self.attention_weight = torch.zeros([self.num_rel, self.num_rel], dtype=torch.float).to(self.args.device)
        for qr in range(self.num_rel):
            qr_inv = self.relation2inv[qr]

            beta1, beta2 = 0.3, 0.3
            C = self.RPG_cover.to(self.args.device).clone()
            A = self.RPG_adj_1.to(self.args.device).clone()

            conf_matrix_1 = C[:, qr] * A[:, qr_inv]
            conf_matrix_multi = torch.matmul(C[:, qr].unsqueeze(1), A[:, qr_inv].unsqueeze(0))
            supp_matrix_1 = C[qr].t() * A[:, qr_inv]
            supp_matrix_2 = torch.matmul(C[qr].unsqueeze(1), A[:, qr_inv].unsqueeze(0))

            B = torch.zeros([A.size(0)]).to(self.args.device)

            conf_index_1 = torch.nonzero(conf_matrix_1 >= beta2)
            conf_index_multi = torch.nonzero(conf_matrix_multi>=beta2)
            supp_index_1 = torch.nonzero(supp_matrix_1 >= beta1)
            supp_index_2 = torch.nonzero(supp_matrix_2 >= beta1)

            if min(conf_index_1.shape) == 0:
                tuple_conf_1 = dict()
            else:
                tuple_conf_1 = {tuple(row):conf_matrix_1[row[0]] for row in conf_index_1.cpu().tolist()}
            if min(conf_index_multi.shape) == 0:
                tuple_conf_multi = dict()
            else:
                tuple_conf_multi = {tuple(row):conf_matrix_multi[row[0], row[1]] for row in conf_index_multi.cpu().tolist()}
            if min(supp_index_1.shape) == 0:
                tuple_supp_1 = set()
            else:
                tuple_supp_1 = set([tuple(row) for row in supp_index_1.cpu().tolist()])
            if min(supp_index_2.shape) == 0:
                tuple_supp_2 = set()
            else:
                tuple_supp_2 = set([tuple(row) for row in supp_index_2.cpu().tolist()])

            paths_1 = {path: conf for path, conf in tuple_conf_1.items() if path in tuple_supp_1}
            paths_2 = {path: conf for path, conf in tuple_conf_multi.items() if path in tuple_supp_2}

            for paths in [paths_1, paths_2]:
                for path in paths.keys():
                    for r in path:
                        if B[r] < paths[path]:
                            B[r] = paths[path]
                            if r % 2 == 0:
                                r_inv = r+1
                            else:
                                r_inv = r-1
                            B[r_inv] = paths[path]
            self.attention_weight[qr] = B

    def prepare_data(self):
        '''from RotatE'''
        self.count = self.count_frequency(self.train)
        self.sr2o_train = self.get_true_head_and_tail(self.train)

    def count_frequency(self, triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for triple in triples:
            h, r, t = triple[0], triple[1], triple[2]
            if (h, r) not in count:
                count[(h, r)] = start
            else:
                count[(h, r)] += 1
            if (t, r+1) not in count:
                count[(t, r+1)] = start
            else:
                count[(t, r+1)] += 1
        return count

    def get_true_head_and_tail(self, triples):
        '''
        Build a dictionary of true triples that will
        be used to filter negative sampling
        '''
        sr2o = dict()
        for h, r, t in triples:
            if (h, r) not in sr2o:
                sr2o[(h, r)] = []
            if (t, r+1) not in sr2o:
                sr2o[(t, r+1)] = []
            sr2o[(t, r+1)].append(h)
            sr2o[(h, r)].append(t)
        for (h, r) in sr2o.keys():
            sr2o[(h, r)] = np.array(list(set(sr2o[(h, r)])))
        return sr2o
    
    def load_test(self):
        if len(self.test) > 0:
            return

        cross_ds = load_dataset(self.data_path + 'cross')
        cross_test_facts = load_fact(cross_ds['test'], 'hrt')
        self.cross_test = self.fact2id(cross_test_facts)

        self.inner_test = []
        self.inner_test_align_0, self.inner_test_align_1, self.inner_test_align_2 = [], [], []
        for ss_id, source in enumerate(self.args.source_list):
            ds = load_dataset(self.data_path + source)
            inner_test_facts = load_fact(ds['test'], 'hrt')
            self.inner_test += self.fact2id(inner_test_facts)
        self.test = self.cross_test + self.inner_test
        _, _, _ = self.expand_kg(self.test, 'test', [], [], [], self.sr2o_all, None, None, None)
        self.split_test_to_source(self.test)

        # only for filter
        data_path = self.args.root_dir + self.args.source_list[0] + '-' + self.args.source_list[1] + '/'
        all_align_pairs = load_align_pair(data_path, 'all_shared_entities.txt')
        ent2align_all = dict()
        for pair in all_align_pairs:
            e1, e2 = pair
            ent2align_all[self.entity2id[e1]] = self.entity2id[e2]
            ent2align_all[self.entity2id[e2]] = self.entity2id[e1]
        self.sr2o_test = dict()
        for facts in [self.train, self.valid, self.test]:
            for fact in facts:
                s, r, o = fact
                item = self.sr2o_test.get((s, r), set())
                item.add(o)
                self.sr2o_test[(s, r)] = item
                item = self.sr2o_test.get((o, r + 1), set())
                item.add(s)
                self.sr2o_test[(o, r + 1)] = item
                if s in ent2align_all.keys():
                    item = self.sr2o_test.get((ent2align_all[s], r), set())
                    item.add(o)
                    self.sr2o_test[(ent2align_all[s], r)] = item
                    item = self.sr2o_test.get((o, r + 1), set())
                    item.add(ent2align_all[s])
                    self.sr2o_test[(o, r + 1)] = item
                if o in ent2align_all.keys():
                    item = self.sr2o_test.get((s, r), set())
                    item.add(ent2align_all[o])
                    self.sr2o_test[(s, r)] = item
                    item = self.sr2o_test.get((ent2align_all[o], r + 1), set())
                    item.add(s)
                    self.sr2o_test[(ent2align_all[o], r + 1)] = item

    def ent_rel2id(self, lst, mode='ent'):
        res = []
        for item in lst:
            if mode == 'ent':
                res.append(self.entity2id[item])
            else:
                res.append(self.relation2id[item])
        return res

    @timing_decorator
    def expand_entity_relation(self, facts):
        '''extract entities and relations from new facts'''
        new_entities, new_relations = set(), set()
        for (s, r, o) in facts:
            '''extract entities'''
            new_entities.add(s)
            new_relations.add(r)
            new_entities.add(o)
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_ent
                self.id2entity[self.num_ent] = s
                self.num_ent += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_ent
                self.id2entity[self.num_ent] = o
                self.num_ent += 1

            '''extract relations'''
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.id2relation[self.num_rel] = r
                self.id2relation[self.num_rel + 1] = r + '_inv'
                self.relation2inv[self.num_rel] = self.num_rel + 1
                self.relation2inv[self.num_rel + 1] = self.num_rel
                self.num_rel += 2
        print('Entities {} Relations {}'.format(self.num_ent, self.num_rel))
        return new_entities, new_relations

    @timing_decorator
    def split_test_to_source(self, test):
        for idx, kg in self.source.items():
            max_id = max(kg.entities)
            min_id = min(kg.entities)
            for fact in test:
                h, r, t = fact
                if h <= max_id and h >= min_id:
                    kg.test.add(fact)

    @timing_decorator
    def fact2id(self, facts):
        '''(s name, r name, o name)-->(s id, r id, o id)'''
        fact_id = []
        for (s, r, o) in facts:
            fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    @timing_decorator
    def expand_kg(self, facts, split, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, r2s, r2o):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        def add_key2val(dict, key, val):
            '''add {key: value} to dict'''
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)
        edge_s.clear()
        edge_o.clear()
        edge_r.clear()
        for (h, r, t) in facts:
            if split == 'train':
                '''edge_index'''
                edge_s.append(h)
                edge_r.append(r)
                edge_o.append(t)
                edge_s.append(t)
                edge_r.append(r+1)
                edge_o.append(h)
                add_key2val(r2s, r, h)
                add_key2val(r2o, r, t)
                add_key2val(r2s, r+1, t)
                add_key2val(r2o, r+1, h)
            '''sr2o'''
            add_key2val(sr2o_all, (h, r), t)
            add_key2val(sr2o_all, (t, self.relation2inv[r]), h)
            if split in ['train', 'valid']:
                add_key2val(sr2o_valid, (h, r), t)
                add_key2val(sr2o_valid, (t, self.relation2inv[r]), h)
        return edge_s, edge_o, edge_r
    
    @timing_decorator
    def expand_align(self, align_pairs):
        for pair in align_pairs:
            if pair[0] not in self.entity2id.keys() or pair[1] not in self.entity2id.keys():
                continue
            self.entity_pairs.append((self.entity2id[pair[0]], self.entity2id[pair[1]]))

    @timing_decorator
    def store_source(self, ss_id, train_new, valid, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, entities, relations):
        '''store source data'''
        if ss_id > 0:
            self.source[ss_id].num_ent = self.num_ent - self.source[ss_id - 1].num_ent
            self.source[ss_id].num_rel = self.num_rel - self.source[ss_id - 1].num_rel
        else:
            self.source[ss_id].num_ent = self.num_ent
            self.source[ss_id].num_rel = self.num_rel

        '''train, valid and test data'''
        self.source[ss_id].train = dcopy(train_new)
        self.source[ss_id].valid = dcopy(valid)
        self.source[ss_id].entities = sorted(entities)
        self.source[ss_id].relations = sorted(relations)

        '''edge_index, edge_type (for GCN)'''
        self.source[ss_id].edge_s = dcopy(edge_s)
        self.source[ss_id].edge_r = dcopy(edge_r)
        self.source[ss_id].edge_o = dcopy(edge_o)

    def get_r2e(self, facts):
        r2o, r2s = dict(), dict()
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
            # r2o
            item = r2o.get(r, set())
            item.add(o)
            r2o[r] = item
            # r2s
            item = r2s.get(r, set())
            item.add(s)
            r2s[r] = item
            # r_inv2o
            item = r2o.get(r+1, set())
            item.add(o)
            r2o[r+1] = item
            # r_inv2s
            item = r2s.get(r+1, set())
            item.add(o)
            r2s[r+1] = item
        return r2s, r2o

    def build_inter_graph_structure(self, train_datasets=None):
        """
        构建图间三元组结构（用于跨图聚合）
        从训练数据集中提取扩展三元组（基于已知对齐对和伪对齐对），并添加反向关系三元组
        :param train_datasets: 训练数据集列表，包含 (head_dataset, tail_dataset)，如果为None则使用原始方法
        :return: 统计信息字典，包含各种三元组数量
        """
        inter_edge_s_list = []
        inter_edge_r_list = []
        inter_edge_o_list = []
        
        # 统计信息
        num_known_expand_triples = 0
        num_pseudo_expand_triples = 0
        
        # 收集所有扩展三元组
        expand_triples = set()
        
        # 如果提供了训练数据集，从数据集中提取扩展三元组
        if train_datasets is not None:
            head_dataset, tail_dataset = train_datasets
            # 合并两个数据集的所有facts（去重）
            all_facts = set(head_dataset.facts) | set(tail_dataset.facts)
            
            # 获取is_expand和is_ea信息
            # 由于head_dataset和tail_dataset的facts顺序可能不同，我们需要通过索引匹配
            head_facts_dict = {}
            for i, fact in enumerate(head_dataset.facts):
                is_expand = head_dataset.is_expand[i]
                is_ea = head_dataset.is_ea[i]
                # 处理tensor类型（可能是2D tensor，需要squeeze）
                if isinstance(is_expand, torch.Tensor):
                    is_expand = is_expand.squeeze().item() if is_expand.numel() > 0 else False
                if isinstance(is_ea, torch.Tensor):
                    is_ea = is_ea.squeeze().item() if is_ea.numel() > 0 else False
                head_facts_dict[fact] = (bool(is_expand), bool(is_ea))
            
            tail_facts_dict = {}
            for i, fact in enumerate(tail_dataset.facts):
                is_expand = tail_dataset.is_expand[i]
                is_ea = tail_dataset.is_ea[i]
                # 处理tensor类型（可能是2D tensor，需要squeeze）
                if isinstance(is_expand, torch.Tensor):
                    is_expand = is_expand.squeeze().item() if is_expand.numel() > 0 else False
                if isinstance(is_ea, torch.Tensor):
                    is_ea = is_ea.squeeze().item() if is_ea.numel() > 0 else False
                tail_facts_dict[fact] = (bool(is_expand), bool(is_ea))
            
            # 合并信息（优先使用head_dataset的信息）
            facts_info = {}
            facts_info.update(head_facts_dict)
            facts_info.update(tail_facts_dict)
            
            # 从训练数据中提取扩展三元组
            for fact in all_facts:
                if fact in facts_info:
                    is_expand, is_ea = facts_info[fact]
                    if is_expand:
                        expand_triples.add(fact)
                        if is_ea:
                            # 伪对齐对扩展的三元组
                            num_pseudo_expand_triples += 1
                        else:
                            # 已知对齐对扩展的三元组
                            num_known_expand_triples += 1
        else:
            # 原始方法：直接从source.train构建（用于向后兼容）
            # 1. 基于已知实体对齐对的扩展三元组
            for source_id, source in self.source.items():
                for fact in source.train:
                    s, r, o = fact
                    # 如果subject在已知对齐对中
                    if s in self.ent2align.keys():
                        expand_triple = (self.ent2align[s], r, o)
                        expand_triples.add(expand_triple)
                        num_known_expand_triples += 1
                    # 如果object在已知对齐对中
                    if o in self.ent2align.keys():
                        expand_triple = (s, r, self.ent2align[o])
                        expand_triples.add(expand_triple)
                        num_known_expand_triples += 1
            
            # 2. 基于伪实体对齐对的扩展三元组
            if hasattr(self, 'pseudo_ent2align') and len(self.pseudo_ent2align) > 0:
                for source_id, source in self.source.items():
                    for fact in source.train:
                        s, r, o = fact
                        # 如果subject在伪对齐对中
                        if s in self.pseudo_ent2align.keys():
                            expand_triple = (self.pseudo_ent2align[s], r, o)
                            expand_triples.add(expand_triple)
                            num_pseudo_expand_triples += 1
                        # 如果object在伪对齐对中
                        if o in self.pseudo_ent2align.keys():
                            expand_triple = (s, r, self.pseudo_ent2align[o])
                            expand_triples.add(expand_triple)
                            num_pseudo_expand_triples += 1
        
        # 为每个扩展三元组添加反向关系三元组
        for (h, r, t) in expand_triples:
            # 原始三元组
            inter_edge_s_list.append(h)
            inter_edge_r_list.append(r)
            inter_edge_o_list.append(t)
            # 反向关系三元组 (t, r+1, h)
            inter_edge_s_list.append(t)
            inter_edge_r_list.append(r + 1)
            inter_edge_o_list.append(h)
        
        # 转换为tensor
        if len(inter_edge_s_list) > 0:
            self.inter_edge_s = torch.LongTensor(inter_edge_s_list)
            self.inter_edge_r = torch.LongTensor(inter_edge_r_list)
            self.inter_edge_o = torch.LongTensor(inter_edge_o_list)
        else:
            self.inter_edge_s = torch.LongTensor([])
            self.inter_edge_r = torch.LongTensor([])
            self.inter_edge_o = torch.LongTensor([])
        
        # 统计信息（包含反向关系三元组）
        num_known_expand_with_reverse = num_known_expand_triples * 2
        num_pseudo_expand_with_reverse = num_pseudo_expand_triples * 2
        num_original_with_reverse = len(self.edge_s)  # 原始图结构已包含反向关系
        
        # 添加调试信息：计算原始三元组数量（不包含反向关系）
        num_original_without_reverse = len(self.train)  # 原始训练三元组数量（不包含反向关系）
        
        # 调试日志：输出详细信息
        if hasattr(self.args, 'logger'):
            self.args.logger.info(
                f'[DEBUG] build_inter_graph_structure: '
                f'num_original_without_reverse={num_original_without_reverse}, '
                f'num_original_with_reverse={num_original_with_reverse}, '
                f'len(edge_s)={len(self.edge_s)}, '
                f'len(train)={len(self.train)}, '
                f'ratio={num_original_with_reverse/num_original_without_reverse if num_original_without_reverse > 0 else 0:.2f}'
            )
            self.args.logger.info(
                f'[DEBUG] Expand triples: '
                f'num_known_expand_triples={num_known_expand_triples}, '
                f'num_known_expand_with_reverse={num_known_expand_with_reverse}, '
                f'num_pseudo_expand_triples={num_pseudo_expand_triples}, '
                f'num_pseudo_expand_with_reverse={num_pseudo_expand_with_reverse}'
            )
        
        stats = {
            'num_original_triples': num_original_with_reverse,
            'num_known_expand_triples': num_known_expand_with_reverse,
            'num_pseudo_expand_triples': num_pseudo_expand_with_reverse
        }
        
        return stats


class Source:
    def __init__(self, args):
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train, self.valid, self.test = list(), list(), set()
        self.cross_train = list()
        self.entities, self.relations = None, None

        self.edge_s, self.edge_r, self.edge_o = [], [], []
        self.cross_edge_s, self.cross_edge_r, self.cross_edge_o = [], [], []
        self.sr2o_all = dict()
        self.edge_index, self.edge_type = None, None



