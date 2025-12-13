from torch.utils.data import Dataset
from src.utils import *
from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
import torch


'''for Cross-KG Link Prediction'''
class CrossLinkPredictionTrainDatasetMarginLoss(Dataset):
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode
        self.original_facts, self.original_is_expand, self.original_is_ea = self.build_facts()
        self.original_is_expand = torch.BoolTensor(self.original_is_expand).reshape(-1, 1)
        self.original_is_ea = torch.BoolTensor(self.original_is_ea).reshape(-1, 1)
        self.original_confidence = torch.ones([len(self.original_facts), 1], dtype=torch.float32)

        self.facts = deepcopy(self.original_facts)
        self.confidence = self.original_confidence.clone()
        self.is_expand = self.original_is_expand.clone()
        self.is_ea = self.original_is_ea.clone()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        '''
        :param idx: idx of the training fact
        :return: a positive facts and its negative facts
        '''
        fact = self.facts[idx]
        conf = self.confidence[idx]
        is_expand = self.is_expand[idx]
        is_ea = self.is_ea[idx]
        if is_expand:
            mf = list(self.kg.RPG_rel2edges.get(fact[1], set()))

        else:
            mf = list()
        '''negative sampling'''
        fact, label, subsampling_weight = self.corrupt(fact, is_ea)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        # 将confidence扩展到所有样本（正样本+负样本）
        num_samples = len(fact)
        conf_expanded = conf.repeat(num_samples, 1) if conf.dim() > 0 else conf.repeat(num_samples)
        return fact, label, conf_expanded, self.mode, subsampling_weight, mf, torch.ones(len(fact), dtype=torch.bool)*is_expand, torch.ones(len(fact), dtype=torch.bool)*is_ea

    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        conf = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        subsampling_weight = torch.cat([_[4] for _ in data], dim=0)
        RPG_facts = []
        for _ in data:
            mf = _[5]
            RPG_facts += mf
        is_expands = torch.cat([_[6] for _ in data], dim=0)
        is_eas = torch.cat([_[7] for _ in data], dim=0)
        return fact[:,0], fact[:,1], fact[:,2], label, conf, mode, subsampling_weight, RPG_facts, is_expands, is_eas

    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        head_facts = list()
        is_expands = list()
        is_ea = list()
        all_relation, cross_relation_head, cross_relation_tail = set(), set(), set()
        for source_id, source in self.kg.source.items():
            expand_fact_num = 0
            for fact in source.train:
                s, r, o = fact
                all_relation.add(r)
                head_facts.append((s, r, o))
                is_expands.append(False)
                is_ea.append(False)
                if self.args.ea_expand_training:
                    if s in self.kg.ent2align.keys():
                        cross_relation_head.add(r)
                        head_facts.append((self.kg.ent2align[s], r, o))
                        is_expands.append(True)
                        is_ea.append(False)
                        expand_fact_num += 1
                    if o in self.kg.ent2align.keys():
                        cross_relation_tail.add(r)
                        head_facts.append((s, r, self.kg.ent2align[o]))
                        is_expands.append(True)
                        is_ea.append(False)
                        expand_fact_num += 1
            if self.args.ea_expand_training:
                self.args.logger.info('KG {} Expand Training Triples: {}'.format(source_id, expand_fact_num))
                self.args.logger.info('All Relation {}, cross relation head {}, cross relation tail {}.'.format(len(all_relation), len(cross_relation_head), len(cross_relation_tail)))
        return head_facts, is_expands, is_ea

    def corrupt(self, fact, is_ea):
        s, r, o = fact
        # 对于基于伪实体扩展的三元组（is_ea=True），也要正常构造负样本，与基于已知实体扩展的三元组一致
        # 使用对齐后的实体来计算subsampling_weight和负样本过滤，与已知实体扩展三元组保持一致
        s_temp, o_temp, r_temp = s, o, r
        try:
            subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(o, self.kg.relation2inv[r])]
        except:
            try:
                # 优先尝试使用已知实体对齐
                if hasattr(self.kg, 'ent2align') and s in self.kg.ent2align:
                    subsampling_weight = self.kg.count[(self.kg.ent2align[s], r)] + self.kg.count[(o, self.kg.relation2inv[r])]
                    s_temp = self.kg.ent2align[s]
                # 然后尝试使用伪实体对齐（与已知实体对齐处理方式一致）
                elif hasattr(self.kg, 'pseudo_ent2align') and s in self.kg.pseudo_ent2align:
                    subsampling_weight = self.kg.count[(self.kg.pseudo_ent2align[s], r)] + self.kg.count[(o, self.kg.relation2inv[r])]
                    s_temp = self.kg.pseudo_ent2align[s]
                else:
                    raise
            except:
                try:
                    # 优先尝试使用已知实体对齐
                    if hasattr(self.kg, 'ent2align') and o in self.kg.ent2align:
                        subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(self.kg.ent2align[o], self.kg.relation2inv[r])]
                        o_temp = self.kg.ent2align[o]
                    # 然后尝试使用伪实体对齐（与已知实体对齐处理方式一致）
                    elif hasattr(self.kg, 'pseudo_ent2align') and o in self.kg.pseudo_ent2align:
                        subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(self.kg.pseudo_ent2align[o], self.kg.relation2inv[r])]
                        o_temp = self.kg.pseudo_ent2align[o]
                    else:
                        raise
                except:
                    try:
                        try:
                            r_temp = self.same[r]
                            subsampling_weight = self.kg.count[(s, r_temp)] + self.kg.count[(o, self.kg.relation2inv[r_temp])]
                        except:
                            r_temp = self.kg.relation2inv[self.inverse[r]]
                            subsampling_weight = self.kg.count[(s, r_temp)] + self.kg.count[(o, self.kg.relation2inv[r_temp])]
                    except:
                        print(s, r, o)
                        # 如果所有方法都失败，使用默认值
                        subsampling_weight = 1.0
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        facts = [fact]
        label = [1]
        negative_sample_size = 0
        negative_sample_list = []

        while negative_sample_size < self.args.neg_ratio:
            if self.mode == 'head-batch':
                negative_sample = np.random.randint(0, self.kg.num_ent, self.args.neg_ratio * 2)

                mask = np.in1d(
                    negative_sample,
                    self.kg.sr2o_train[(o_temp, self.kg.relation2inv[r_temp])],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                negative_sample = np.random.randint(0, self.kg.num_ent, self.args.neg_ratio * 2)

                mask = np.in1d(
                    negative_sample,
                    self.kg.sr2o_train[(s_temp, r_temp)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_size += negative_sample.size
            negative_sample_list.append(negative_sample)

        negative_sample = np.concatenate(negative_sample_list)[:self.args.neg_ratio]
        if self.mode == 'head-batch':
            facts.extend([(negative_sample[i], r, o) for i in range(self.args.neg_ratio)])
            label += [-1 for i in range(self.args.neg_ratio)]
        elif self.mode == 'tail-batch':
            facts.extend([(s, r, negative_sample[i]) for i in range(self.args.neg_ratio)])
            label += [-1 for i in range(self.args.neg_ratio)]

        return facts, label, subsampling_weight

    def add_facts_using_relations(self, same, inverse):
        self.same = {val:key for key, val in same.items()}
        self.inverse = {val:key for key, val in inverse.items()}
        facts, confidence, is_expands, is_ea = [], [], [], []
        for fact in self.kg.train:
            s, r, o = fact
            if r in same.keys():
                facts.append((s, same[r], o))
                confidence.append([0.5])
                is_expands.append(True)
                is_ea.append(False)
            if r in inverse.keys():
                facts.append((o, inverse[r], s))
                confidence.append([0.5])
                is_expands.append(True)
                is_ea.append(False)
        self.args.logger.info('Add {} facts using relations!'.format(len(facts)))
        self.facts = self.original_facts + facts
        self.confidence = torch.cat([self.original_confidence, torch.FloatTensor(confidence)], dim=0)
        self.is_expand = torch.cat([self.original_is_expand, torch.BoolTensor(is_expands).reshape(-1, 1)])
        self.is_ea = torch.cat([self.original_is_ea, torch.BoolTensor(is_ea).reshape(-1, 1)])

    def _compute_triple_score(self, model, s, r, o, mode='tail-batch'):
        """
        计算三元组的得分（使用模型）
        :param model: 模型
        :param s: subject实体ID
        :param r: relation ID
        :param o: object实体ID
        :param mode: 计算模式
        :return: 三元组得分
        """
        device = next(model.parameters()).device
        
        s_tensor = torch.LongTensor([s]).to(device)
        r_tensor = torch.LongTensor([r]).to(device)
        o_tensor = torch.LongTensor([o]).to(device)
        
        jobs = {
            'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": s_tensor}, 'mode': mode},
            'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": r_tensor}, 'mode': mode},
            'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": o_tensor}, 'mode': mode},
        }
        
        with torch.no_grad():
            score = model.forward(jobs=jobs, stage='train', mode=mode, margin=self.args.margin)
            # score可能是tensor，需要转换为标量
            if isinstance(score, torch.Tensor):
                if score.numel() == 1:
                    return score.item()
                else:
                    return score[0].item() if len(score) > 0 else 0.0
            return float(score)
    
    def _compute_entity_degree(self, entity):
        """
        计算实体的出入度平均值（基于训练集）
        :param entity: 实体ID
        :return: 出入度平均值
        """
        out_degree = 0  # 作为subject的三元组数量
        in_degree = 0   # 作为object的三元组数量
        
        # 统计出度：作为subject的三元组
        for fact in self.kg.train:
            s, r, o = fact
            if s == entity:
                out_degree += 1
        
        # 统计入度：作为object的三元组
        for fact in self.kg.train:
            s, r, o = fact
            if o == entity:
                in_degree += 1
        
        # 计算平均值
        avg_degree = (out_degree + in_degree) / 2.0 if (out_degree + in_degree) > 0 else 0.0
        return avg_degree
    
    def add_facts_using_pseudo_entities(self, model, replace_ratio=1.0, selection_ratio=None):
        """
        基于伪实体对构造扩展三元组，支持四种选择策略
        :param model: 模型，用于计算三元组得分
        :param replace_ratio: 实体替换的比例（0.0-1.0），控制部分实体替换
        :param selection_ratio: 三元组选择比例，如果为None则使用args中的值
        :return: 添加的伪扩展三元组数量
        """
        self.args.logger.info(
            f'[DEBUG] add_facts_using_pseudo_entities: Starting (replace_ratio={replace_ratio}, selection_ratio={selection_ratio})'
        )
        if not hasattr(self.kg, 'pseudo_ent2align') or len(self.kg.pseudo_ent2align) == 0:
            self.args.logger.info('[DEBUG] add_facts_using_pseudo_entities: No pseudo_ent2align found, returning 0')
            return 0
        
        pseudo_ent2align = self.kg.pseudo_ent2align
        
        # 收集所有伪实体对
        pseudo_pairs = []
        used_entities = set()
        for e1, e2 in pseudo_ent2align.items():
            if e1 not in used_entities and e2 not in used_entities:
                pseudo_pairs.append((e1, e2))
                used_entities.add(e1)
                used_entities.add(e2)
        
        # 根据replace_ratio选择伪实体对
        num_pairs_to_use = max(1, int(len(pseudo_pairs) * replace_ratio))
        if num_pairs_to_use < len(pseudo_pairs):
            selected_indices = np.random.choice(len(pseudo_pairs), size=num_pairs_to_use, replace=False)
            selected_pairs = [pseudo_pairs[i] for i in selected_indices]
        else:
            selected_pairs = pseudo_pairs
        
        selected_entities = set()
        for e1, e2 in selected_pairs:
            selected_entities.add(e1)
            selected_entities.add(e2)
        
        # 生成所有候选伪三元组
        candidate_triples = []  # [(s, r, o, pseudo_entity, is_subject), ...]
        for fact in self.kg.train:
            s, r, o = fact
            # 如果subject在选中的伪实体对中，替换它
            if s in selected_entities and s in pseudo_ent2align:
                aligned_s = pseudo_ent2align[s]
                candidate_triples.append((aligned_s, r, o, s, True))  # True表示替换的是subject
            # 如果object在选中的伪实体对中，替换它
            if o in selected_entities and o in pseudo_ent2align:
                aligned_o = pseudo_ent2align[o]
                candidate_triples.append((s, r, aligned_o, o, False))  # False表示替换的是object
        
        self.args.logger.info(f'[DEBUG] add_facts_using_pseudo_entities: Found {len(candidate_triples)} candidate triples')
        
        if len(candidate_triples) == 0:
            self.args.logger.info('[DEBUG] add_facts_using_pseudo_entities: No candidate triples, returning 0')
            return 0
        
        # 根据选择策略筛选三元组
        selection_strategy = getattr(self.args, 'pseudo_triple_selection', 'random')
        if selection_ratio is None:
            selection_ratio = getattr(self.args, 'pseudo_triple_selection_ratio', 1.0)
        
        if selection_strategy == 'random':
            # 随机选择
            num_to_select = max(1, int(len(candidate_triples) * selection_ratio))
            if num_to_select < len(candidate_triples):
                selected_indices = np.random.choice(len(candidate_triples), size=num_to_select, replace=False)
                selected_triples = [candidate_triples[i] for i in selected_indices]
            else:
                selected_triples = candidate_triples
        
        elif selection_strategy in ['score_high', 'score_low']:
            # 按得分排序选择
            triple_scores = []
            for triple in candidate_triples:
                s, r, o = triple[0], triple[1], triple[2]
                mode = 'tail-batch' if triple[4] else 'head-batch'  # 根据替换的是subject还是object选择mode
                score = self._compute_triple_score(model, s, r, o, mode)
                triple_scores.append((triple, score))
            
            # 排序
            if selection_strategy == 'score_high':
                triple_scores.sort(key=lambda x: x[1], reverse=True)  # 从高到低
            else:
                triple_scores.sort(key=lambda x: x[1], reverse=False)  # 从低到高
            
            # 选择前k%
            num_to_select = max(1, int(len(triple_scores) * selection_ratio))
            selected_triples = [item[0] for item in triple_scores[:num_to_select]]
        
        elif selection_strategy in ['degree_high', 'degree_low']:
            # 按伪实体出入度排序选择
            entity_degrees = {}  # {pseudo_entity: avg_degree}
            for triple in candidate_triples:
                pseudo_entity = triple[3]
                if pseudo_entity not in entity_degrees:
                    entity_degrees[pseudo_entity] = self._compute_entity_degree(pseudo_entity)
            
            # 按出入度排序伪实体
            sorted_entities = sorted(entity_degrees.items(), 
                                    key=lambda x: x[1], 
                                    reverse=(selection_strategy == 'degree_high'))
            
            # 选择前k%的伪实体
            num_entities_to_select = max(1, int(len(sorted_entities) * selection_ratio))
            selected_entity_set = set([item[0] for item in sorted_entities[:num_entities_to_select]])
            
            # 选择这些伪实体对应的所有三元组
            selected_triples = [triple for triple in candidate_triples if triple[3] in selected_entity_set]
        
        else:
            # 默认随机选择
            num_to_select = max(1, int(len(candidate_triples) * selection_ratio))
            if num_to_select < len(candidate_triples):
                selected_indices = np.random.choice(len(candidate_triples), size=num_to_select, replace=False)
                selected_triples = [candidate_triples[i] for i in selected_indices]
            else:
                selected_triples = candidate_triples
        
        # 构建最终的三元组列表
        facts, confidence, is_expands, is_ea = [], [], [], []
        for triple in selected_triples:
            s, r, o = triple[0], triple[1], triple[2]
            facts.append((s, r, o))
            confidence.append([1.0])
            is_expands.append(True)
            is_ea.append(True)
        
        self.args.logger.info(
            f'[DEBUG] add_facts_using_pseudo_entities: Selected {len(selected_triples)} triples '
            f'from {len(candidate_triples)} candidates using {selection_strategy} strategy'
        )
        
        if len(facts) > 0:
            self.args.logger.info('Add {} pseudo expand triples using {} strategy (ratio={:.4f})!'.format(
                len(facts), selection_strategy, selection_ratio))
            self.facts = self.original_facts + facts
            self.confidence = torch.cat([self.original_confidence, torch.FloatTensor(confidence)], dim=0)
            self.is_expand = torch.cat([self.original_is_expand, torch.BoolTensor(is_expands).reshape(-1, 1)])
            self.is_ea = torch.cat([self.original_is_ea, torch.BoolTensor(is_ea).reshape(-1, 1)])
            self.args.logger.info(
                f'[DEBUG] add_facts_using_pseudo_entities: Updated dataset - '
                f'total_facts={len(self.facts)}, original_facts={len(self.original_facts)}, '
                f'new_facts={len(facts)}'
            )
        else:
            self.args.logger.info('[DEBUG] add_facts_using_pseudo_entities: No facts to add, returning 0')
        
        return len(facts)


class RelationBatchSampler(BatchSampler):
    '''
    This class is created to save memory for the attn embedder.
    The training using attn embedder involves facts with the same relation placed within the same batch.
    '''
    def __init__(self, args, kg, dataset, batch_size, shuffle=True):
        self.args = args
        self.kg = kg
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.args.use_attn:
            #  to save memory, we group identical relations into the same batch
            self.relation_samples = self._separate_relations()
            super(RelationBatchSampler, self).__init__(self.relation_samples, self.batch_size, self.shuffle)
        else:
            super(RelationBatchSampler, self).__init__(self.dataset, self.batch_size, self.shuffle)

    def _separate_relations(self):
        relation_samples = {}
        for idx, sample in enumerate(self.dataset):
            try:
                relation = sample['fact'][1]
            except:
                relation = sample[1]
            try:
                if relation not in relation_samples.keys():
                    relation_samples[relation] = []
            except:
                print(relation)
            relation_samples[relation].append(idx)
        return relation_samples

    def __iter__(self):
        if not self.args.use_attn:
            dataset = [i for i in range(len(self.dataset))]
            batches = []
            if self.shuffle:
                random.shuffle(dataset)
            num_batches = len(dataset) // self.batch_size
            for i in range(num_batches):
                batch = list(dataset)[i * self.batch_size: (i + 1) * self.batch_size]
                batches.append(batch)
                # yield(batch)
            if len(dataset) % self.batch_size > 0:
                # yield self.dataset[num_batches * self.batch_size:]
                batches.append(list(dataset)[num_batches * self.batch_size:])
            if self.shuffle:
                random.shuffle(batches)
            return iter(batches)
        else:
            self.relation_indices = list(self.relation_samples.keys())
            if self.shuffle:
                random.shuffle(self.relation_indices)
            batches = []
            for relation in self.relation_indices:
                samples = self.relation_samples[relation]
                random.shuffle(samples)
                num_samples = len(samples)
                num_batches = num_samples // self.batch_size
                for i in range(num_batches):
                    batch = samples[i * self.batch_size: (i + 1) * self.batch_size]
                    batches.append(batch)
                if num_samples % self.batch_size > 0:
                    batches.append(samples[num_batches * self.batch_size:])
            if self.shuffle:
                random.shuffle(batches)
            return iter(batches)

    def __len__(self):
        if self.args.use_attn:
            num = 0
            for samples in self.relation_samples.values():
                num += len(samples)//self.batch_size
                if len(samples) % self.batch_size > 0:
                    num += 1
            return num
        else:
            return (len(self.dataset) - 1) // self.batch_size + 1


class CrossLinkPredictionValidDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode

        '''prepare data for validation and testing'''
        self.original_valid = self.build_facts()
        self.valid = deepcopy(self.original_valid)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        ele = self.valid[idx]
        fact, label, source_id = torch.LongTensor(ele['fact']), ele['label'], ele['source_id']

        label = self.get_label(list(label), source_id)
        return fact[0], fact[1], fact[2], label.float(), self.mode

    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        o = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return s, r, o, label, mode

    def get_label(self, label, source_id):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        y = np.zeros([self.kg.num_ent], dtype=np.float32)
        if source_id == 0:
            y[len(self.kg.source[0].entities):] = 1.0
        else:
            y[:len(self.kg.source[0].entities)] = 1.0
        for e2 in label: y[e2] = 1.0

        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid = []
        for source_id, source in self.kg.source.items():
            for fact in source.valid:
                s, r, o = fact
                if self.mode == 'head-batch':
                    label = set(self.kg.sr2o_valid[(o, r+1)])
                elif self.mode == 'tail-batch':
                    label = set(self.kg.sr2o_valid[(s, r)])
                valid.append({'fact': (s, r, o), 'label': label, 'source_id': source_id})
        return valid

    def add_facts_using_relations(self, same, inverse):
        new_valid = []
        for source_id, source in self.kg.source.items():
            for fact in source.valid:
                s, r, o = fact
                if r in same:
                    if self.mode == 'head-batch':
                        label = set(self.kg.sr2o_valid[(o, r + 1)])
                    elif self.mode == 'tail-batch':
                        label = set(self.kg.sr2o_valid[(s, r)])
                    new_valid.append({'fact': (s, same[r], o), 'label': label, 'source_id': source_id})
                if r in inverse:
                    if self.mode == 'head-batch':
                        label = set(self.kg.sr2o_valid[(o, r + 1)])
                    elif self.mode == 'tail-batch':
                        label = set(self.kg.sr2o_valid[(s, r)])
                    new_valid.append({'fact': (o, inverse[r], s), 'label': label, 'source_id': source_id})
        self.valid = self.original_valid + new_valid


class CrossLinkPredictionTestDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode

        '''prepare data for validation and testing'''
        self.test = self.build_facts()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        ele = self.test[idx]
        fact, label, source_id, known = torch.LongTensor(ele['fact']), ele['label'], ele['source_id'], ele['known']
        type = torch.LongTensor([ele['type']])
        label = self.get_label(list(label), known)
        return fact[0], fact[1], fact[2], label.float(), self.mode, type

    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        o = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        types = torch.stack([_[5] for _ in data], dim=0)
        return s, r, o, label, mode, types

    def get_label(self, label, known):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        if known:
            y = np.ones([self.kg.num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.kg.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        test = []
        inner_test = set(self.kg.inner_test)
        for source_id, source in self.kg.source.items():
            for fact in source.test:
                if fact in self.kg.expand_train:
                    continue
                else:
                    known = False
                s, r, o = fact
                if self.mode == 'tail-batch':
                    label = set(self.kg.sr2o_test[(s, r)])
                else:
                    label = set(self.kg.sr2o_test[(o, r+1)])
                if (s, r, o) in inner_test:
                    type = -1
                else:
                    type = 1
                test.append({'fact': (s, r, o), 'label': label, 'source_id': source_id, 'known': known, 'type': type})
        return test


class RPGTrainDataset(Dataset):
    def __init__(self, args, kg, mode='head-batch', neg_ratio=0):
        self.args = args
        self.kg = kg
        self.mode = mode
        self.neg_ratio = neg_ratio

        self.original_facts, self.original_confidence = self.build_facts()
        self.original_confidence = torch.FloatTensor(self.original_confidence).reshape(-1, 1)

        self.facts = deepcopy(self.original_facts)
        self.confidence = self.original_confidence.clone()
        self.mapping_weights = torch.ones_like(self.confidence, dtype=torch.float32)

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        '''
        :param idx: idx of the training fact
        :return: a positive facts and its negative facts
        '''
        fact = self.facts[idx]
        conf = self.confidence[idx]
        mapping_weights = self.mapping_weights[idx]


        '''negative sampling'''
        fact, label, subsampling_weight = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, conf, self.mode, subsampling_weight, mapping_weights

    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        conf = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        subsampling_weight = torch.cat([_[4] for _ in data], dim=0)
        mapping_weight = torch.cat([_[5] for _ in data], dim=0)

        return fact[:,0], fact[:,1], fact[:,2], label, conf, mode, subsampling_weight, mapping_weight

    @timing_decorator
    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        head_facts = list()
        confidence = list()
        for fact in self.kg.RPG_edge:
            s, r, o, c = fact
            if r % 2 == 0:
                head_facts.append((s, r, o))
                confidence.append(c)
        return head_facts, confidence

    def corrupt(self, fact):
        s, r, o = fact[0], fact[1], fact[2]
        s_temp, o_temp, r_temp = s, o, r
        subsampling_weight = self.kg.RPG_count[(s, r)] + self.kg.RPG_count[(o, self.kg.relation2inv[r])]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        facts = [(s, r, o)]
        label = [1]
        negative_sample_size = 0
        negative_sample_list = []

        if self.neg_ratio <= 0:
            return facts, label, subsampling_weight

        while negative_sample_size < self.neg_ratio:
            negative_sample = np.random.randint(0, self.kg.num_rel, self.neg_ratio * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.kg.RPG_sr2o[(o_temp, self.kg.relation2inv[r_temp])],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.kg.RPG_sr2o[(s_temp, r_temp)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_size += negative_sample.size
            negative_sample_list.append(negative_sample)

        negative_sample = np.concatenate(negative_sample_list)[:self.neg_ratio]
        # Concatenate negative samples, then merge them into the facts while updating the labels.
        if self.mode == 'head-batch':
            facts.extend([(negative_sample[i], r, o) for i in range(self.neg_ratio)])
            label += [-1 for i in range(self.neg_ratio)]
        elif self.mode == 'tail-batch':
            facts.extend([(s, r, negative_sample[i]) for i in range(self.neg_ratio)])
            label += [-1 for i in range(self.neg_ratio)]

        return facts, label, subsampling_weight

    def update_facts(self, facts):
        head_facts = list()
        confidence = list()
        visited_facts = dict()
        facts_mapping_num = list()
        num = 0
        for fact in facts:
            s, r, o, c = fact
            if r % 2 == 0:
                if (s, r, o) not in visited_facts:
                    head_facts.append((s, r, o))
                    confidence.append(c)
                    facts_mapping_num.append(1)
                    visited_facts[(s, r, o)] = num
                    num+=1
                else:
                    idx = visited_facts[(s, r, o)]
                    facts_mapping_num[idx] += 1
                    confidence[idx] = max(confidence[idx], c)
        self.facts = head_facts
        self.confidence = torch.FloatTensor(confidence).reshape(-1, 1)
        self.mapping_weights = torch.FloatTensor(facts_mapping_num).reshape(-1, 1)
        return head_facts, confidence, facts_mapping_num