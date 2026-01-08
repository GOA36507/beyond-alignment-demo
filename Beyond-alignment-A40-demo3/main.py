import argparse
from src.train import *
from src.test import *
import sys
torch.autograd.set_detect_anomaly(True)
import shutil
from datetime import datetime
from src.data_load.KnowledgeGraph import *
from src.model.controller import Controller


class experiment:
    def __init__(self, args):
        self.args = args

        '''1. prepare data file path, model saving path and log path'''
        self.prepare()

        '''2. load data'''
        self.kg = KnowledgeGraph(args)

        '''3. create model and optimizer'''
        self.model, self.optimizer = self._create_model()
        self.start_epoch = 0

        if self.args.load_checkpoint is not None:
            self.start_epoch = self.load_checkpoint(os.path.join(self.args.load_checkpoint, 'model_best.tar'))
            self.model.args = self.args
            self.model.kg = self.kg
        self.args.logger.info(self.args)

    def _create_model(self):
        '''
        Initialize KG embedding model and optimizer.
        return: model, optimizer
        '''
        model = Controller(self.args, self.kg)
        model.to(self.args.device)
        init_param(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model, optimizer

    def train(self):
        '''
        Training process
        :return: training time
        '''
        start_time = time.time()
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)
        # [RPG 已禁用] filler = RPGFiller(self.args, self.kg, self.model)

        print("Start Training ===============================>")
        '''Training iteration'''
        for epoch in range(self.start_epoch, int(self.args.epoch_num)):
            # [RPG 已禁用] RPG 相关的训练循环逻辑
            # if self.args.RPG and epoch >= self.args.warmup and (epoch-self.args.warmup) % self.args.RPG_update_span==0:
            #     same, inverse = filler.fill_cross_KG_part()
            #     if self.args.use_augment:
            #         trainer.train_processor.add_facts_using_relations(same, inverse)
            #
            #     if epoch == self.args.warmup:
            #         self.best_valid = 0
            
            self.args.epoch = epoch
            '''training'''
            loss, valid_res, num_known_pairs, num_pseudo_pairs, triple_stats = trainer.run_epoch()
            '''early stop'''
            if self.best_valid <= valid_res[self.args.valid_metrics]:
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = max(0, self.stop_epoch-5)
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info('Early Stopping! Epoch: {} Best Results: {}'.format(epoch, round(self.best_valid*100, 3)))
                    break
            '''logging'''
            if epoch % 1 == 0:
                # 构建对齐对数量信息
                align_info = ''
                if self.args.use_known_alignment:
                    align_info += '\tKnown_Align_Pairs:{}'.format(num_known_pairs)
                if self.args.use_pseudo_alignment:
                    align_info += '\tPseudo_Align_Pairs:{}'.format(num_pseudo_pairs)
                # 添加三元组统计信息
                triple_info = ''
                if triple_stats:
                    triple_info += '\tOriginal_Triples:{}'.format(triple_stats.get('num_original', 0))
                    triple_info += '\tKnown_Expand_Triples:{}'.format(triple_stats.get('num_known_expand', 0))
                    triple_info += '\tPseudo_Expand_Triples:{}'.format(triple_stats.get('num_pseudo_expand', 0))
                    # 无论是否使用图间三元组，都输出图结构统计信息（用于调试）
                    triple_info += '\tGraph_Original:{}'.format(triple_stats.get('graph_original', 0))
                    triple_info += '\tGraph_Known_Expand:{}'.format(triple_stats.get('graph_known_expand', 0))
                    triple_info += '\tGraph_Pseudo_Expand:{}'.format(triple_stats.get('graph_pseudo_expand', 0))
                self.args.logger.info('Epoch:{}\tLoss:{}\tH@1:{}\tH@3:{}\tH@5:{}\tH@10:{}\tMRR:{}\tBest:{}{}{}'.format(epoch,round(loss, 3), round(valid_res['hits1'] * 100, 2), round(valid_res['hits3'] * 100, 2), round(valid_res['hits5'] * 100, 2), round(valid_res['hits10'] * 100, 2), round(valid_res['mrr'] * 100, 2), round(self.best_valid * 100,2), align_info, triple_info))
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def test(self, load_best=True):
        self.kg.load_test()
        if load_best and self.args.load_checkpoint is None:
            best_checkpoint = os.path.join(self.args.save_path, 'model_best.tar')
            self.load_checkpoint(best_checkpoint)
        tester = Tester(self.args, self.kg, self.model)

        res = tester.test()
        print(res)
        return res

    def prepare(self):
        '''
        set the log path, the model saving path and device
        :return: None
        '''
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        args.lambda_1 = float(args.lambda_1)
        args.lambda_2 = float(args.lambda_2)
        args.alpha = float(args.alpha)

        '''set data path'''
        self.args.data_path = args.data_path + args.dataset + '/'
        self.args.save_path = args.save_path + args.dataset + '-' + args.scorer + '-' + args.encoder +'-'+ str(args.emb_dim)+'-' + str(args.margin)

        '''add logging implement to model path for ablation_study'''
        # if self.args.ea_expand_training:
        #     self.args.save_path = self.args.save_path + '-ea_expand_training'
        # [RPG 已禁用] RPG 相关的路径配置
        # if self.args.RPG:
        #     self.args.save_path = self.args.save_path + '-RPG'
        #     if not self.args.use_attn:
        #         self.args.save_path=self.args.save_path + '-wo attn'
        #     if not self.args.use_RPG_triple:
        #         self.args.save_path=self.args.save_path + '-wo triple'
        #     if not self.args.use_augment:
        #         self.args.save_path=self.args.save_path + '-wo augment'

        if self.args.use_known_alignment:
            self.args.save_path = self.args.save_path + '-known_align' + str(self.args.known_align_weight)
        if self.args.use_pseudo_alignment:
            self.args.save_path = self.args.save_path + '-pseudo_align' + str(self.args.pseudo_align_weight) + '-pseudo_align_threshold' + str(self.args.pseudo_align_threshold)
        if self.args.use_pseudo_triple_loss:
            self.args.save_path = self.args.save_path + '-pseudo_triple_weight' + str(self.args.pseudo_triple_weight)
        
        
        # 添加选择策略信息
        if getattr(self.args, 'use_entropy_selection', False):
            self.args.save_path = self.args.save_path + '-entropy' + str(self.args.entropy_threshold) 
        else:
            self.args.save_path = self.args.save_path + '-' + str(self.args.pseudo_triple_selection_ratio)

        # 添加 InfoNCE 损失配置信息到模型路径
        if getattr(self.args, 'use_triple_infonce', False):
            self.args.save_path = self.args.save_path + '-infonce' + str(self.args.triple_infonce_weight) + '-temp' + str(self.args.triple_infonce_temperature)
            # 添加伪对齐对 InfoNCE 配置信息
            if getattr(self.args, 'use_pseudo_triple_infonce', False):
                self.args.save_path = self.args.save_path + '-pseudo_infonce' + str(getattr(self.args, 'pseudo_triple_infonce_weight', 0.1))
    
        self.args.save_path = self.args.save_path + '-' + str(self.args.ea_rate) + '--' + str(self.args.learning_rate)+ '-' + str(args.seed) + '-neg_ratio-' + str(args.neg_ratio)
        
        # 添加图结构配置信息到模型路径
        # if self.args.use_inter_triples:
        #     self.args.save_path = self.args.save_path + '-use_inter'
        
        # 添加 InfoNCE 损失配置信息到模型路径
        # if getattr(self.args, 'use_triple_infonce', False):
        #     self.args.save_path = self.args.save_path + '-infonce' + str(self.args.triple_infonce_weight) + '-temp' + str(self.args.triple_infonce_temperature)
        #     # 添加伪对齐对 InfoNCE 配置信息
        #     if getattr(self.args, 'use_pseudo_triple_infonce', False):
        #         self.args.save_path = self.args.save_path + '-pseudo_infonce' + str(getattr(self.args, 'pseudo_triple_infonce_weight', 0.1))
        if self.args.note != '':
            self.args.save_path = self.args.save_path + self.args.note

        if os.path.exists(args.save_path) and args.load_checkpoint is None:
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        self.args.log_path = args.log_path + datetime.now().strftime('%Y%m%d/')
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + args.dataset + '-' + args.scorer + '-' + args.encoder +'-'+ str(args.emb_dim)+ '-' + str(args.margin)

        # '''add logging implement to log path for ablation_study'''
        # if self.args.ea_expand_training:
        #     self.args.log_path = self.args.log_path + '-ea_expand_training'
        # [RPG 已禁用] RPG 相关的日志路径配置
        # if self.args.RPG:
        #     self.args.log_path = self.args.log_path + '-RPG'
        #     if not self.args.use_attn:
        #         self.args.log_path=self.args.log_path + '-wo attn'
        #     if not self.args.use_RPG_triple:
        #         self.args.log_path=self.args.log_path + '-wo triple'
        #     if not self.args.use_augment:
        #         self.args.log_path=self.args.log_path + '-wo augment'
        # 添加对齐配置信息到日志路径
        if self.args.use_known_alignment:
            self.args.log_path = self.args.log_path + '-known_align' + str(self.args.known_align_weight)
        if self.args.use_pseudo_alignment:
            self.args.log_path = self.args.log_path + '-pseudo_align' + str(self.args.pseudo_align_weight) + '-pseudo_align_threshold' + str(self.args.pseudo_align_threshold)
        if self.args.use_pseudo_triple_loss:
            self.args.log_path = self.args.log_path + '-pseudo_triple_weight' + str(self.args.pseudo_triple_weight)
        
        
        # 添加选择策略信息
        if getattr(self.args, 'use_entropy_selection', False):
            self.args.log_path = self.args.log_path  + '-entropy' + str(self.args.entropy_threshold) 
        else:
            self.args.log_path = self.args.log_path  + '-' + str(self.args.pseudo_triple_selection_ratio)
        
        if getattr(self.args, 'use_triple_infonce', False):
            self.args.log_path = self.args.log_path + '-infonce' + str(self.args.triple_infonce_weight) + '-temp' + str(self.args.triple_infonce_temperature)
            # 添加伪对齐对 InfoNCE 配置信息
            if getattr(self.args, 'use_pseudo_triple_infonce', False):
                self.args.log_path = self.args.log_path + '-pseudo_infonce' + str(getattr(self.args, 'pseudo_triple_infonce_weight', 0.1))


        self.args.log_path = self.args.log_path + '-' + str(self.args.ea_rate) + '--' +str(self.args.learning_rate) + '-' + str(args.seed) + '-neg_ratio-' + str(args.neg_ratio)
        
        # 添加图结构配置信息到日志路径
        # if self.args.use_inter_triples:
        #     self.args.log_path = self.args.log_path + '-use_inter'
        

        # if getattr(self.args, 'use_triple_infonce', False):
        #     self.args.log_path = self.args.log_path + '-infonce' + str(self.args.triple_infonce_weight) + '-temp' + str(self.args.triple_infonce_temperature)
        #     # 添加伪对齐对 InfoNCE 配置信息
        #     if getattr(self.args, 'use_pseudo_triple_infonce', False):
        #         self.args.log_path = self.args.log_path + '-pseudo_infonce' + str(getattr(self.args, 'pseudo_triple_infonce_weight', 0.1))
        
        '''add additional note to log name'''
        if self.args.note != '':
            self.args.log_path = self.args.log_path + self.args.note

        '''set logger'''
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = args.log_path + '.txt'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        '''set device'''
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device

    def save_model(self, is_best=False, name=''):
        '''
        Save trained model.
        :param is_best: If True, save it as the best model.
        After training on each snapshot, we will use the best model to evaluate.
        '''
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.model.state_dict()
        checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint_dict['epoch_id'] = self.args.epoch

        if is_best:
            self.args.logger.info('Saving Best Model to {}/model_best.tar'.format(self.args.save_path))
            out_tar = os.path.join(self.args.save_path, 'model_best.tar')
            torch.save(checkpoint_dict, out_tar)
            # [RPG 已禁用] 保存 attention weight
            # if self.args.RPG and self.args.use_attn:
            #     atten_weight_path = os.path.join(self.args.save_path, 'attn_weight_best.npy')
            #     self.kg.best_attention_weight = deepcopy(self.kg.attention_weight)
            #     np.save(atten_weight_path, self.kg.best_attention_weight.cpu().detach().numpy())
        if name != '':
            out_tar = os.path.join(name)
            torch.save(checkpoint_dict, out_tar)
    def load_checkpoint(self, input_file):
        if os.path.isfile(os.path.join(os.getcwd(), input_file)):
            logging.info('=> loading checkpoint \'{}\''.format(os.path.join(os.getcwd(), input_file)))
            checkpoint = torch.load(os.path.join(os.getcwd(), input_file), map_location="cuda:{}".format(self.args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if '-wo attn' not in input_file and 'erge' not in input_file:
                attn_path = os.path.join(os.getcwd(), input_file)[:-15]
                attn_file = os.path.join(attn_path, 'attn_weight_best.npy')
                if os.path.exists(attn_file):  # 添加文件存在性检查
                    self.kg.best_attention_weight = torch.tensor(np.load(attn_file)).to(self.args.device)
                else:
                    logging.warning('Attention weight file not found: {}'.format(attn_file))
            return int(checkpoint['epoch_id']) + 1
        else:
            logging.info('=> no checkpoint found at \'{}\''.format(input_file))
    # def load_checkpoint(self, input_file):
    #     if os.path.isfile(os.path.join(os.getcwd(), input_file)):
    #         logging.info('=> loading checkpoint \'{}\''.format(os.path.join(os.getcwd(), input_file)))
    #         checkpoint = torch.load(os.path.join(os.getcwd(), input_file), map_location="cuda:{}".format(self.args.gpu))
    #         self.model.load_state_dict(checkpoint['state_dict'])
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         if '-wo attn' not in input_file and 'erge' not in input_file:
    #             attn_path = os.path.join(os.getcwd(), input_file)[:-15]
    #             self.kg.best_attention_weight = torch.tensor(np.load(os.path.join(attn_path, 'attn_weight_best.npy'))).to(self.args.device)
    #             return int(checkpoint['epoch_id']) + 1
    #     else:
    #         logging.info('=> no checkpoint found at \'{}\''.format(input_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # training control
    parser.add_argument('-dataset', dest='dataset', default='WIKI-YAGO', help='dataset name, DBP-FB, WIKI-YAGO')
    parser.add_argument('-load_checkpoint', dest='load_checkpoint', default=None, help='./model_best.tar')

    # base setting
    parser.add_argument('-optimizer_name', dest='optimizer_name', default='Adam')
    parser.add_argument('-epoch_num', dest='epoch_num', default=1000, help='max epoch num')
    parser.add_argument('-batch_size', dest='batch_size', default=2048, help='Mini-batch size')
    parser.add_argument('-test_batch_size', dest='test_batch_size', default=100, help='Mini-batch size')
    parser.add_argument('-learning_rate', dest='learning_rate', default=0.0005)
    parser.add_argument('-emb_dim', dest='emb_dim', default=256, help='embedding dimension')
    parser.add_argument('-l2', dest='l2', default=0.0, help='optimizer l2')

    parser.add_argument('-patience', dest='patience', default=5, help='early stop step')
    parser.add_argument('-neg_ratio', dest='neg_ratio', default=256)
    parser.add_argument('-margin', dest='margin', default=9.0, help='')
    parser.add_argument('-gpu', dest='gpu', default=0)

    parser.add_argument('-encoder', dest='encoder', default='lookup_gat', help='lookup, lookup_attn,lookup_gat')
    parser.add_argument('-scorer', dest='scorer', default='TransE', help='')

    # for ea
    parser.add_argument('-ea_rate', dest='ea_rate', default='0.3', help='')
    parser.add_argument('-RPG', dest='RPG', default='False', help='')
    parser.add_argument('-ea_expand_training', dest='ea_expand_training', default='True', help='')

    '''Ablation Study'''
    parser.add_argument('-use_attn', dest='use_attn', default='True', help='')
    parser.add_argument('-use_RPG_triple', dest='use_RPG_triple', default='True', help='')
    parser.add_argument('-use_augment', dest='use_augment', default='True', help='')
    
    '''Entity Alignment Loss'''
    parser.add_argument('-use_known_alignment', dest='use_known_alignment', default='True', help='是否使用已知实体对的对齐损失')
    parser.add_argument('-use_pseudo_alignment', dest='use_pseudo_alignment', default='True', help='是否使用伪对齐对的对齐损失')
    parser.add_argument('-known_align_weight', dest='known_align_weight', default=0.1, type=float, help='已知对齐损失的权重')
    parser.add_argument('-pseudo_align_weight', dest='pseudo_align_weight', default=0.1, type=float, help='伪对齐损失的权重')
    parser.add_argument('-pseudo_align_threshold', dest='pseudo_align_threshold', default=0.01, type=float, help='伪对齐对的置信度阈值λ')
    
    '''Pseudo Triple Loss'''
    parser.add_argument('-use_pseudo_triple_loss', dest='use_pseudo_triple_loss', default='True', type=str, help='是否使用分离的伪三元组损失 (True/False)')
    parser.add_argument('-pseudo_triple_weight', dest='pseudo_triple_weight', default=0.1, type=float, help='伪三元组损失的权重（默认1.0，与原始和已知扩展三元组相同）')
    
    '''Triple InfoNCE Loss'''
    parser.add_argument('-use_triple_infonce', dest='use_triple_infonce', default='True', type=str, help='是否使用基于扩展三元组的 InfoNCE 损失 (True/False)')
    parser.add_argument('-triple_infonce_temperature', dest='triple_infonce_temperature', default=0.07, type=float, help='InfoNCE 损失的温度参数')
    parser.add_argument('-triple_infonce_weight', dest='triple_infonce_weight', default=0.1, type=float, help='InfoNCE 损失的权重（用于已知对齐对的扩展三元组）')
    parser.add_argument('-use_pseudo_triple_infonce', dest='use_pseudo_triple_infonce', default='True', type=str, help='是否对基于伪对齐对的扩展三元组计算 InfoNCE 损失 (True/False)')
    parser.add_argument('-pseudo_triple_infonce_weight', dest='pseudo_triple_infonce_weight', default=0.1, type=float, help='伪对齐对扩展三元组的 InfoNCE 损失权重')
    
    '''Pseudo Triple Selection Strategy'''
    parser.add_argument('-pseudo_triple_selection', dest='pseudo_triple_selection', default='score_high', type=str, 
                        choices=['random', 'score_high', 'score_low', 'degree_high', 'degree_low'],
                        help='伪三元组选择策略: random(随机), score_high(得分高到低), score_low(得分低到高), degree_high(出入度高到低), degree_low(出入度低到高)')
    
    # 基于熵的top-p筛选策略参数
    parser.add_argument('-use_entropy_selection', dest='use_entropy_selection', default='True', type=str,
                        help='是否使用基于熵的top-p筛选策略 (True/False)')
    parser.add_argument('-entropy_n_sigma', dest='entropy_n_sigma', default=1.0, type=float,
                        help='截断高斯分布的sigma倍数（用于计算概率）')
    parser.add_argument('-entropy_threshold', dest='entropy_threshold', default=10.0, type=float,
                        help='熵累加的阈值，超过此值停止选择伪三元组')
    
    # 保留兼容性：如果未使用熵筛选，仍可使用固定比例
    parser.add_argument('-pseudo_triple_selection_ratio', dest='pseudo_triple_selection_ratio', default=1.0, type=float, 
                        help='三元组选择比例k%（0.0-1.0），仅在未使用熵筛选时生效')

    '''RPG'''
    parser.add_argument('-topk', dest='topk', default=3, help='')
    parser.add_argument('-lambda_1', dest='lambda_1', default=0.7, help='')
    parser.add_argument('-lambda_2', dest='lambda_2', default=0.3, help='')
    parser.add_argument('-warmup', dest='warmup', default=10)
    parser.add_argument('-RPG_update_span', dest='RPG_update_span', default=5)
    parser.add_argument('-alpha', dest='alpha', default=1.0)
    
    '''Graph Structure Update'''
    parser.add_argument('-use_inter_triples', dest='use_inter_triples', default='True', type=str,
                        help='是否使用图间三元组进行跨图聚合 (True/False)')

    # others
    parser.add_argument('-save_path', dest='save_path', default='./checkpoint/')
    # parser.add_argument('-data_path', dest='data_path', default='./dataset/data/')
    parser.add_argument('-data_path', dest='data_path', default='CrossLPData/')
    parser.add_argument('-root_dir', dest='root_dir', default='CrossLPData/')
    # parser.add_argument('-root_dir', dest='root_dir', default='yncui-nju/CrossLPData/')
    parser.add_argument('-log_path', dest='log_path', default='./logs/')
    parser.add_argument('-num_workers', dest='num_workers', default=10)
    parser.add_argument('-seed', dest='seed', default=2024)
    parser.add_argument('-valid_metrics', dest='valid_metrics', default='mrr')
    parser.add_argument('-note', dest='note', default='develop', help='The note of log file name')
    args = parser.parse_args()
    retype_parameters(args)
    same_seeds(args.seed)

    # [RPG 已禁用] 强制禁用所有 RPG 相关功能
    args.RPG = False
    args.use_augment = False
    args.use_attn = False
    args.use_RPG_triple = False
    # if not args.RPG:
    #     args.use_augment = False
    #     args.use_attn = False
    #     args.use_RPG_triple = False
    # if args.use_attn:
    #     args.encoder = 'lookup_gat'

    args.source_list = args.dataset.split('-')
    E = experiment(args)
    E.train()
    E.test()
