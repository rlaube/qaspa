from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
import nengo_spa as spa

ALG_NAME_TO_NENGO_ALG = {
    'hrr': spa.algebras.HrrAlgebra(),
    'vtb': spa.algebras.VtbAlgebra(),
    'tvtb': spa.algebras.TvtbAlgebra(),
    'vtb': spa.algebras.VtbAlgebra(),
}

class QASPA(nn.Module):
    def __init__(self, device, encoder_only, k, skip_type, skip_placement, algebra, qa_context, sent_trans, sent_dim, 
                 concept_dim, fc_dim, sp_hidden_dim, sp_output_dim, 
                 n_fc_layer, p_sp, p_fc, sp_layer_norm,
                 pretrained_concept_emb, qa_emb, permute_vec,
                 init_range, normalize_graphs, normalize_embeddings, score_mlp):
        super().__init__()
        self.init_range = init_range

        self.encoder_only = encoder_only

        self.normalize_embeddings = normalize_embeddings
        
        if encoder_only:
            self.fc = MLP(sp_output_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True, score_mlp=True)
            self.dropout_fc = nn.Dropout(p_fc)
        else:
            self.algebra = ALG_NAME_TO_NENGO_ALG[algebra]
            self.qa_context = qa_context
            self.normalize_graphs = normalize_graphs
            
            self.concept_emb = torch.from_numpy(pretrained_concept_emb)
            self.qa_emb = qa_emb

            if self.normalize_embeddings:
                for i, concept in enumerate(self.concept_emb):
                    norm = np.linalg.norm(concept)
                    if norm > 0:
                       self.concept_emb[i] = self.concept_emb[i] / norm
                self.qa_emb['ISQUESTIONCONCEPT'] = torch.from_numpy(self.qa_emb['ISQUESTIONCONCEPT'] / np.linalg.norm(self.qa_emb['ISQUESTIONCONCEPT']))
                self.qa_emb['ISANSWERCONCEPT'] = torch.from_numpy(self.qa_emb['ISANSWERCONCEPT'] / np.linalg.norm(self.qa_emb['ISANSWERCONCEPT']))

                # for i, concept in enumerate(self.concept_emb):
                #     if np.linalg.norm(concept) > 0:
                #        self.concept_emb[i] = self.algebra.make_unitary(concept)
                # self.qa_emb['ISQUESTIONCONCEPT'] = self.algebra.make_unitary(self.qa_emb['ISQUESTIONCONCEPT'])
                # self.qa_emb['ISANSWERCONCEPT'] = self.algebra.make_unitary(self.qa_emb['ISANSWERCONCEPT'])
            
            self.concept_emb = self.concept_emb.to(device)
            self.qa_emb['ISQUESTIONCONCEPT'] = self.qa_emb['ISQUESTIONCONCEPT'].to(device)
            self.qa_emb['ISANSWERCONCEPT'] = self.qa_emb['ISANSWERCONCEPT'].to(device)

            # project to same dimension of sp (semantic pointer) MLP ouptut
            self.sent_trans = sent_trans
            if sent_trans:
                self.svec2nvec = nn.Linear(sent_dim, sp_output_dim)
                self.activation = GELU()

            # define ff layers applied to sp graphs
            self.score_mlp = score_mlp # do True if I want original dropout (if True, MLP block doesn't have dropout, norm, or activation function in last layer)
            if skip_type == 0:
                self.spa = MLP(concept_dim, sp_hidden_dim, sp_output_dim, k, p_sp, layer_norm=sp_layer_norm, score_mlp=self.score_mlp)
            elif skip_type == 1:
                self.spa = MLP_Residual1(concept_dim, sp_hidden_dim, sp_output_dim, k, p_sp, layer_norm=sp_layer_norm, skip_placement=skip_placement, score_mlp=self.score_mlp)
            elif skip_type == 2:
                self.spa = MLP_Residual2(concept_dim, sp_hidden_dim, sp_output_dim, k, p_sp, layer_norm=sp_layer_norm, skip_placement=skip_placement, score_mlp=self.score_mlp)
            elif skip_type == 3:
                self.spa = MLP_Residual3(concept_dim, sp_hidden_dim, sp_output_dim, k, p_sp, layer_norm=sp_layer_norm, skip_placement=skip_placement, score_mlp=self.score_mlp)
            self.fc = MLP(sent_dim + sp_output_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True, score_mlp=True)

            # refactored dropout
            self.dropout_fc = nn.Dropout(p_fc)

            self.permute_vec = permute_vec

        if init_range > 0:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def hrr_bind(self, tensor1, tensor2):
        return torch.fft.irfft(torch.fft.rfft(tensor1) * torch.fft.rfft(tensor2))

    def forward(self, sent_vecs, graph_sp, concept_ids, node_type_ids):
        """
        sent_vecs: (batch_size * num_choice, dim_sent)
        graph_sp: (batch_size * num_choice, concept_dim)
        concept_ids: (batch_size * num_choice, max # of nodes)
        node_type_ids: (batch_size * num_choice, max # of nodes)

        returns: (batch_size * num_choice, 1)
        """
        
        if self.encoder_only:
            concat = self.dropout_fc(sent_vecs)
            logits = self.fc(concat)
        else:
            sent_input = self.activation(self.svec2nvec(sent_vecs)) if self.sent_trans else sent_vecs #(batch_size, dim_sent)
        
            if self.qa_context:
                # if concepts are normalized, should normalize qa_context too (should it be normalized in final layer??)
                if self.normalize_embeddings:
                    sent_input = F.normalize(sent_input, p=2, dim=-1)

                # sent_input_copy = sent_input.detach().cpu().numpy()
                # graph_sp_copy = graph_sp.detach().cpu().numpy()
                
                # if self.normalize_embeddings:
                #     for i in range(sent_input_copy.shape[0]):
                #         sent_input_copy[i] = self.algebra.make_unitary(sent_input_copy[i])
                #         # sent_input[i] = self.make_unitary(sent_input[i])

                # if there is a permutation vector, permute the tail concept of each triple
                if self.permute_vec is None:
                    # iterate through the batch
                    for i in range(graph_sp.shape[0]):
                        q_concept_ids = concept_ids[i][node_type_ids[i] == 0]
                        a_concept_ids = concept_ids[i][node_type_ids[i] == 1]

                        # bind the qa context to each question and answer entity with 2 unique relation embeddings
                        for q_id in q_concept_ids.tolist():
                            # graph_sp_copy[i] += self.algebra.bind(
                            #                             self.algebra.bind(self.concept_emb[q_id], self.qa_emb['ISQUESTIONCONCEPT']),
                            #                             sent_input_copy[i]
                            #                         )
                            graph_sp[i] += self.hrr_bind(
                                                self.hrr_bind(self.concept_emb[q_id], self.qa_emb['ISQUESTIONCONCEPT']), 
                                                sent_input[i]
                                            )
                        for a_id in a_concept_ids.tolist():
                            # graph_sp_copy[i] += self.algebra.bind(
                            #                             self.algebra.bind(self.concept_emb[a_id], self.qa_emb['ISANSWERCONCEPT']),
                            #                             sent_input_copy[i]
                            #                         )
                            graph_sp[i] += self.hrr_bind(
                                                self.hrr_bind(self.concept_emb[q_id], self.qa_emb['ISANSWERCONCEPT']), 
                                                sent_input[i]
                                            )
                else:
                    # iterate through the batch
                    for i in range(graph_sp.shape[0]):
                        q_concept_ids = concept_ids[i][node_type_ids[i] == 0]
                        a_concept_ids = concept_ids[i][node_type_ids[i] == 1]

                        # bind the qa context to each question and answer entity with 2 unique relation embeddings
                        for q_id in q_concept_ids.tolist():
                             graph_sp[i] += self.hrr_bind(
                                                        self.hrr_bind(self.concept_emb[q_id], self.qa_emb['ISQUESTIONCONCEPT']),
                                                        sent_input[i][self.permute_vec]
                                                    )
                        for a_id in a_concept_ids.tolist():
                            graph_sp[i] += self.hrr_bind(
                                                        self.hrr_bind(self.concept_emb[a_id], self.qa_emb['ISANSWERCONCEPT']),
                                                        sent_input[i][self.permute_vec]
                                                    )
            
                # graph_sp = torch.from_numpy(graph_sp_copy).to(graph_sp.device)

            if self.normalize_graphs:
                graph_sp = F.normalize(graph_sp, p=2, dim=-1)

            spa_output = self.spa(graph_sp.to(torch.float32))

            # A: used in refactor
            concat = self.dropout_fc(torch.cat((sent_input, spa_output), 1))

            # # B: keep together
            # concat = torch.cat((sent_input, spa_output), 1) 
            # # assert(self.score_mlp)

            logits = self.fc(concat)

        return logits
    
    def make_unitary(tensor):
        fft_val = torch.fft.fft(tensor)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = torch.sqrt(fft_imag**2 + fft_real**2)
        invalid = fft_norms <= 0.0
        fft_val[invalid] = 1.0
        fft_norms[invalid] = 1.0
        fft_unit = fft_val / fft_norms
        return torch.fft.irfft(fft_unit, n=len(tensor))


class LM_QASPA(nn.Module):
    def __init__(self, model_name, encoder_only, skip_type, skip_placement, algebra, qa_context, sent_trans, k, 
                 pretrained_concept_emb, qa_emb, permute_vec,
                fc_dim, sp_hidden_dim, sp_output_dim, 
                n_fc_layer, p_sp, p_fc, sp_layer_norm, normalize_graphs, normalize_embeddings, score_mlp,
                device, concept_dim=1024,init_range=0.02, encoder_config={}):
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QASPA(device, encoder_only, k, skip_type, skip_placement, algebra, qa_context, sent_trans, 
                             self.encoder.sent_dim, concept_dim, fc_dim, sp_hidden_dim, sp_output_dim, 
                             n_fc_layer, p_sp, p_fc, sp_layer_norm,
                             pretrained_concept_emb, qa_emb, permute_vec,
                             init_range, normalize_graphs, normalize_embeddings, score_mlp)


    def forward(self, *inputs, layer_id=-1, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        
        #Here, merge the batch dimension and the num_choice dimension
        batched_lm_inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-3]]

        concept_ids = torch.cat(inputs[-2], dim=0)

        node_type_ids = torch.cat(inputs[-1], dim=0)
        
       
        graph_sp = inputs[-3].view(inputs[-3].size(0) * inputs[-3].size(1), inputs[-3].size(2))
        sent_vecs, all_hidden_states = self.encoder(*batched_lm_inputs, layer_id=layer_id)
        logits = self.decoder(sent_vecs.to(graph_sp.device), graph_sp, concept_ids, node_type_ids)
        logits = logits.view(bs, nc)
        if not detail:
            return logits
        else:
            return logits


class LM_QASPA_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path, train_sp,
                 dev_statement_path, dev_adj_path, dev_sp,
                 test_statement_path, test_adj_path, test_sp,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, debug=False, debug_sample_size = 32):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.debug = debug
        self.debug_sample_size = debug_sample_size

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
    
        self.train_graph_sp, *self.train_node_data = load_graph_sp(train_adj_path, max_node_num, num_choice, train_sp)
        self.dev_graph_sp, *self.dev_node_data = load_graph_sp(dev_adj_path, max_node_num, num_choice, dev_sp)

        # print(len(self.dev_qids))
        # print(self.dev_labels.size(0))
        # for data in self.dev_encoder_data:
        #     print(data.size(0))
        # print(torch.from_numpy(self.dev_graph_sp).size(0))

        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + [torch.from_numpy(self.train_graph_sp)])
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + [torch.from_numpy(self.dev_graph_sp)])

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            self.test_graph_sp, *self.test_node_data = load_graph_sp(test_adj_path, max_node_num, num_choice, test_sp)
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + [torch.from_numpy(self.test_graph_sp)])


        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_graph_sp = [x[:n_train] for x in self.train_graph_sp]
                assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + [torch.from_numpy(self.train_graph_sp)])
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.debug:
            train_indexes = torch.arange(self.debug_sample_size)
        elif self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=torch.from_numpy(self.train_graph_sp), node_data=self.train_node_data)

    def train_eval(self):
        return MultiGPUBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=torch.from_numpy(self.train_graph_sp), node_data=self.train_node_data)

    def dev(self):
        if self.debug:
            dev_indexes = torch.arange(self.debug_sample_size)
        else:
            dev_indexes = torch.arange(len(self.dev_qids))
        return MultiGPUBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, dev_indexes, self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=torch.from_numpy(self.dev_graph_sp), node_data=self.dev_node_data)

    def test(self):
        if self.debug:
            test_indexes = torch.arange(self.debug_sample_size)
        elif self.is_inhouse:
            test_indexes = self.inhouse_test_indexes
        else:
            test_indexes = torch.arange(len(self.test_qids))
        if self.is_inhouse:
            return MultiGPUBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=torch.from_numpy(self.train_graph_sp), node_data=self.train_node_data)
        else:
            return MultiGPUBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, test_indexes, self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=torch.from_numpy(self.test_graph_sp), node_data=self.test_node_data)