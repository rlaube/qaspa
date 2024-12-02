import random
from collections import defaultdict, OrderedDict
import numpy as np
import optuna
import logging
import wandb
import math

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from modeling.modeling_qaspa import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *
from utils import utils
import os, socket, subprocess, datetime


DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa': 1e-3,
}


def main(args):
    if args.mode == 'train':
        if args.hyperparameter_tuning == True:
            logger = logging.getLogger()    # Setup the root logger.
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.FileHandler("optuna_studies/" + args.study_name + ".log", mode="w"))
            optuna.logging.enable_propagation()  # Propagate logs to the root logger.

            storage_name = "sqlite:///{}.db".format("optuna_studies/" + args.study_name)
            
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                        # pruner=optuna.pruners.NopPruner(),
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
                                        study_name=args.study_name, direction='maximize', 
                                        storage=storage_name, load_if_exists=True)
            
            logger.info("Start optimization.")

            study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

            with open(args.study_name + ".log") as f:
                assert f.readline().startswith("A new study created")
                assert f.readline() == "Start optimization.\n"

            with open(args.study_name + ".pkl", "wb") as fout:
                pickle.dump(study, fout)
        else:
            val_acc = train(args)
    elif args.mode == 'eval_detail':
        raise NotImplementedError
        # eval_detail(args)
    else:
        raise ValueError('Invalid mode')


def objective(trial, args):
    cycles = trial.suggest_int('cycle', 1, 2)
    elr = trial.suggest_float('elr', 1e-6, 1e-4, log=True)
    dropoutf = trial.suggest_float('dropoutf', 0, 0.3, step=0.1)
    unfreeze_epoch = trial.suggest_int('unfreeze_epoch', 0, 10, step=5)
    refreeze_epoch = trial.suggest_int('refreeze_epoch', 20, 30, step=5)

    args.encoder_lr = elr
    args.cycles = cycles
    args.dropoutf = dropoutf
    args.unfreeze_epoch = unfreeze_epoch
    args.refreeze_epoch = refreeze_epoch

    args.run_name = "elr{:.2e}_cycles{}_dropf{:.1f}_unfrz{}_refrz{}".format(elr, cycles, dropoutf, unfreeze_epoch, refreeze_epoch)

    # bs = trial.suggest_categorical('bs', [32, 64, 128])
    # cycles = trial.suggest_int('cycle', 1, 2)
    # warmup_steps = trial.suggest_int('warmup_steps', 50, 200, step=50)
    # dlr = trial.suggest_float('dlr', 1e-5, 1e-3, log=True)
    # dropoutspa = trial.suggest_float('dropoutspa', 0.3, 0.7, step=0.1)
    # dropoutf = trial.suggest_float('dropoutf', 0, 0.3, step=0.1)

    # args.batch_size = bs
    # args.decoder_lr = dlr
    # args.cycles = cycles
    # args.warmup_steps = warmup_steps
    # args.dropoutspa = dropoutspa
    # args.dropoutf = dropoutf


    # sp_dir='../SPA-Embeddings/'+args.dataset+'/bert-concept_bert-rel/made-unitary/pruned/'+algebra+'/'
       
    # args.train_sp = sp_dir + 'train_graph_sp.npy'
    # args.dev_sp = sp_dir + 'dev_graph_sp.npy'
    # args.test_sp = sp_dir + 'test_graph_sp.npy'


    # args.run_name = "dlr{:.2e}_warm{}_cycle{}_bs{}_dropsp{:.1f}_dropf{:.1f}".format(dlr, warmup_steps, cycles, bs, dropoutspa, dropoutf)



    val_acc = train(args, trial)
    return val_acc

def train(args,trial=None):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    model_path = os.path.join(args.save_dir, 'model.pt')
    check_path(model_path)
    if not args.hyperparameter_tuning:
        config_path = os.path.join(args.save_dir, 'config.json')
        log_path = os.path.join(args.save_dir, 'log.csv')
        export_config(args, config_path)
        with open(log_path, 'w') as fout:
            fout.write('step,dev_acc,test_acc\n')

    # set up wandb 
    if not args.use_wandb:
        wandb_mode = "disabled"
    elif args.debug:
        wandb_mode = "offline"
    else:
        wandb_mode = "online"

    wandb_dir = os.path.join(os.getcwd(), args.save_dir)
    if args.dataset == 'csqa':
        project_name = 'QASPA'
    elif args.dataset == 'obqa':
        project_name = 'QASPA-OBQA'
    elif args.dataset == 'medqa':
        project_name = 'QASPA-MedQA'
    if args.study_name is None:
        wandb.init(project=project_name, entity="ryan-laube", config=args, dir=wandb_dir, name=args.run_name, mode=wandb_mode)
    else:
        wandb.init(project=project_name, entity="ryan-laube", config=args, dir=wandb_dir, name=args.run_name, group=args.study_name, mode=wandb_mode, reinit=True)
        
    print(socket.gethostname())
    print ("pid:", os.getpid())
    print ("conda env:", os.environ.get('CONDA_DEFAULT_ENV'))
    utils.print_cuda_info()
    wandb.run.log_code('.')

    ###################################################################################################
    #   Load data (concept embeddings and permutation vec)                                            #                                         
    ###################################################################################################
    # define 2 random 1024-dimensional vectors for IsAnswerConcept and IsQuestionConcept (to bind qa context to qa graph)
    with open("data/cpnet/normalized_cpnet_vocab_sp_QA.pkl", 'rb') as f:
        qa_emb = pickle.load(f)
        
    cpnet_emb = np.load(args.cpnet_emb_path)

    concept_dim = cpnet_emb.shape[-1]
    if args.permute_vec_path:
        permute_vec = np.load(args.permute_vec_path)
    else:
        permute_vec = None

    # try:
    if True:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        print('device0',torch.cuda.get_device_name(device0))
        print('device1',torch.cuda.get_device_name(device1))
        dataset = LM_QASPA_DataLoader(args, args.train_statements, args.train_adj, args.train_sp,
                                        args.dev_statements, args.dev_adj, args.dev_sp,
                                        args.test_statements, args.test_adj, args.test_sp,
                                        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                        device=(device0, device1),
                                        model_name=args.encoder,
                                        max_seq_length=args.max_seq_len,
                                        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                        subsample=args.subsample)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################

        model = LM_QASPA(args.encoder, encoder_only=args.encoder_only, skip_type=args.skip_type, skip_placement=args.skip_placement, 
                            algebra=args.algebra, qa_context=args.qa_context, sent_trans=args.sent_trans, k=args.k - 1, 
                            pretrained_concept_emb=cpnet_emb, qa_emb=qa_emb, permute_vec=permute_vec,
                            fc_dim=args.fc_dim, sp_hidden_dim = args.sp_hidden_dim, sp_output_dim = args.sp_output_dim,
                            n_fc_layer=args.fc_layer_num, p_sp=args.dropoutspa, p_fc=args.dropoutf, sp_layer_norm=args.sp_layer_norm,
                            normalize_graphs=args.normalize_graphs, normalize_embeddings=args.normalize_embeddings, score_mlp=args.score_mlp, 
                            device=device1, concept_dim=concept_dim, init_range=args.init_range, encoder_config={})
        if args.load_model_path:
            print (f'loading and initializing model from {args.load_model_path}')
            model_state_dict = torch.load(args.load_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state_dict['model'])
            optimizer.load_state_dict(model_state_dict['optimizer'])
            scheduler.load_state_dict(model_state_dict['scheduler'])

        model.encoder.to(device0)
        model.decoder.to(device1)


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
   
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)
    elif args.lr_schedule == 'warmup_poly':
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, power=args.power)
    elif args.lr_schedule =='warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, num_cycles=args.cycles)
    elif args.lr_schedule =='warmup_cosine_restarts':
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, num_cycles=args.cycles)
    

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    def calc_eval_loss_and_acc(eval_set, model):
        n_samples, n_correct = 0, 0
        model.eval()
        loss_acm = 0.0
        n_correct_acm = 0
        with torch.no_grad():
            for qids, labels, *input_data in eval_set:
                logits = model(*input_data)
                n_samples += labels.size(0)
                loss, n_correct = compute_loss_and_acc(logits, labels)
                loss_acm += float(loss) * len(qids)
                n_correct_acm += n_correct
        return n_correct_acm / n_samples, loss_acm / n_samples
    
    def compute_loss_and_acc(logits, labels):
        if logits is None:
            loss = 0.
        elif args.loss == 'margin_rank':
            num_choice = logits.size(1)
            flat_logits = logits.view(-1)
            correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
            correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
            wrong_logits = flat_logits[correct_mask == 0]
            y = wrong_logits.new_ones((wrong_logits.size(0),))
            loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
            n_correct = (logits.argmax(1) == labels).sum().item()
        elif args.loss == 'cross_entropy':
            loss = loss_func(logits, labels)
            n_correct = (logits.argmax(1) == labels).sum().item()
        return loss, n_correct

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################
    print('-' * 71)
    if args.fp16:
        print ('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()

    train_size = dataset.train().indexes.size(0)

    global_step = 0

    # eval (randomly initialized) model before training
    model.eval()
    dev_acc, dev_loss = calc_eval_loss_and_acc(dataset.dev(), model)
    test_acc, test_loss = calc_eval_loss_and_acc(dataset.test(), model)


    print('-' * 71)
    print('| epoch ' + str(0) + ' | step ' + str(global_step) + ' | dev_acc ' + str(dev_acc) + ' | test_acc ' + str(test_acc) + ' |')
    print('-' * 71)
    if not args.hyperparameter_tuning:
        with open(log_path, 'a') as fout:
            fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))

    best_dev_acc = dev_acc
    best_test_acc = test_acc
    best_dev_epoch = 0
    
    wandb.log({"dev_acc": dev_acc, "dev_loss": dev_loss, "best_dev_acc": best_dev_acc, "best_dev_epoch": best_dev_epoch}, step=global_step)
    wandb.log({"test_acc": test_acc, "test_loss": test_loss, "best_test_acc": best_test_acc}, step=global_step)
    
    n_samples_acm = n_corrects_acm = 0
    total_loss = 0.0
    freeze_net(model.encoder)
    if True:
        for epoch_id in tqdm(range(args.n_epochs)):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            model.train()
            start_time = time.time()


            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            logits = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                            loss, n_correct = compute_loss_and_acc(logits, labels[a:b])
                    else:
                        logits = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                        loss, n_correct = compute_loss_and_acc(logits, labels[a:b])

                    # loss is mean reduction (i.e. loss / (b-a)), to get average of batch multiply by (b - a) / bs
                    loss = loss * (b - a) / bs
                    total_loss += loss.item()
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    n_corrects_acm += n_correct
                    n_samples_acm += (b - a)
                    
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()

                global_step += bs

                # log train stats log_interval times per epoch
                if global_step % (train_size // args.log_interval) < bs:
                    ms_per_sample = 1000 * (time.time() - start_time) / n_samples_acm 
                    print('| samples {:5} | elr: {:.2e} | dlr: {:.2e} | loss {:7.4f} | ms/sample {:7.2f} |'.format(global_step, scheduler.get_last_lr()[0], scheduler.get_last_lr()[2], total_loss, ms_per_sample))
                    
                    wandb.log({
                            # current total_loss is averaged by batch size. Change it to be averaged by # of training samples
                            "train_loss": total_loss * bs / n_samples_acm,
                            "train_acc": n_corrects_acm / n_samples_acm,
                            "ms_per_sample": ms_per_sample,
                            "elr": scheduler.get_last_lr()[0],
                            "dlr": scheduler.get_last_lr()[2]}
                            , step=global_step)
                    
                    if math.isnan(total_loss) and epoch_id > 0:
                        # needs epoch_id >= 1 to have an actual best_dev_acc
                        return best_dev_acc
                    total_loss = 0.0
                    n_corrects_acm = n_samples_acm = 0

                

            # evaluate model after every epoch
            model.eval()
            dev_acc, dev_loss = calc_eval_loss_and_acc(dataset.dev(), model)
            test_acc, test_loss = calc_eval_loss_and_acc(dataset.test(), model)


            print('-' * 71)
            print('| epoch ' + str(epoch_id) + ' | step ' + str(global_step) + ' | dev_acc ' + str(dev_acc) + ' | test_acc ' + str(test_acc) + ' |')
            print('-' * 71)
            if not args.hyperparameter_tuning:
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))

            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                best_test_acc = test_acc
                best_dev_epoch = epoch_id
            if (args.save_model==2): 
                checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "best_test_acc": best_test_acc, "config": args}
                torch.save(checkpoint, model_path +".{}".format(epoch_id))
            elif (args.save_model==1) and (best_dev_epoch==epoch_id):
                checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "best_test_acc": best_test_acc, "config": args}
                torch.save(checkpoint, model_path)
            
            wandb.log({"dev_acc": dev_acc, "dev_loss": dev_loss, "best_dev_acc": best_dev_acc, "best_dev_epoch": best_dev_epoch}, step=global_step)
            wandb.log({"test_acc": test_acc, "test_loss": test_loss, "best_test_acc": best_test_acc}, step=global_step)
                
            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                return best_dev_acc
            
            if args.hyperparameter_tuning:
                trial.report(dev_acc, epoch_id)

                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_dev_acc


# def eval_detail(args):
#     assert args.load_model_path is not None
#     model_path = args.load_model_path

#     cp_emb = [np.load(path) for path in args.ent_emb_paths]
#     cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
#     concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
#     print('| num_concepts: {} |'.format(concept_num))

#     model_state_dict, old_args = torch.load(model_path, map_location=torch.device('cpu'))
#     model = LM_QASPA(old_args.encoder, k=old_args.k - 1,
#                         fc_dim=old_args.fc_dim, sp_hidden_dim = old_args.sp_hidden_dim, sp_output_dim = old_args.sp_output_dim,
#                         n_fc_layer=old_args.fc_layer_num, p_sp=old_args.dropoutspa, p_fc=old_args.dropoutf, 
#                         concept_dim=old_args.concept_dim, init_range=old_args.init_range, encoder_config={})
    
#     model.load_state_dict(model_state_dict)

#     if torch.cuda.device_count() >= 2 and args.cuda:
#         device0 = torch.device("cuda:0")
#         device1 = torch.device("cuda:1")
#     elif torch.cuda.device_count() == 1 and args.cuda:
#         device0 = torch.device("cuda:0")
#         device1 = torch.device("cuda:0")
#     else:
#         device0 = torch.device("cpu")
#         device1 = torch.device("cpu")
#     print('device0',torch.cuda.get_device_name(device0))
#     print('device1',torch.cuda.get_device_name(device1))
    
#     model.encoder.to(device0)
#     model.decoder.to(device1)
#     model.eval()

#     statement_dic = {}
#     for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
#         statement_dic.update(load_statement_dict(statement_path))

#     use_contextualized = 'lm' in old_args.ent_emb

#     print ('inhouse?', args.inhouse)

#     print ('args.train_statements', args.train_statements)
#     print ('args.dev_statements', args.dev_statements)
#     print ('args.test_statements', args.test_statements)

#     dataset = LM_QASPA_DataLoader(args, args.train_statements, args.train_sp,
#                                     args.dev_statements, args.dev_sp,
#                                     args.test_statements, args.test_sp,
#                                     batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
#                                     device=(device0, device1),
#                                     model_name=args.encoder,
#                                     max_seq_length=args.max_seq_len,
#                                     is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
#                                     subsample=args.subsample)

#     save_test_preds = args.save_model
#     dev_acc, dev_loss = calc_eval_loss_and_acc(dataset.dev(), model)
#     print('dev_acc {:7.4f}'.format(dev_acc))
#     if not save_test_preds:
#         test_acc, test_loss = calc_eval_loss_and_acc(dataset.test(), model) if args.test_statements else 0.0
#     else:
#         eval_set = dataset.test()
#         total_acc = []
#         count = 0
#         dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
#         preds_path = os.path.join(args.save_dir, 'test_preds_{}.csv'.format(dt))
#         with open(preds_path, 'w') as f_preds:
#             with torch.no_grad():
#                 for qids, labels, *input_data in tqdm(eval_set):
#                     count += 1
#                     logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
#                     predictions = logits.argmax(1) #[bsize, ]
#                     preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
#                     for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
#                         acc = int(pred.item()==label.item())
#                         print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
#                         f_preds.flush()
#                         total_acc.append(acc)
#         test_acc = float(sum(total_acc))/len(total_acc)
        
#         print('-' * 71)
#         print('test_acc {:7.4f}'.format(test_acc))
#         print('-' * 71)



if __name__ == '__main__':
    parser = get_parser()
    args, _ = parser.parse_known_args()
    # general
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/qaspa/', help='model output directory')
    parser.add_argument('--save_model', default=0, type=float, help="0: do not save model checkpoints. 1: save if best dev. 2: save always")
    parser.add_argument('--load_model_path', default=None)
    parser.add_argument('--debug_sample_size', default=64, type=int, help='debugging dataset sample size')
    parser.add_argument('--study_name', default=None, help='optuna study name')
    parser.add_argument("--run_name", default=None, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=True, type=utils.bool_flag, nargs='?', const=True, help="Whether to use wandb or not.")
    
    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--train_sp', default=f'data/{args.dataset}/spa/hrr_normalized/train_graph_sp.npy')
    parser.add_argument('--dev_sp', default=f'data/{args.dataset}/spa/hrr_normalized/dev_graph_sp.npy')
    parser.add_argument('--test_sp', default=f'data/{args.dataset}/spa/hrr_normalized/test_graph_sp.npy')
    parser.add_argument('--permute_vec_path', default=None, type=str)
    
    # model architecture
    parser.add_argument('--encoder_only', default=False, type=bool_flag, nargs='?', const=True, help='use model without the spa layers')

    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--concept_dim', default=1024, type=int, help='dimension of the concept (and graph) embeddings')
    parser.add_argument('--sp_hidden_dim', default=1024, type=int, help='dimension of hidden layers in spa MLP')
    parser.add_argument('--sp_output_dim', default=1024, type=int, help='dimension of output layer in spa MLP')

    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')

    parser.add_argument('--algebra', default='hrr', choices=['hrr', 'tvtb', 'vtb'], help='binding operation (when including qa context as node)')
    parser.add_argument('--qa_context', default=True, type=bool_flag, nargs='?', const=True, help='bind LM qa context to graph embeddings')
    parser.add_argument('--sent_trans', default=True, type=bool_flag, nargs='?', const=True, help='add linear trans and Gelu activation to LM output')
    parser.add_argument('--normalize_graphs', default=True, type=bool_flag, nargs='?', const=True, help='normalize graph sps')
    parser.add_argument('--normalize_embeddings', default=True, type=bool_flag, nargs='?', const=True, help='normalize graph sps')

    parser.add_argument('--score_mlp', default=False, type=bool_flag, nargs='?', const=True, help='normalize graph sps')    
    parser.add_argument('--skip_type', default=0, type=int, help='What type of skip blocks to use')
    parser.add_argument('--skip_placement', default=0, type=int, help='Where within the block is skip connection places')
    
    # regularization
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')
    parser.add_argument('--dropoutspa', type=float, default=0.2, help='dropout for fully-connected sp-layers')
    parser.add_argument('--sp_layer_norm', type=bool_flag, default=False, help='dropout for fully-connected sp-layers')
    

    # optimization
    parser.add_argument('--hyperparameter_tuning',default=False, type=bool_flag, nargs='?', const=True, help='')
    parser.add_argument('--n_trials', default=100, type=int)
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=8, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=1000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')
    main(args)
