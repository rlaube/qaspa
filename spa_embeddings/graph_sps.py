# generate QA graph VSA embeddings
# CALLED BY generate_graph_sps.bat

import numpy as np
import nengo_spa as spa
from tqdm import tqdm
import csv
import argparse
import pickle
import os

def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='medqa', help='QA dataset (csqa, obqa)') 
    parser.add_argument('--pruned', default=False, type=bool_flag, help='Use pruned graphs (to 200 nodes)') 
    parser.add_argument('--split', default='train', help='train, val, or test')
    parser.add_argument('--emb-dir', default='.', help='Directory with concept embeddings, strings')
    parser.add_argument('--normalize-concept', default='unitary', help='Normalize concepts') 
    parser.add_argument('--normalize-rel', default='unitary', help='Normalize relations') 
    parser.add_argument('--concept-file', default='tzw_concept_emb.npy', help='Concept embeddings np array')
    parser.add_argument('--relation-file', default='random_relation_emb.npy', help='Concept embeddings np array')
    parser.add_argument('--output-dir', default='medqa', help='output directory')
    parser.add_argument('--algebra', default='hrr', help='Algebra for binding (hrr or tvtb)')
    parser.add_argument('--permute-vector-path', default='../data/perm_vectors/perm_vector_seed', type=str) 
    parser.add_argument('--permute-vector-seed', default=None, type=int, help='Permute the tail concept of each triple using the given permute vector seed') 

    parser.add_argument('--partition-total', default=1, type=int, help='Number of sections to split the dataset')
    parser.add_argument('--partition', default=1, type=int, help='Portion of the dataset to process (1-total)')
    parser.add_argument('--mc-options', default=5, help='number of multiple choice options in the dataset')

    
    args = parser.parse_args()

    print('Dataset: ', args.dataset, args.split)
    print('Algebra:', args.algebra)
    print('Partition: ', args.partition)
    print('Pruned:', args.pruned)
    print('Normalize concepts:', args.normalize_concept)
    print('Normalize relations:', args.normalize_rel)

    # prepare outdir
    current_directory = os.getcwd()
    new_directory = os.path.join(current_directory, args.output_dir)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    print('Output dir:', new_directory)

    # load graph adjacency and concept id info
    graph_file = args.split+'_graph_ids_pruned'+'.npz' if args.pruned else args.split+'_graph_ids.npz'
    graphs = np.load(args.dataset+'/'+graph_file, allow_pickle=True)

    # load vocab semantic pointers (embeddings) and mapping
    # with open(args.emb_dir+'cpnet_vocab_sp.pkl', 'rb') as f:
    #     all_sp = pickle.load(f)

    concept_sp = np.load(args.emb_dir+args.concept_file)
    rel_sp = np.load(args.emb_dir+args.relation_file)

    concept_norms = np.linalg.norm(concept_sp, axis=1)[:, np.newaxis]
    rel_norms = np.linalg.norm(rel_sp, axis=1)[:, np.newaxis]
    print('concept_norms.max',concept_norms.max())
    print('concept_norms.min',concept_norms.min())
    print('rel_norms.max',rel_norms.max())
    print('rel_norms.min',rel_norms.min())
    
    if args.algebra == 'hrr':
        alg = spa.algebras.HrrAlgebra() 
    elif args.algebra == 'tvtb':
         alg = spa.algebras.TvtbAlgebra()
    elif args.algebra == 'vtb':
        alg = spa.algebras.VtbAlgebra()

    # normalize sps
    if args.normalize_concept == 'unitary':
        for i, concept in enumerate(concept_sp):
            if np.linalg.norm(concept) > 0:
                concept_sp[i] = alg.make_unitary(concept)
    elif args.normalize_concept == 'l2':
        concept_sp = np.divide(concept_sp, concept_norms, out=np.zeros_like(concept_sp), where=concept_norms!=0)


    if args.normalize_rel == 'unitary':
        for i, rel in enumerate(rel_sp):
            if np.linalg.norm(rel) > 0:
                rel_sp[i] = alg.make_unitary(rel)
    elif args.normalize_rel == 'l2':
        rel_sp = np.divide(rel_sp, rel_norms, out=np.zeros_like(rel_sp), where=rel_norms!=0)
        
    print('new concept_norms.max',np.linalg.norm(concept_sp, axis=-1).max())
    print('new rel_norms.max',np.linalg.norm(rel_sp, axis=-1).max())
    # with open(args.emb_dir+'cpnet_vocab_map.pkl', 'rb') as f:
    #     vocab_dict = pickle.load(f)

    # create graph sps
    dataset_size = len(graphs['concept_ids'])
    part_range = np.arange((args.partition - 1) * dataset_size // args.partition_total, 
                           np.min((dataset_size, args.partition * dataset_size // args.partition_total)))

    # load permutation vector
    if args.permute_vector_seed is not None:
        pemute_vector = np.load(args.permute_vector_path+str(args.permute_vector_seed)+'.npy')

    dim = concept_sp.shape[1]

    concept_ids = graphs['concept_ids']
    rels = graphs['rels']
    subjects = graphs['subjects']
    objects = graphs['objects']

    all_graph_sp = []
    # all_triples_strings = []

    for i in tqdm(part_range):
        graph_sp = np.zeros(dim)
        # triples_strings = []
        if args.permute_vector_seed is None:
            for subj, rel, obj in zip(subjects[i], rels[i], objects[i]):
                subj_id = concept_ids[i][subj]
                obj_id = concept_ids[i][obj]
                # subj_str = vocab_dict['CONCEPT'+str(subj_id)]
                # rel_str = vocab_dict['RELATION'+str(rel)]
                # obj_str = vocab_dict['CONCEPT'+str(obj_id)]
                
                # do (subj * rel) * obj
                assert(np.linalg.norm(concept_sp[subj_id]) > 0)
                assert(np.linalg.norm(concept_sp[obj_id]) > 0)
                triple = alg.bind(
                                alg.bind(concept_sp[subj_id], rel_sp[rel]), 
                                concept_sp[obj_id])
                graph_sp += triple
        else:
            for subj, rel, obj in zip(subjects[i], rels[i], objects[i]):
                subj_id = concept_ids[i][subj]
                obj_id = concept_ids[i][obj]

                assert(np.linalg.norm(concept_sp[subj_id]) > 0)
                assert(np.linalg.norm(concept_sp[obj_id]) > 0)
                # do (subj * rel) * perm(obj)
                triple = alg.bind(
                                alg.bind(concept_sp[subj_id], rel_sp[rel]), 
                                concept_sp[obj_id][pemute_vector])
                graph_sp += triple
            # triples_strings.append(" ".join([subj_str, rel_str, obj_str]))
        all_graph_sp.append(graph_sp)
        # all_triples_strings.append(triples_strings)

    
    np.save(new_directory+'/'+args.split+'_graph_sp'+str(args.partition)+'.npy', all_graph_sp)

    # with open(new_directory+'/'+args.split+'_triples_strings'+str(args.partition)+'.csv', "w", encoding='utf-8') as f:
    #     wr = csv.writer(f)
    #     wr.writerows(all_triples_strings)


if __name__ == '__main__':
    main()