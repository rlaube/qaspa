# COMBINE AND RESHAPE SP DATA AND TRIPLES
# CALLED BY run_split_graph_sp.bat
import numpy as np
import csv
import argparse
import os

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dir', default=None, required=True, help='QA dataset (csqa, obqa)')
    parser.add_argument('--dataset', default='medqa', help='QA dataset (csqa, obqa)') 
    parser.add_argument('--split', default='train', help='train, val, or test')
    parser.add_argument('--mc-options', type=int, default=5, help='number of multiple choice options in the dataset')
    parser.add_argument('--partitions', type=int, default=1, help='number of partitions the dataset split was divided into')
    parser.add_argument('--delete-partitions', type=bool, default=False, help='delete files for each partition')
    args = parser.parse_args()

    # dir = './roberta-large/pruned-cpnet/'

    all_sp = []
    for i in range(args.partitions):
        sp_subset = np.load(args.dir+'/'+args.split+'_graph_sp'+str(i+1)+'.npy')
        all_sp.append(sp_subset)

    all_sp = np.vstack(all_sp)
    reshaped_sp = np.reshape(all_sp, (int(all_sp.shape[0]/args.mc_options), args.mc_options, -1))
    np.save(args.dir+'/'+args.split+'_graph_sp', reshaped_sp)

    # all_triples = []
    # for i in range(args.partitions):
    #     with open(args.dir+'/'+args.split+'_triples_strings'+str(i+1)+'.csv', "r", encoding='utf-8') as f:
    #         reader = csv.reader(f)
    #         for i, row in enumerate(reader):
    #             if i%2 == 0:
    #                 all_triples.append(row)

    # with open(args.dir+'/'+args.split+'_triples_strings.csv', "w") as f:
    #     wr = csv.writer(f)
    #     wr.writerows(all_triples)

    if args.delete_partitions:
        for i in range(args.partitions):  
                try:
                    os.remove(args.dir+'/'+args.split+'_graph_sp'+str(i+1)+'.npy')
                except OSError as e:
                    # If it fails, inform the user.
                    print("Error: %s - %s." % (e.filename, e.strerror))

                # try:
                #     os.remove(args.dir+'/'+args.split+'_triples_strings'+str(i+1)+'.csv')
                # except OSError as e:
                #     # If it fails, inform the user.
                #     print("Error: %s - %s." % (e.filename, e.strerror)) 
    

if __name__ == '__main__':
    main()