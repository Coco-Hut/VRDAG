import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import StrucEvaluator,AttrEvaluator,evaluate_attr
from generator import VRDAG
import time
from var_dist import *
from config import args 
import gc
import utils

import warnings
warnings.filterwarnings('ignore')

# 设置cpu和gpu随机数
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    
    
    graph_seq = utils.load_data(args.dataset, args.data_id)
    A_list, X_list = utils.Padding(graph_seq, args)
    args.max_num_nodes=A_list[-1].shape[0]
    print('{} data loaded!'.format(args.dataset))
    
    # -----------------------------Training----------------------------
    
    train_loss = 0
    model = VRDAG(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.5)

    judge_s=StrucEvaluator(eval_time_len=args.seq_len,eval_method=args.eval_method,
                       verbose=args.verbose,is_ratio=args.is_ratio,mmd_beta=args.mmd_beta) 
    judge_a=AttrEvaluator(eval_time_len=args.seq_len,num_bins=args.num_bins)

    A_src=[utils.filtered_adj(A_list[j].numpy()) for j in range(args.seq_len)]
    X_src=[X_list[k].numpy() for k in range(args.seq_len)] 
    for n_epoch in range(args.num_epoch):
        
        if args.ini_method=='zero':
            h=torch.zeros(args.max_num_nodes,args.h_dim,device=args.device)
        elif args.ini_method=='embed':
            h = model.id_embedding.weight
        else:
            raise ValueError('Wrong initialization method!')

        train_loss=0
        avg_kld_loss=0
        avg_struc_loss=0
        avg_attr_loss=0
        
        time_start = time.time() 
        
        for t in range(args.seq_len):
            
            optimizer.zero_grad(set_to_none=True)

            loss_step = 0
            
            n_nodes=graph_seq[t].number_of_nodes() 
            
            A, X = A_list[t].to(args.device), X_list[t].to(args.device)
            
            if args.is_vectorize:
                t_vec=model.time_to_vec(torch.FloatTensor([t]).to(args.device))
            else:
                t_vec=None
                
            h, kld_loss, struc_loss, attr_loss = model(A, X, h.data,t_vec,n_nodes)
            
            avg_kld_loss+=kld_loss.data.item()
            avg_struc_loss+=struc_loss.data.item()
            avg_attr_loss+=attr_loss.data.item()
            
            loss_step+=kld_loss+struc_loss+attr_loss
        
            loss_step.backward()
        
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            
            optimizer.step()
            
        
        time_end = time.time()    

        print('One Epoch Training Running Time: {} Sec \n'.format(time_end-time_start))
            
        print('Epoch-{}: Average Attribute Loss is {}'.format(n_epoch+1,avg_attr_loss/args.seq_len))
        print('Epoch-{}: Average Structure Loss is {}'.format(n_epoch+1,avg_struc_loss/args.seq_len))
        print('Epoch-{}: Average Latent distribution Loss is {}\n'.format(n_epoch+1,avg_kld_loss/args.seq_len))

        time_start = time.time()
        gen_data=model._sampling(args.seq_len)
        time_end = time.time()    

        print('One Round Testing Running Time: {} Sec \n'.format(time_end-time_start))
    
        if (n_epoch+1) % args.sample_interval == 0:

            with torch.no_grad():
                '''Evaluation'''
                gen_data=model._sampling(args.seq_len)
                
                if args.device in ['cuda:0','cuda:1']:
                    A_gen=[utils.filtered_adj(gen_data[i][0].cpu().numpy()) for i in range(args.seq_len)]
                    X_gen=[gen_data[i][1].cpu().numpy() for i in range(args.seq_len)]
                else:
                    A_gen=[gen_data[i][0].numpy() for i in range(args.seq_len)]
                    X_gen=[gen_data[i][1].numpy() for i in range(args.seq_len)]
                
                
       
                res_dict=judge_s.comp_graph_stats(A_src,A_gen,stats_list=['deg_dist_in',
                                                                        'deg_dist_out',
                                                                        'clus_dist',
                                                                        ])
                
              
                
                attr_entrophy_js,attr_corr=evaluate_attr(src_attrs=X_src,
                                                    gen_attrs=X_gen,
                                                    diff='js',
                                                    bins=20,
                                                    low_bound=0,
                                                    upper_bound=1,
                                                    n_step=args.seq_len)
                
                attr_entrophy_emd,attr_corr=evaluate_attr(src_attrs=X_src,
                                                gen_attrs=X_gen,
                                                diff='emd',
                                                bins=20,
                                                low_bound=0,
                                                upper_bound=1,
                                                n_step=args.seq_len)
                
                
                print('-------------Graph structure statistics are as follows-------------')
                for k,v in res_dict.items():
                    print('{}: {}'.format(k,v))
                print('-------------------------------------------------------------------\n')

                print('-------------Node attributes evaluation are as follows-------------')
                
                
                print('Attribute Entrophy- : JSD: {}  EMD: {}'.format(attr_entrophy_js,attr_entrophy_emd))
                if attr_corr is not None:
                    print('Attribute Correlation Error: {}'.format(attr_corr))
                
                
                print('-------------------------------------------------------------------\n')
                
                del gen_data
                del A_gen
                del X_gen
                
                gc.collect()
    
    
    utils.save_model(model,args.m_version,args.dataset)
    print('Model Saved!')
