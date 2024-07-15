import torch
import argparse
import warnings
warnings.filterwarnings('ignore')

# EPS = torch.finfo(torch.float).eps  # numerical logs
parser = argparse.ArgumentParser()
parser.add_argument('--x_dim', default=1, help='feature dimension')
parser.add_argument('--h_dim', default=8, help='hidden state')
parser.add_argument('--z_dim', default=8,
                    help='latent random variable dimmension')
parser.add_argument('--bi_flow',default=True,help='is bi-flow encoder')
parser.add_argument('--enc_hid_dim', default=16,
                    help='dimmension of hidden state in graph encoder')
parser.add_argument('--post_hid_dim', default=16,
                    help='hidden dimension of posterior distribution')
parser.add_argument('--prior_hid_dim', default=16,
                    help='hidden dimension of prior distribution')
parser.add_argument('--attr_hid_dim', default=16,
                    help='hidden dimension of attribute decoder')
parser.add_argument('--n_encoder_layer', default=2,
                    help='number of encoder layer')
parser.add_argument('--n_rnn_layer', default=1, help='layer number of RNN')
parser.add_argument('--max_num_nodes', default=3783,
                    help='maximum number of generated nodes')
parser.add_argument('--seq_len', default=14, help='length of graph sequence')
parser.add_argument('--ini_method',default='zero',help='initialization method')

parser.add_argument('--num_mix_component', default=3,
                    help='number of mixture components')
parser.add_argument('--bernoulli_hid_dim', default=16,
                    help='hidden dimension of bernoulli model')
parser.add_argument('--reduce', default='mean', help='reduction method')

parser.add_argument('--dec_method',default='gnn',help='attribute decoding method')
parser.add_argument('--no_neg',default=True,help='non negtive')

parser.add_argument('--activation', default='sin',
                    help='activation function for timevec')
parser.add_argument('--is_vectorize',default=False,help='if vectorize time or not')


parser.add_argument('--eps',default=1e-6,help='eps')
parser.add_argument('--learning_rate', default=6e-2, help='learning_rate')
parser.add_argument('--pos_weight',default=3,help='positive weight')
parser.add_argument('--neg_num',default=35,help='negtive sampling number by row')
parser.add_argument('--use_rec_loss',default=False,help='if use reconstruction loss')
parser.add_argument('--attr_optimize',default='sce',help='Attribute optimize method')
parser.add_argument('--num_epoch', default=100, help='epoch number')
parser.add_argument('--sample_interval', default=10, help='sample interval')
parser.add_argument('--clip_norm', default=10, help='clip norm')
parser.add_argument('--save_interval',default=25,help='model save interval')


parser.add_argument('--dataset', default='email', help='dataset name')
parser.add_argument('--m_version',default=1,help='model version')
parser.add_argument('--attr_col', default={'bitcoin': [0],
                                           'email':[0,1],
                                           'vote':[0],
                                           'loan': [0, 1]}, help='attributes column')
parser.add_argument('--data_id', default=2, help='dataset id')


parser.add_argument('--verbose',default=False,help='print verbose')
parser.add_argument('--is_ratio',default=True,help='computing ratio')
parser.add_argument('--mmd_beta',default=2.0,help='beta value for mmd')
parser.add_argument('--eval_method',default='mean',help='evaluation method')

 
parser.add_argument('--num_bins',default=50,help='number of bins')

parser.add_argument('--n_box',default=5,help='number of box')
parser.add_argument('--deg_sep',default=2,help='degree seperatioin')


parser.add_argument('--device', default='cuda:0', help='computaton device')
parser.add_argument('--seed', default=2023, help='random seed')
args = parser.parse_args(args=[])