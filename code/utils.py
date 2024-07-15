import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def filtered_adj(adj):
    
    adj=sp.coo_matrix(adj)
    indices = np.vstack((adj.row, adj.col))
    edges=list(zip(indices[0],indices[1]))
    G=nx.DiGraph(edges)
    return np.array(nx.adjacency_matrix(G).todense())

def save_model(model_file,m_version,
               data_name):
   
    with open('../save_model/{}/{}_v{}.pkl'.format(data_name,data_name,m_version),'wb') as pkl_obj:
        pkl.dump(model_file,pkl_obj)

def load_model(data_name,m_version):

    with open('../save_model/{}/{}_v{}.pkl'.format(data_name,data_name,m_version),'rb') as pkl_obj:
        model=pkl.load(pkl_obj)
    
    return model

def save_data(name,snapshots,id=0):
    save_path = "../src_data/{}/{}_{}.pkl".format(name,name,id)
    with open(save_path, "wb") as f:
        pkl.dump(snapshots, f)
    print("Processed data has been saved at {}".format(save_path))

def load_data(name,id=0):
    save_path = "../src_data/{}/{}_{}.pkl".format(name,name,id)
    with open(save_path, "rb") as f:
        data=pkl.load(f)
    print("{} data has been loaded !".format(name))
    return data

def save_gen_data(name,samples,id=0):
    # samples是[(A_1,X_1)...(A_T,X_T)]
    save_path = "../gen_data/{}/{}_{}.pkl".format(name,name,id)
    with open(save_path, "wb") as f:
        pkl.dump(samples, f)
    print("Generated data has been saved at {}".format(save_path))

def load_gen_data(name,id=0):
    save_path = "../gen_data/{}/{}_{}.pkl".format(name,name,id)
    with open(save_path, "rb") as f:
        data=pkl.load(f)
    print("Gnerated {} data has been loaded !".format(name))
    return data

def email_maker(start_win=480,end_win=5,SLICE_DAYS=1):
    
    email_url='../data/email/Email.edges'
    
    data=pd.read_csv(email_url,index_col=False)
    
    data['date']=pd.to_datetime(data['value'],unit='s')
    
    links=list(zip(data['src'],data['dst'],data['date'])) 
   
    links.sort(key =lambda x: x[2])
    links=links[1:] 
   
    ts=[link[2] for link in links] 
    
    START_DATE=min(ts)+timedelta(start_win) 
    END_DATE = max(ts)-timedelta(end_win) 
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))
    
    POST=1
    slice_links = defaultdict(lambda: nx.DiGraph())
    links_groups=defaultdict(lambda:[])
    for (a, b, time) in links:
        
        datetime_object = time
        
        if datetime_object<=START_DATE or datetime_object>END_DATE:
            continue
        
        slice_id = (datetime_object - START_DATE).days//SLICE_DAYS
        slice_id = max(slice_id, 0)

        if slice_id not in slice_links.keys():
            slice_links[slice_id] = nx.DiGraph()
            if slice_id > 0:
                
                slice_links[slice_id].add_nodes_from(slice_links[slice_id-1].nodes(data=True))
        slice_links[slice_id].add_edge(a,b)
        links_groups[slice_id].append([a,b,POST])
        
    for slice_id in range(len(links_groups)):
        links_groups[slice_id]=pd.DataFrame(links_groups[slice_id],columns=['src','dst','value'])
    
    used_nodes = []
    for id, slice in slice_links.items():
        print("In snapshoot {:<2}, #Nodes={:<5}, #Edges={:<5}".format(id, \
                            slice.number_of_nodes(), slice.number_of_edges()))

        for node in slice.nodes():
            if not node in used_nodes:
                used_nodes.append(node)
                
    # remap nodes in graphs. Cause start time is not zero, the node index is not consistent
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)}
    for id, slice in slice_links.items():
        slice_links[id] = nx.relabel_nodes(slice, nodes_consistent_map) 
        
    for slice_id in range(len(links_groups)):
        links_groups[slice_id]['src']=links_groups[slice_id]['src'].map(nodes_consistent_map)
        links_groups[slice_id]['dst']=links_groups[slice_id]['dst'].map(nodes_consistent_map)

    snapshots=[] 
    for id, slice in slice_links.items():
        attributes = [] 
        for node in slice:
            attrs=[slice.in_degree(node),slice.out_degree(node)]
            attributes.append(attrs)
        slice.graph["feat"]=attributes 
        snapshots.append(slice)

    print('Finished !')

def bitcoin_maker(start_win=200,end_win=1000,SLICE_DAYS=30):
    
    data=pd.read_csv('../data/bitcoin/bitcoin.csv')

    data['date']=pd.to_datetime(data['value'],unit='s')

    links=list(zip(data['src'],data['dst'],data['weight'],data['date'])) 

    links.sort(key =lambda x: x[3])

    ts=[link[3] for link in links] 
    
    START_DATE=min(ts)+timedelta(start_win) 
    END_DATE = max(ts)-timedelta(end_win) 
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))

    slice_links = defaultdict(lambda: nx.DiGraph())
    links_groups=defaultdict(lambda:[])
    for (a, b, v, time) in links:
        
        datetime_object = time
        
        if datetime_object<=START_DATE or datetime_object>END_DATE:
            continue
        
        slice_id = (datetime_object - START_DATE).days//SLICE_DAYS
        slice_id = max(slice_id, 0)

        if slice_id not in slice_links.keys():
            slice_links[slice_id] = nx.DiGraph()
            if slice_id > 0:
               
                slice_links[slice_id].add_nodes_from(slice_links[slice_id-1].nodes(data=True))
                slice_links[slice_id].add_edges_from(slice_links[slice_id-1].edges())
                links_groups[slice_id].extend(links_groups[slice_id-1])
        slice_links[slice_id].add_edge(a,b)
        links_groups[slice_id].append([a,b,v])
        
    for slice_id in range(len(links_groups)):
        links_groups[slice_id]=pd.DataFrame(links_groups[slice_id],columns=['src','dst','value'])

    used_nodes = []
    for id, slice in slice_links.items():
        print("In snapshoot {:<2}, #Nodes={:<5}, #Edges={:<5}".format(id, \
                            slice.number_of_nodes(), slice.number_of_edges()))
     
        for node in slice.nodes():
            if not node in used_nodes:
                used_nodes.append(node)
    
    # remap nodes in graphs. Cause start time is not zero, the node index is not consistent
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)} 
    for id, slice in slice_links.items():
        
        slice_links[id] = nx.relabel_nodes(slice, nodes_consistent_map) 
        
    for slice_id in range(len(links_groups)):
        links_groups[slice_id]['src']=links_groups[slice_id]['src'].map(nodes_consistent_map)
        links_groups[slice_id]['dst']=links_groups[slice_id]['dst'].map(nodes_consistent_map)

    snapshots=[] 
    for id, slice in slice_links.items():
        attributes = [] 
        for node in slice:
            avg_rates=links_groups[id].query('dst=={}'.format(node)).value.mean() if slice.in_degree(node)!=0 else 0
            attrs=[avg_rates,slice.in_degree(node),slice.out_degree(node)]
            attributes.append(attrs)
        slice.graph["feat"]=attributes 
        snapshots.append(slice)

    print('Finished !')
    
    return snapshots

def vote_maker(start_win=200,end_win=540,SLICE_DAYS=15,divide_num=20):
    
    data=pd.read_csv('../data/vote/vote.edges',sep=' ',index_col=False)
   
    data['date']=pd.to_datetime(data['value'],unit='s')
   
    links=list(zip(data['src'],data['dst'],data['weight'],data['date'])) 
    
    links.sort(key =lambda x: x[3])
    
    ts=[link[3] for link in links] 
    
    START_DATE=min(ts)+timedelta(start_win) 
    END_DATE = max(ts)-timedelta(end_win) 
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))

    slice_links = defaultdict(lambda: nx.DiGraph())
    links_groups=defaultdict(lambda:[])
    for (a, b, v, time) in links:
        
        datetime_object = time
        
        if datetime_object<=START_DATE or datetime_object>END_DATE:
            continue
        
        slice_id = (datetime_object - START_DATE).days//SLICE_DAYS
        slice_id = max(slice_id, 0)

        if slice_id not in slice_links.keys():
            slice_links[slice_id] = nx.DiGraph()
            if slice_id > 0:
                slice_links[slice_id].add_nodes_from(slice_links[slice_id-1].nodes(data=True))
                slice_links[slice_id].add_edges_from(slice_links[slice_id-1].edges())
                links_groups[slice_id].extend(links_groups[slice_id-1]) 
        slice_links[slice_id].add_edge(a,b)
        links_groups[slice_id].append([a,b,v])
        
    for slice_id in range(len(links_groups)):
        links_groups[slice_id]=pd.DataFrame(links_groups[slice_id],columns=['src','dst','value'])
    
    used_nodes = []
    for id, slice in slice_links.items():
        print("In snapshoot {:<2}, #Nodes={:<5}, #Edges={:<5}".format(id, \
                            slice.number_of_nodes(), slice.number_of_edges()))

        for node in slice.nodes():
            if not node in used_nodes:
                used_nodes.append(node)
    
    # remap nodes in graphs. Cause start time is not zero, the node index is not consistent
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)} # 有向映射到整数下标
    for id, slice in slice_links.items():
        
        slice_links[id] = nx.relabel_nodes(slice, nodes_consistent_map) 
        

    for slice_id in range(len(links_groups)):
        links_groups[slice_id]['src']=links_groups[slice_id]['src'].map(nodes_consistent_map)
        links_groups[slice_id]['dst']=links_groups[slice_id]['dst'].map(nodes_consistent_map)
        
    
    snapshots=[] 
    for id, slice in slice_links.items():
        attributes = [] 
        for node in slice:
            sum_votes=links_groups[id].query('dst=={}'.format(node)).value.sum()/divide_num
            attrs=[sum_votes]
            attributes.append(attrs)
        slice.graph["feat"]=attributes 
        snapshots.append(slice)

    print('Finished !')
    
    return snapshots

def Padding_G_t(config,graph_seq,t):

    seq_len=config.seq_len
    if seq_len==None:
        seq_len=len(graph_seq)
    
    max_num_nodes=graph_seq[seq_len-1].number_of_nodes()
    
    Adj=torch.from_numpy(nx.adjacency_matrix(graph_seq[t]).todense()).type(torch.float32)
    pad_adj=nn.ZeroPad2d((0,max_num_nodes-Adj.shape[0],0,max_num_nodes-Adj.shape[1])) 
    Adj=pad_adj(Adj)
    
    # 2. 属性部分
    X=torch.from_numpy(np.array(graph_seq[t].graph['feat'])[:,config.attr_col[config.dataset]]).type(torch.float32)
    pad_attr=nn.ZeroPad2d((0,0,0,max_num_nodes-X.shape[0])) 
    X=pad_attr(X)
    
    return Adj,X

def Padding(graph_seq,
            config):
    
    A_list=[]
    X_list=[]
    
    seq_len=config.seq_len
    if seq_len==None:
        seq_len=len(graph_seq)
    
    max_num_nodes=graph_seq[seq_len-1].number_of_nodes()
    
    for t in range(seq_len):
        
        Adj=torch.from_numpy(nx.adjacency_matrix(graph_seq[t]).todense()).type(torch.float32)
        pad_adj=nn.ZeroPad2d((0,max_num_nodes-Adj.shape[0],0,max_num_nodes-Adj.shape[1])) 
        Adj=pad_adj(Adj)
        A_list.append(Adj)
        
        X=torch.from_numpy(np.array(graph_seq[t].graph['feat'])[:,config.attr_col[config.dataset]]).type(torch.float32)
        pad_attr=nn.ZeroPad2d((0,0,0,max_num_nodes-X.shape[0]))
        X=pad_attr(X)
        X_list.append(X)
        
    return A_list,X_list