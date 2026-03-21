import time
import pickle
from itertools import product
import numpy as np
np.random.seed(42)
from layered_graph_ann import *

np.random.seed(42)

metric = 'cosine'

metric_map = {'cosine': lambda x, y: np.dot(x.vector,y.vector)/(x.norm*y.norm), 'euclidean': lambda x, y: np.linalg.norm(x.vector - y.vector)}
metric_func = metric_map[metric]

num_neighbors = 10
num_queries = 100

test_data = {'wide_clusters':{'build_times':{}, 'bf_times':{},'retrieve_stats':{}}, 'moderate_clusters':{'build_times':{}, 'bf_times':{},'retrieve_stats':{}}, 'close_clusters':{'build_times':{}, 'bf_times':{},'retrieve_stats':{}}}

def generate_data(dim, theta, sz, scale=1):
	cov = (scale**2/dim) * np.eye(dim)
	ctr1 = np.zeros(dim)
	ctr1[0] = 1
 
	ctr2 = np.zeros(dim)
	ctr2[0] = np.cos(theta)
	ctr2[1] = np.sin(theta)
 
	ctr3 = np.zeros(dim)
	ctr3[0] = np.cos(theta)
	ctr3[1] = -np.sin(theta)
	
	
	ret1 = ctr1 + np.random.multivariate_normal(np.zeros(dim), cov, sz)
	norms1 = np.linalg.norm(ret1, axis=1, keepdims=True)
	ret1 = ret1/(norms1 + 1e-12)

	ret2 = ctr2 + np.random.multivariate_normal(np.zeros(dim), cov, sz)
	norms2 = np.linalg.norm(ret2, axis=1, keepdims=True)
	ret2 = ret2/(norms2 + 1e-12)
 
	ret3 = ctr3 + np.random.multivariate_normal(np.zeros(dim), cov, sz)
	norms3 = np.linalg.norm(ret3, axis=1, keepdims=True)
	ret3 = ret3/(norms3 + 1e-12)
	
	labels = np.array([0]*sz + [1]*sz + [2]*sz)
	data = np.vstack([ret1,ret2,ret3])
	
	return labels, data

def generate_queries(dim, theta, num_queries, scale=2):
	cov = (scale**2/dim)*np.eye(dim)
	locs = []

	for factor in [-1.25, -0.5, 0, 0.5, 1.25]:
		ctr = np.zeros(dim)
		ctr[0] = np.cos(factor*theta)
		ctr[1] = np.sin(factor*theta)
		locs.append(ctr)

	queries = []
	for _ in range(num_queries):
		ctr = locs[np.random.randint(len(locs))]
		q = np.random.multivariate_normal(np.zeros(dim), cov)
		queries.append((ctr+q)/(np.linalg.norm(ctr+q)+1e-12))

	return queries

dims = [5, 25, 50, 75, 100]
sizes = [10000, 20000, 30000, 40000]
exp_levels = [2, 6, 10]
pool_factors = [5, 10, 20, 40, 80]
centers = {'wide_clusters':2*np.pi/3, 'moderate_clusters':np.pi/3, 'close_clusters':np.pi/6}

for ctr in centers:
	a = centers[ctr]
	for dim, sz in product(dims, sizes):
		#get data
		labels, data = generate_data(dim, a, sz)
		doc_list = [{'vector':data[i], 'data':data[i], 'source':i} for i in range(len(data))]
		node_list = [VectorNode(doc['vector'], doc['data'], doc['source']) for doc in doc_list]
		
		
		#get queries
		raw_queries = generate_queries(dim, a, num_queries)
		query_list = [{'vector':vec, 'data':vec} for vec in raw_queries]
		query_nodes = [VectorNode(vec, vec) for vec in raw_queries]
  
		#brute force benchmark
		bf_retrieval = []
		bf_avg_score = []
		bf_time = 0
		for query in query_nodes:
			start_time = time.perf_counter()
			best_bf = sorted([[node, metric_func(query, node)] for node in node_list], key=lambda x: x[1])[-1*num_neighbors:]
			end_time = time.perf_counter()
			
			bf_retrieval.append(best_bf)
			bf_avg_score.append(sum([pair[1] for pair in best_bf])/num_neighbors)
			bf_time += (end_time - start_time)
		bf_time /= num_queries
		test_data[ctr]['bf_times'][(dim, 3*sz)] = bf_time
			
		for exp_lvl in exp_levels:
			print(f"Testing {ctr=}, {dim=}, {sz=}, and {exp_lvl=}:")
			#build vector store
			new_VectorStore = VectorStore(exp_level=exp_lvl, metric=metric)
			
			start_time = time.perf_counter()
			new_VectorStore.build_vectorstore(doc_list)
			end_time = time.perf_counter()
			test_data[ctr]['build_times'][(dim, 3*sz, exp_lvl)] = end_time-start_time	
				
			for pool_factor in pool_factors:
				print(f"{pool_factor=}")
				recalls = 0
				times = 0
				inflations =  0
				for vec, bf_result, bf_score  in zip(query_list, bf_retrieval, bf_avg_score):
					bf_nodes = {node[0].source for node in bf_result}
					start_time = time.perf_counter()
					retr_nbrs = new_VectorStore.find_neighbors(vec, num_neighbors, pool_factor)
					end_time = time.perf_counter()
					times += (end_time - start_time)
					
					score = 0.0
					for nbr in retr_nbrs:
						score += nbr[1]
						if nbr[0].source in bf_nodes:
							recalls += 1
					score /= num_neighbors
					inflations += bf_score/score
				print(f"\trecall={recalls/(num_neighbors*num_queries)}")
				print(f"\tinflation={inflations/num_queries:.4f}")				
				test_data[ctr]['retrieve_stats'][(dim, 3*sz, exp_lvl, pool_factor)] = {'recall':recalls/(num_neighbors*num_queries), 'inflation':inflations/num_queries ,'time':times/num_queries}	

			
with open('ANN_Test_Data_Cosine.pkl', 'wb') as f:
    pickle.dump(test_data, f)	
			
			

