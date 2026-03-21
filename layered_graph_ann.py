import numpy as np
import warnings
from itertools import product
from tqdm import tqdm

class MetricException(Exception):
	pass
	
class ValueErrorException(Exception):
	pass
	
class VectorStoreException(Exception):
	pass

class VectorNode:
	def __init__(self, vector, data, source=None, vec_id = -1, max_level=0, eps = 1e-10):
		self.id = vec_id
		self.vector = np.asarray(vector, dtype=float)
		self.data = data
		self.source = source
		self.max_level = max_level
		self.norm = np.linalg.norm(self.vector)
		if self.norm < eps:
			self.norm = eps
	
	def __str__(self):
		return f"VectorNode {self.id}\nData: {self.data}\nSource: {self.source}"
		
	def __repr__(self):
		return f"VectorNode(id={self.id}, level={self.max_level})"
		
def weak_component_analysis(adj_list, max_node_num):
	components = []
	node_list = sorted(adj_list.keys())	

	#set up symmetric adjacency list for weak component checking
	sym_adj = [set() for _ in range(max_node_num)]
	for n in node_list:
		nbr_list = adj_list[n]
		for nbr in nbr_list:
			sym_adj[n].add(nbr)
			sym_adj[nbr].add(n)
	
	#search for components
	visited = [True for _ in range(max_node_num)]
	for nd in node_list:
		visited[nd] = False

	while False in visited:
		new_component = set()
		new_node = visited.index(False)
		new_component.add(new_node)
		visited[new_node]=True
		nbr_set = sym_adj[new_node].copy()
		
		while len(nbr_set) > 0:
			new_node = nbr_set.pop()
			if visited[new_node]:
				continue
			new_component.add(new_node)
			visited[new_node] = True
			
			new_nbrs = sym_adj[new_node].copy()
			for nbr in new_nbrs:
				if visited[nbr] == False:
					nbr_set.add(nbr)
		components.append(sorted(list(new_component)))
		
	return components
	
def KruskalMST(edge_list, minimal=False):
    edge_list = sorted(edge_list, key=lambda x: x[2], reverse=minimal)

    vertex_set = set()
    for edge in edge_list:
        vertex_set.add(edge[0])
        vertex_set.add(edge[1])

    vertex_id = {}
    tree_id = {}

    for i, vrtx in enumerate(vertex_set):
        vertex_id[vrtx] = i
        tree_id[i] = {vrtx}

    chosen_edges = []
    target_edges = len(vertex_set) - 1

    while edge_list and len(chosen_edges) < target_edges:
        new_edge = edge_list.pop()
        vrtx0, vrtx1 = new_edge[0], new_edge[1]

        id0 = vertex_id[vrtx0]
        id1 = vertex_id[vrtx1]

        if id0 != id1:
            chosen_edges.append(new_edge)

            # merge smaller tree into larger tree
            if len(tree_id[id0]) < len(tree_id[id1]):
                id0, id1 = id1, id0

            for vrtx in tree_id[id1]:
                vertex_id[vrtx] = id0

            tree_id[id0].update(tree_id[id1])
            del tree_id[id1]

    return chosen_edges


class VectorStore:
	def __init__(self, exp_level, neighbors_bottom = 16, neighbors_upper = 8, max_level = None, metric = 'cosine', pool_factor=None):
		if metric not in ['cosine', 'euclidean']:
			raise MetricException(f"Unrecognized metric: {metric}")
		self.exp_level = exp_level
		if max_level is None:
			max_level = 5*exp_level
		if max_level <= exp_level:
			max_level = exp_level +1
		self.max_level = max_level
		self.metric_type = metric
		self.metric_map = {'cosine': [lambda x, y: np.dot(x.vector,y.vector)/(x.norm*y.norm), 1], 'euclidean': [lambda x, y: -np.linalg.norm(x.vector - y.vector), -1]}
		self.metric = self.metric_map[metric][0]
		self.metric_multiplier = self.metric_map[metric][1]
		if pool_factor is None:
			pool_factor = 5
		self.pool_factor = pool_factor
		self.node_list = []
		self.layered_graph = dict()
		self.entry_point = None
		for i in range(max_level+1):
			self.layered_graph[i] = dict()
			
		self.neighbors_bottom = neighbors_bottom
		self.neighbors_upper = neighbors_upper
		self.component_info = None
			
	def __str__(self):
		return f"VectorStore Information\nOccupancy: {self.store_size()} nodes\nCurrent Top Level: {self.max_level}\nMetric Type: {self.metric_type}"
		
	def vector_store_layer_info(self):
		if self.component_info is None:
			print("VectorStore not currently built")
		else:
			print(f"Vector Store Components by Layer:")
			for lyr in range(len(self.component_info)):
				comps = self.component_info[lyr]
				n_comps = len(comps)
				sz_comps = [len(comp) for comp in comps]
				tot_sz = sum(sz_comps)
				print(f"Layer {lyr}: Total nodes = {tot_sz}, Total components = {n_comps}")
				for i in range(n_comps):
					print(f"\tComponent {i+1}: Nodes = {sz_comps[i]}")
			
	def store_size(self):
		return len(self.node_list)
		
	def choose_level(self):
		level = np.random.geometric(p=1/(self.exp_level+1))-1
		if level > self.max_level:
			level = self.max_level
		return level
		
	def top_layer_setup(self, node_list, n_nbrs, dummy_nbrs, max_node_num):
		metric_func = self.metric
		
		n = len(node_list)
		if n_nbrs >= n:
			n_nbrs = n-1
		
		if n_nbrs == 0:
			raise ValueErrorException("n_nbrs set to 0: Nodes must have at least one neighbor.")
			
		id_to_idx = {node_list[i].id:i for i in range(n)}
		
		metric_matrix = np.zeros((n,n))
		for i in range(n):
			for j in range(i):
				m = metric_func(node_list[i], node_list[j])
				metric_matrix[i][j] = m
				metric_matrix[j][i] = m
			metric_matrix[i][i] = -np.inf
				
		adj_list = {}
		for i in range(n):
			indxs = np.argsort(metric_matrix[i,:])[-n_nbrs:]
			nbrs = [node_list[j].id for j in indxs]
			adj_list[node_list[i].id] = nbrs
			
		sym_adj_list = {node_list[i].id: set() for i in range(n)}
		for i in range(n):
			node_id = node_list[i].id
			nbrs = adj_list[node_id]
			for nbr in nbrs:
				sym_adj_list[node_id].add(nbr)
				sym_adj_list[nbr].add(node_id)
		for i in range(n):
			sym_adj_list[node_list[i].id] = list(sym_adj_list[node_list[i].id])
			
		components = weak_component_analysis(sym_adj_list, max_node_num)

		#set metric_matrix diagonals to zero for medoid computation
		for i in range(n):
			metric_matrix[i][i] = 0
		
		#separate metric_matrix by component
		comp_metric_matrices = []
		for comp in components:
			comp_idcs = [id_to_idx[nd] for nd in comp]
			comp_metric_matrices.append(metric_matrix[np.ix_(comp_idcs, comp_idcs)])
			
		# dummy entry node neighborhood with up to dummy_nbrs nodes
		dummy_entry_nbrs = []
		
		#Pull Medoid from each component
		for c in range(len(components)):
			comp = components[c]
			submatrix = comp_metric_matrices[c]
			medoid_idx = np.argmax(np.sum(submatrix, axis=1))
			dummy_entry_nbrs.append([comp[medoid_idx]])

		rem = dummy_nbrs - len(dummy_entry_nbrs)
		if rem > 0:
			probs = [len(comp)/n for comp in components]
			rng = np.random.default_rng()
			nums_to_pull = rng.multinomial(rem, probs)
			
			#Pull num in nums_to_pull from each component
			for c in range(len(components)):
				comp = components[c]
				cand_nodes = dummy_entry_nbrs[c]
				if nums_to_pull[c] > len(comp) - 1:
					nums_to_pull[c] = len(comp)-1
					
				while len(cand_nodes) < nums_to_pull[c]+1:
					lst = []
					for nd in comp:
						if nd not in cand_nodes:
							mx_sim = np.max([metric_matrix[id_to_idx[nd]][id_to_idx[cand]] for cand in cand_nodes])
							lst.append([nd, mx_sim])
					cand_nodes.append(sorted(lst, key=lambda x: x[1])[0][0])
			
		dummy_entry_nbrs = [nd for lst in dummy_entry_nbrs for nd in lst]
		
		return sym_adj_list, dummy_entry_nbrs, components
		
	def add_single_node(self, doc):
		if self.store_size() == 0:
			raise VectorStoreException("VectorStore not created.  Use batch entry mode to create VectorStore.")
		
		#set up node and add to list
		node = VectorNode(doc['vector'], doc.get('data'), source=doc.get('source'))
		node_lvl = self.choose_level()
		if node_lvl == self.max_level:
			node_lvl -= 1
		node.max_level = node_lvl
		node.id = self.store_size()
		self.node_list.append(node)
		
		# Add empty adjacency lists for this node at all levels it belongs to
		for lvl in range(node_lvl + 1):
			self.layered_graph[lvl][node.id] = []

		# Start from best entry point in reserved top layer
		nbr_list = sorted(
			[[entry_pt, self.metric(node, self.node_list[entry_pt])] for entry_pt in self.entry_point],
			key=lambda x: x[1]
		)
		curr_nbr_id = nbr_list[-1][0]
		curr_record = nbr_list[-1][1]

		# Follow current structure down through layers above node_lvl
		for lvl in range(self.max_level - 1, node_lvl, -1):
			while True:
				nbr_nodes = self.layered_graph[lvl][curr_nbr_id]
				if not nbr_nodes:
					break

				nbr_list = sorted(
					[[nbr, self.metric(node, self.node_list[nbr])] for nbr in nbr_nodes],
					key=lambda x: x[1]
				)

				if nbr_list[-1][1] > curr_record:
					curr_nbr_id = nbr_list[-1][0]
					curr_record = nbr_list[-1][1]
				else:
					break

		# Insert node from node_lvl down to 0
		for i in range(node_lvl, -1, -1):
			# Refine current neighborhood at this layer
			while True:
				nbr_nodes = self.layered_graph[i][curr_nbr_id]
				if not nbr_nodes:
					break

				nbr_list = sorted(
					[[nbr, self.metric(node, self.node_list[nbr])] for nbr in nbr_nodes],
					key=lambda x: x[1]
				)

				if nbr_list[-1][1] > curr_record:
					curr_nbr_id = nbr_list[-1][0]
					curr_record = nbr_list[-1][1]
				else:
					break

			# Set layer-specific pool sizes
			if i != 0:
				max_candidates = self.pool_factor * self.neighbors_upper
				final_sz = self.neighbors_upper
			else:
				max_candidates = self.pool_factor * self.neighbors_bottom
				final_sz = self.neighbors_bottom

			# Build candidate pool at this layer
			candidate_list = [[curr_nbr_id, self.metric(node, self.node_list[curr_nbr_id])]]
			nodes_added = {curr_nbr_id}
			nbrs_pulled = set()

			while len(candidate_list) < max_candidates:
				next_node_id = None
				for cand in candidate_list:
					if cand[0] not in nbrs_pulled:
						next_node_id = cand[0]
						break

				if next_node_id is None:
					break

				next_nbrs = self.layered_graph[i][next_node_id]
				nbrs_pulled.add(next_node_id)

				for nbr in next_nbrs:
					if nbr not in nodes_added:
						candidate_list.append([nbr, self.metric(node, self.node_list[nbr])])
						nodes_added.add(nbr)

				candidate_list = sorted(candidate_list, key=lambda x: x[1], reverse=True)

			candidate_list = candidate_list[:final_sz]

			# Connect node to selected neighbors
			self.layered_graph[i][node.id] = [x[0] for x in candidate_list]

			# Update connections for neighbors
			for cand in candidate_list:
				nbr = cand[0]

				if len(self.layered_graph[i][nbr]) < final_sz:
					self.layered_graph[i][nbr].append(node.id)
				else:
					lst = [
						[x, self.metric(self.node_list[x], self.node_list[nbr])]
						for x in self.layered_graph[i][nbr]
					]
					lst = sorted(lst, key=lambda x: x[1], reverse=True)

					if cand[1] > lst[-1][1]:
						lst = lst[:-1] + [[node.id, cand[1]]]
						lst = sorted(lst, key=lambda x: x[1], reverse=True)

					self.layered_graph[i][nbr] = [x[0] for x in lst]
		
	
	def build_from_VectorNodeList(self, node_list):
		if self.store_size() > 0:
			raise VectorStoreException("Primary VectorStore already created.  Use single entry mode.")
		if len(node_list) < 2:
			raise VectorStoreException("Cannot build VectorStore with less than 2 nodes.")
			
		if len(node_list) < self.pool_factor * self.neighbors_bottom + 1:
			warnings.warn(
				"Number of nodes is smaller than pool_factor * neighbors_bottom; "
				"bottom-layer candidate pools may be undersized.",
				RuntimeWarning
			)
		
		print("Building Top Layer...")	
		np.random.shuffle(node_list)
		# Assign level to each node and set VectorStore node_list
		curr_max_level = 0
		for i in range(len(node_list)):
			node = node_list[i]
			node.id = i
			lvl = self.choose_level()
			node.max_level = lvl
			if lvl > curr_max_level:
				curr_max_level = lvl
		self.node_list = node_list
		self.max_level = curr_max_level
		
		#build top level
		top_nodes = [node for node in node_list if node.max_level == curr_max_level]
		sym_adj_list, dummy_entry_nbrs, components = self.top_layer_setup(top_nodes, self.neighbors_upper, self.neighbors_upper, self.store_size())
		self.entry_point = dummy_entry_nbrs
		self.component_info = [components]
		self.layered_graph[curr_max_level] = sym_adj_list
		
		print(f"Top Layer built.  Layer {self.max_level} has {len(top_nodes)} nodes.")
		
		#build lower levels
		for i in tqdm(range(curr_max_level-1,-1,-1),desc="Layer"):
			#start with graph from level above
			self.layered_graph[i] = {node_id: nbrs.copy() for node_id, nbrs in self.layered_graph[i+1].items()}
			
			#add new points
			layer_nodes = [node for node in self.node_list if node.max_level == i]
			
			for node in layer_nodes:
				#follow current structure down to layer above current
				nbr_list = sorted([[entry_pt, self.metric(node, self.node_list[entry_pt])] for entry_pt in self.entry_point], key=lambda x: x[1])
				curr_nbr_id = nbr_list[-1][0]
				curr_record = nbr_list[-1][1]
				
				for lvl in range(curr_max_level, i, -1):
					#identify candidate neighbor in layer
					while True:
						nbr_nodes = self.layered_graph[lvl][curr_nbr_id]
						nbr_list = sorted([[nbr, self.metric(node, self.node_list[nbr])] for nbr in nbr_nodes], key=lambda x: x[1])
						if nbr_list[-1][1] > curr_record:
							curr_nbr_id = nbr_list[-1][0]
							curr_record = nbr_list[-1][1]
						else:
							break
							
				#connect node at layer
				if i != 0:
					max_candidates = self.pool_factor * self.neighbors_upper
					final_sz = self.neighbors_upper
				else:
					max_candidates = self.pool_factor * self.neighbors_bottom
					final_sz = self.neighbors_bottom
					
				candidate_list = [[curr_nbr_id, self.metric(node, self.node_list[curr_nbr_id])]]
				nodes_added = {curr_nbr_id}
				nhbrs_pulled = set()
				
				while len(candidate_list) < max_candidates:
					#pull neighbors of best among candidates, check if seen, add that node to nhbrs_pulled, add neighbors to candidate list, sort  
					next_node_id = None
					for cand in candidate_list:
						if cand[0] not in nhbrs_pulled:
							next_node_id = cand[0]
							break
					if next_node_id is None:
						break
						
					next_nhbrs = self.layered_graph[i][next_node_id]
					nhbrs_pulled.add(next_node_id)
					for nbr in next_nhbrs:
						if nbr not in nodes_added:
							candidate_list.append([nbr, self.metric(node, self.node_list[nbr])])
							nodes_added.add(nbr)
					candidate_list = sorted(candidate_list, key=lambda x: x[1], reverse=True)
				
				candidate_list = candidate_list[:final_sz]
				
				self.layered_graph[i][node.id] = [x[0] for x in candidate_list]
				
				# update connections for neighbors (if better or list too small)
				for cand in candidate_list:
					nbr = cand[0]
					if len(self.layered_graph[i][nbr]) < final_sz:
						self.layered_graph[i][nbr].append(node.id)
					else:
						lst = [[x, self.metric(self.node_list[x], self.node_list[nbr])] for x in self.layered_graph[i][nbr]]
						lst = sorted(lst, key=lambda x: x[1], reverse=True)
						if cand[1] > lst[-1][1]:
							lst = lst[:-1] + [[node.id, cand[1]]]
						self.layered_graph[i][nbr] = [x[0] for x in lst]
		
			# get component info for layer
			layer_comps = weak_component_analysis(self.layered_graph[i], self.store_size())
			self.component_info = [layer_comps] + self.component_info
			
		print("Connecting layer components...")
		# create connections between components at each layer for navigability
		for lvl in tqdm(range(self.max_level,-1,-1), desc='Layer'):
			components = self.component_info[lvl]
			if len(components) <= 1:
				continue
				
			sample_sz = [int(np.ceil(np.sqrt(len(comp)))) for comp in components]
			num_connections = [max(1, int(np.log2(len(comp)))) for comp in components]
			
			samples = []
			
			for i, comp in enumerate(components):
				samples.append(np.random.choice(comp, size=sample_sz[i], replace=False))
				
			edge_list = []
			for i in range(len(components)):
				samples_i = samples[i]
				
				for j in range(i+1, len(components)):
					samples_j = samples[j]
					record = -np.inf
					pairs = []
					
					for idx_i, idx_j in product(samples_i, samples_j):
						score = self.metric(self.node_list[idx_i], self.node_list[idx_j])
						pairs.append((idx_i, idx_j, score))
						if score > record:
							record = score
							
					pairs = sorted(pairs, key=lambda x:x[2], reverse=True)
					edge_list.append([i,j,record,pairs])
			
			chosen_edges = KruskalMST(edge_list)
			
			for edge in chosen_edges:
				comp_1 = edge[0]
				comp_2 = edge[1]
				pairs = edge[3]
				budget = min(num_connections[comp_1], num_connections[comp_2])
				
				best_pair = pairs[0]
				pairs = pairs[1:]
				
				ordered_pairs = [best_pair]
				
				seen_1 = {best_pair[0]}
				seen_2 = {best_pair[1]}
				
				# first pass
				for pair in pairs:
					if len(ordered_pairs) >= budget:
						break
						
					if pair[0] not in seen_1 and pair[1] not in seen_2:
						ordered_pairs.append(pair)
						seen_1.add(pair[0])
						seen_2.add(pair[1])
				
				# second pass
				for pair in pairs:
					if len(ordered_pairs) >= budget:
						break
						
					if pair not in ordered_pairs and (pair[0] not in seen_1 or pair[1] not in seen_2):
						ordered_pairs.append(pair)
						seen_1.add(pair[0])
						seen_2.add(pair[1])
				
				# last pass
				for pair in pairs:
					if len(ordered_pairs) >= budget:
						break
						
					if pair not in ordered_pairs:
						ordered_pairs.append(pair)
						
				for pair in ordered_pairs:
					self.layered_graph[lvl][pair[0]].append(pair[1])
					self.layered_graph[lvl][pair[1]].append(pair[0])
	
	def build_vectorstore(self, doc_list):
		#assumes doc_list is a list of dictionaries {'vector':vector, 'data':data, 'source':source}
		#data and source entries can be None
		node_list = []
		for doc in doc_list:
			node_list.append(VectorNode(doc['vector'], doc.get('data'), source=doc.get('source')))
		self.build_from_VectorNodeList(node_list)					

	def find_neighbors(self, doc, k, pool_factor = None):
		if self.store_size() == 0:
			print("Error: VectorStore empty!")
			return []
		if k <= 0:
			return []
			
		query = VectorNode(doc['vector'], doc.get('data'), source=doc.get('source'))
			
		if pool_factor is None:
			pool_factor = self.pool_factor
			
		entry_id_list = sorted([[entry_pt, self.metric(query, self.node_list[entry_pt])] for entry_pt in self.entry_point], key=lambda x: x[1])[-3:]
		
		nbr_candidates = []
		
		for entry in entry_id_list:
			current_node = entry[0]
			current_level = self.max_level
			current_layer = self.layered_graph[current_level]
			curr_record = self.metric(query, self.node_list[current_node])
			
			# Traverse graph down to level 0
			while current_level > 0:
				while True:
					nbrs = current_layer[current_node]
					if not nbrs:
						current_level -= 1
						current_layer = self.layered_graph[current_level]
						break
						
					candidates = sorted([[nbr,self.metric(query, self.node_list[nbr])] for nbr in nbrs], key=lambda x: x[1])
					candidate = candidates[-1]
					
					if candidate[1] > curr_record:
						curr_record = candidate[1]
						current_node = candidate[0]
					else:
						current_level -= 1
						current_layer = self.layered_graph[current_level]
						break
				
			# Find neighbors at level 0
			best_nbr_candidates = [[current_node, curr_record]]
			nbrs_pulled = set()
			nodes_added = {current_node}
			
			search_sz = max(k * pool_factor, self.neighbors_bottom)
			
			while len(best_nbr_candidates) < search_sz:
				nbrs_pulled.add(current_node)
				nbrs = current_layer[current_node]
				for nbr in nbrs:
					entry = [nbr,self.metric(query, self.node_list[nbr])]
					if entry[0] not in nodes_added:
						best_nbr_candidates.append(entry)
						nodes_added.add(entry[0])
				best_nbr_candidates = sorted(best_nbr_candidates, key=lambda x: x[1], reverse=True)
				for cand in best_nbr_candidates:
					if cand[0] not in nbrs_pulled:
						current_node = cand[0]
						break
				if current_node in nbrs_pulled:
					break
			nbr_ids = {nbr[0] for nbr in nbr_candidates}		
			for cand in best_nbr_candidates:
				if cand[0] not in nbr_ids:
					nbr_candidates.append(cand)
					nbr_ids.add(cand[0])
					
		# Sort candidates by score, keep k best, return nodes and scores
		nbr_candidates = sorted(nbr_candidates, key=lambda x: x[1], reverse=True)
		best_nbrs = nbr_candidates[:k]
		return [[self.node_list[nbr[0]], self.metric_multiplier*nbr[1]] for nbr in best_nbrs]
