# Approximate Nearest Neighbor (ANN) VectorStore

## Overview

This project implements a prototype Approximate Nearest Neighbor (ANN) search structure based on a layered graph approach inspired by hierarchical navigable small world (HNSW)-style methods.

The goal of this project is not to produce a production-optimized ANN system, but rather to:

- Demonstrate understanding of ANN design principles  
- Explore tradeoffs between accuracy (recall), speed, and graph structure  
- Provide a clean, readable implementation suitable for experimentation and extension  

---

## Design Philosophy

### 1. Layered Graph Structure
Each vector is assigned a random level (via a geometric distribution), producing a multi-layer graph:

- Higher layers: sparse, long-range connections (fast navigation)
- Lower layers: dense, local connections (accurate refinement)

---

### 2. Metric Abstraction
The system supports two similarity measures:

- Cosine similarity
- Euclidean distance

Euclidean distance is internally converted to a similarity score (negative distance), and results are returned in natural units.

---

### 3. Controlled Graph Connectivity
Each node maintains a bounded number of neighbors:

- neighbors_upper: connections in higher layers  
- neighbors_bottom: connections in base layer  

Candidate neighbors are selected using a greedy expansion + pruning strategy:
- Explore a candidate pool (pool_factor × neighbors)
- Retain only the best connections  

---

### 4. Component Repair (Connectivity Guarantee)
Disconnected components are detected and reconnected using a minimum spanning tree (Kruskal-based), ensuring global navigability.

---

### 5. Entry Points via Medoids
Top-layer entry points are selected using medoid-like representatives from each component.

---

### 6. Two Build Modes

- Batch construction (build_vectorstore)  
- Incremental insertion (add_single_node)  

---

## Usage

### 1. Creating a VectorStore

from layered_graph_ann import VectorStore

vs = VectorStore(
    exp_level=2,
    neighbors_bottom=16,
    neighbors_upper=8,
    metric='cosine',
    pool_factor=5
)

---

### 2. Preparing Data

doc_list = [
    {'vector': [0.1, 0.2, 0.3], 'data': "point A"},
    {'vector': [0.4, 0.5, 0.6], 'data': "point B"},
]

---

### 3. Building the VectorStore

vs.build_vectorstore(doc_list)

---

### 4. Querying the VectorStore

query = {'vector': [0.2, 0.1, 0.3]}
neighbors = vs.find_neighbors(query, k=5)

---

### 5. Inspecting Results

for node, score in neighbors:
    print(node.id, score, node.data)

---

### 6. Incremental Insertion

vs.add_single_node({'vector': [0.3, 0.2, 0.1], 'data': "new point"})

(Only valid after initial build)
