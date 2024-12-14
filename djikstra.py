# === Generate Results ===

# Here's a C++ implementation of Dijkstra's algorithm with documentation:

# ```python
import heapq
from collections import defaultdict, deque


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))

    def dijkstra(self, start, end):
        dist = [float("inf")] * len(
            self.graph
        )  # Distance to the start node is initially infinity
        visited = [False] * len(self.graph)  # Initialize all vertices as not visited
        prev = [None] * len(self.graph)
        pq_update = []  # Priority queue for vertices with updated distances
        pq_visited = []  # Priority queue for already visited vertices

        dist[start] = 0  # Set starting node distance to zero
        heapq.heappush(
            pq_update, (0, start)
        )  # Add starting node to update priority queue

        while pq_update or pq_visited:
            # Pop the vertex with the smallest distance from either update or visited priority queues
            if not pq_update:
                dist_v, v = heapq.heappop(pq_visited)
            else:
                dist_v, v = heapq.heappushpop(pq_update, heapq.heappop(pq_update))

            # Mark vertex as visited and add its neighbors to pq_update if shorter path is found
            visited[v] = True
            for neighbor, weight in self.graph[v]:
                dist_neighbor = dist[v] + weight
                if not visited[neighbor] and (
                    not pq_visited or dist_neighbor < min(pq_visited)[0]
                ):
                    heapq.heappush(pq_update, (dist_neighbor, neighbor))
                elif not visited[
                    neighbor
                ]:  # If neighbor is already in pq_visited and has a shorter distance, update it there
                    heapq.heapreplace(pq_visited, (dist_neighbor, neighbor))

            # Move vertices from pq_update to pq_visited when their distance is finalized
            if not visited[v]:
                heapq.heappush(pq_visited, (dist_v, v))

        # Reconstruct shortest path using BFS starting from the end node and return it
        path = deque()
        curr = end
        while prev[curr]:
            path.appendleft(curr)
            curr = prev[curr]
        if start not in path:
            return None, None  # No path found

        path.appendleft(start)
        return list(path), dist[end]
