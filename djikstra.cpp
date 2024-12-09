=== Generate Results ===

Here's a C++ implementation of Dijkstra's algorithm with documentation:

```c++
// Graph class representing a graph using adjacency list representation
class Graph {
public:
    // Define a structure for a node in the graph
    struct Node {
        int vertex;   // Vertex number of this node
        int weight;  // Weight of the edge leading to this node from source vertex
        Node* next; // Pointer to the next node in the adjacency list for this vertex
    };

    // Define a structure for the adjacency list representation of the graph
    struct AdjList {
        int vertex;   // Vertex number of this node
        int weight;  // Weight of the edge leading to this node from source vertex
        Node* next; // Pointer to the next node in the adjacency list for this vertex
    };

    // Map representing graph. Key is vertex number, value is AdjList structure
    unordered_map<int, AdjList*> map;
};

// Function to add an edge to the graph
void addEdge(Graph& graph, int src, int dest, int weight) {
    // Create a new node and initialize its properties
    Node* newNode = new Node();
    newNode->vertex = dest;
    newNode->weight = weight;
    
    // If the adjacency list for this vertex doesn't exist, create one and add it to the graph map
    if (graph.map.find(src) == graph.map.end()) {
        AdjList* node = new AdjList();
        node->vertex = src;
        node->next = nullptr;
        graph.map[src] = node;
    }

    // Add the edge to the adjacency list of the source vertex
    Node** head = &(graph.map[src]->next);
    while ((*head) != nullptr && (*head)->vertex < dest) {
        head = &((*head)->next);
    }
    if ((*head) == nullptr || (*head)->vertex > dest) {
        (*head) = newNode;
    } else if ((*head)->weight > weight) { // If the edge already exists and the incoming weight is smaller, update it
        Node* temp = *head;
        *head = newNode;
        newNode->next = temp;
    }
}

// Dijkstra's algorithm implementation for finding shortest path from source vertex to all other vertices in the graph
void dijkstra(Graph& graph, int src) {
    // Initialize distances and predecessors arrays with infinite values and set source distance as 0
    vector<int> distances(graph.map.size(), INT_MAX);
    vector<int> predecessors(graph.map.size(), -1);
    distances[src] = 0;

    // Initialize a priority queue to store the vertices in order of their distance from source vertex
    priority_queue<pair<int, int>, vector<pair<int, int>, greater<pair<int, int>>> > pq;
    for (auto it = graph.map.begin(); it != graph.map.end(); ++it) {
        if (it->first != src) {
            pq.push({ it->second->weight, it->first });
        }
    }

    // Loop until the priority queue is empty
    while (!pq.empty()) {
        int vertex = pq.top().second;
        int distance = pq.top().first;
        pq.pop();

        // Check if we have already processed this vertex or if its distance is greater than the current shortest path
        if (distances[vertex] != INT_MAX && distances[vertex] <= distance) {
            continue;
        }

        // Update the distance of all adjacent vertices and their predecessors
        for (Node* adj = graph.map[vertex]; adj != nullptr; adj = adj->next) {
            int newDistance = distances[vertex] + adj->weight;
            if (distances[adj->vertex] > newDistance) {
                distances[adj->vertex] = newDistance;
                predecessors[adj->vertex] = vertex;

                // Update the priority queue with the new distance of this vertex
                pq.push({ newDistance, adj->vertex });
            }
        }
    }

    // Print the shortest path from source to all other vertices and their distances
    for (int i = 0; i < graph.map.size(); ++i) {
        if (i != src) {
            cout << "Distance from source " << src << " to vertex " << i << ": " << distances[i] << endl;
            int prevVertex = predecessors[i];
            while (prevVertex != -1) {
                cout << "  --> " << prevVertex << endl;
                prevVertex = predecessors[prevVertex];
            }
        }
    }
}
```