import pulp as pl
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from warehouse_generator import generate_warehouse_dataset, create_graph_from_data, set_node_positions
import time

def main():
    num_items = 200
    num_orders = 50
    num_aisles = 9
    max_items_per_order = 12
    max_items_per_aisle = 30
    random_seed = 10 
    num_batches = 5
    batch_size = 10
    aisle_length = 200
    aisle_width = 12

    warehouse_data = generate_warehouse_dataset(num_items, num_aisles, num_orders, max_items_per_order, max_items_per_aisle, random_seed)
    warehouse_data.to_csv('warehouse_orders1.csv', index=False)
    warehouse_graph = create_graph_from_data(warehouse_data)

    orders_aisles = {order: [] for order in range(1, num_orders + 1)}
    for _, row in warehouse_data.iterrows():
        orders_aisles[row['Order']].append(row['Aisle'])

    plt.figure(figsize=(10, 8))
    pos = set_node_positions(warehouse_data)
    labels = {node: data['label'] for node, data in warehouse_graph.nodes(data=True)}
    nx.draw_networkx_nodes(warehouse_graph, pos, nodelist=[node for node in warehouse_graph.nodes if warehouse_graph.nodes[node]['type'] == 'order'], node_color='green', node_size=200, label='Orders')
    nx.draw_networkx_nodes(warehouse_graph, pos, nodelist=[node for node in warehouse_graph.nodes if warehouse_graph.nodes[node]['type'] == 'aisle'], node_color='blue', node_size=200, label='Aisles')
    nx.draw_networkx_nodes(warehouse_graph, pos, nodelist=[node for node in warehouse_graph.nodes if warehouse_graph.nodes[node]['type'] == 'item'], node_color='red', node_size=200, label='Items')
    nx.draw_networkx_nodes(warehouse_graph, pos, nodelist=[node for node in warehouse_graph.nodes if warehouse_graph.nodes[node]['type'] == 'end_of_aisle'], node_color='yellow', node_size=200, label='Ends of Aisles')
    nx.draw_networkx_nodes(warehouse_graph, pos, nodelist=[node for node in warehouse_graph.nodes if warehouse_graph.nodes[node]['type'] == 'depot'], node_color='black', node_size=300, label='Depot')
    nx.draw_networkx_edges(warehouse_graph, pos, edgelist=warehouse_graph.edges(), alpha=0.5)
    nx.draw_networkx_labels(warehouse_graph, pos, labels)
    plt.title('Warehouse orders representation')
    plt.axis('off')
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

    model = pl.LpProblem("Warehouse_Batching", pl.LpMinimize)
    
    X = pl.LpVariable.dicts("X", [(i, k) for i in range(1, num_orders + 1) for k in range(1, num_batches + 1)], cat=pl.LpBinary)
    Y = pl.LpVariable.dicts("Y", [(m, k) for m in range(1, num_aisles + 1) for k in range(1, num_batches + 1)], cat=pl.LpBinary)
    A = {(m, i): 1 if m in orders_aisles[i] else 0 for m in range(1, num_aisles + 1) for i in range(1, num_orders + 1)}
    LastA = pl.LpVariable.dicts("LastA", range(1, num_batches + 1), 1, num_aisles + 1, cat=pl.LpInteger)
    N = pl.LpVariable.dicts("N", range(1, num_batches + 1), 1, num_aisles + 1, cat=pl.LpInteger)
    D = pl.LpVariable.dicts("D", range(1, num_batches + 1), lowBound=0, cat=pl.LpContinuous)

    model += pl.lpSum(D[k] for k in range(1, num_batches + 1))

    for k in range(1, num_batches + 1):
        model += pl.lpSum(X[(i, k)] for i in range(1, num_orders + 1)) <= batch_size

    for i in range(1, num_orders + 1):
        model += pl.lpSum(X[(i, k)] for k in range(1, num_batches + 1)) == 1

    M = num_orders + 1
    for m in range(1, num_aisles + 1):
        for k in range(1, num_batches + 1):
            model += Y[(m, k)] * M >= pl.lpSum(A[(m, i)] * X[(i, k)] for i in range(1, num_orders + 1))

    for k in range(1, num_batches + 1):
        for m in range(1, num_aisles + 1):
            model += LastA[k] >= m * Y[(m, k)]

    for k in range(1, num_batches + 1):
        model += N[k] == pl.lpSum(Y[(m, k)] for m in range(1, num_aisles + 1))

    for k in range(1, num_batches + 1):
        model += D[k] == (((LastA[k] - 1) * 2 * aisle_width) + (aisle_length * N[k]) + (LastA[k] * 2 * aisle_width) + aisle_length * (N[k] + 1))/2

    solver = pl.CPLEX_CMD()

    try:
        start_time = time.time()
        model.solve(solver)
        end_time = time.time()
    except KeyboardInterrupt:
        print("Solver was interrupted.")
        exit()

    if pl.LpStatus[model.status] == 'Optimal':
        print('Solution:')
        print('Objective value =', pl.value(model.objective))
        print("Solution time: ", end_time - start_time)
        total_distance = 0
        for k in range(1, num_batches + 1):
            print(f'Batch {k}:')
            orders = [i for i in range(1, num_orders + 1) if X[(i, k)].value() > 0.5]
            aisles = [m for m in range(1, num_aisles + 1) if Y[(m, k)].value() > 0.5]
            print('Orders:', ', '.join(str(order) for order in orders))
            print('Aisles:', ', '.join(str(aisle) for aisle in aisles))
            print('Last Aisle:', LastA[k].value())
            print('Total Aisles:', N[k].value())
            print('Total Distance:', D[k].value())
            print()
            total_distance += D[k].value()
        print(f"Total Distance: {total_distance}")
    else:
        print('No solution found!')

if __name__ == '__main__':
    main()
