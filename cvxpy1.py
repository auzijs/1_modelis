import cvxpy as cp
import numpy as np
from warehouse_generator import generate_warehouse_dataset, create_graph_from_data, set_node_positions
import matplotlib.pyplot as plt
import networkx as nx

def main():
    num_items = 150
    num_orders = 24
    num_aisles = 9
    max_items_per_order = 4
    max_items_per_aisle = 20
    random_seed = 10 
    num_batches = 8
    batch_size = 3
    aisle_length = 200
    aisle_width = 12

    warehouse_data = generate_warehouse_dataset(num_items, num_aisles, num_orders, max_items_per_order, max_items_per_aisle, random_seed)
    warehouse_data.to_csv('warehouse_orders2.csv', index=False)
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

    X = cp.Variable((num_orders, num_batches), boolean=True)
    Y = cp.Variable((num_aisles, num_batches), boolean=True)
    LastA = cp.Variable(num_batches, integer=True)
    N = cp.Variable(num_batches, integer=True)
    D = cp.Variable(num_batches)

    A_matrix = np.zeros((num_aisles, num_orders))

    for i in range(num_orders):
        for m in range(num_aisles):
            if m+1 in orders_aisles[i+1]:
                A_matrix[m, i] = 1

    A = cp.Parameter((num_aisles, num_orders), integer=True)
    A.value = A_matrix

    objective = cp.Minimize(cp.sum(D))

    constraints = []
    constraints += [cp.sum(X, axis=0) <= batch_size]

    constraints += [cp.sum(X, axis=1) == 1]

    M = num_orders + 1
    constraints += [Y * M >= A @ X]

    for k in range(num_batches):
        constraints += [LastA[k] >= cp.multiply(Y[:, k], np.arange(1, num_aisles+1))]

    constraints += [N == cp.sum(Y, axis=0)]

    constraints += [D == (((LastA - 1) * 2 * aisle_width) + (aisle_length * N) + (LastA * 2 * aisle_width) + aisle_length * (N + 1)) / 2]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CPLEX)

    if (problem.status == cp.OPTIMAL) or (problem.status == cp.OPTIMAL_INACCURATE):
        if (problem.status == cp.OPTIMAL_INACCURATE):
            print('INNACURATE!')
        print('Solution:')
        print('Objective value =', problem.value)
        for k in range(num_batches):
            print(f'Batch {k+1}:')
            orders_in_batch = [i+1 for i in range(num_orders) if X.value[i, k] > 0.5]
            aisles_in_batch = [m+1 for m in range(num_aisles) if Y.value[m, k] > 0.5]
            print('Orders:', ', '.join(str(order) for order in orders_in_batch))
            print('Aisles:', ', '.join(str(aisle) for aisle in aisles_in_batch))
            print('Last Aisle:', LastA.value[k])
            print('Total Aisles:', N.value[k])
            print('Total Distance:', D.value[k])
            print()
    else:
        print('No solution found!')

if __name__ == '__main__':
    main()
