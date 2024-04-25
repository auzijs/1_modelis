import pyomo.environ as pyo
from warehouse_generator import generate_warehouse_dataset, create_graph_from_data, set_node_positions
import matplotlib.pyplot as plt
import networkx as nx


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
    warehouse_data.to_csv('dataset3.csv', index=False)
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
    model = pyo.ConcreteModel()

    model.orders = pyo.RangeSet(1, num_orders)
    model.batches = pyo.RangeSet(1, num_batches)
    model.aisles = pyo.RangeSet(1, num_aisles)


    def order_aisle_matrix(model, m, i):
        return 1 if m in orders_aisles[i] else 0
    model.A = pyo.Param(model.aisles, model.orders, initialize=order_aisle_matrix, within=pyo.Binary)

    model.X = pyo.Var(model.orders, model.batches, within=pyo.Binary)
    model.Y = pyo.Var(model.aisles, model.batches, within=pyo.Binary)
    model.LastA = pyo.Var(model.batches)
    model.N = pyo.Var(model.batches, within=pyo.NonNegativeIntegers, bounds=(1, num_aisles+1))
    model.D = pyo.Var(model.batches, within=pyo.NonNegativeReals)

    def objective_rule(model):
        return sum(model.D[k] for k in model.batches)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def batch_size_rule(model, k):
        return sum(model.X[i, k] for i in model.orders) <= batch_size
    model.batch_size_constraint = pyo.Constraint(model.batches, rule=batch_size_rule)


    def assign_orders_rule(model, i):
        return sum(model.X[i, k] for k in model.batches) == 1
    model.assign_orders = pyo.Constraint(model.orders, rule=assign_orders_rule)

    M = num_orders + 1
    def aisle_visit_rule(model, m, k):
        return model.Y[m, k] * M >= sum(model.A[m, i] * model.X[i, k] for i in model.orders)
    model.aisle_visit = pyo.Constraint(model.aisles, model.batches, rule=aisle_visit_rule)


    def last_aisle_rule(model, k, m):
        return model.LastA[k] >= m * model.Y[m, k]
    model.last_aisle = pyo.Constraint(model.batches, model.aisles, rule=last_aisle_rule)

    def count_aisles_rule(model, k):
        return model.N[k] == sum(model.Y[m, k] for m in model.aisles)
    model.count_aisles = pyo.Constraint(model.batches, rule=count_aisles_rule)

    def calculate_distance_rule(model, k):
        return model.D[k] == (((model.LastA[k] - 1) * 2 * aisle_width) + (aisle_length * model.N[k]) + (model.LastA[k] * 2 * aisle_width) + aisle_length * (model.N[k] + 1))/2
    model.calculate_distance = pyo.Constraint(model.batches, rule=calculate_distance_rule)

    solver = pyo.SolverFactory('cplex')

    try:
        solution = solver.solve(model, tee=True)
    except KeyboardInterrupt:
        print("Solver was interrupted.")
        exit()

    if (solution.solver.termination_condition == pyo.TerminationCondition.optimal):
        print('Solution:')
        print("Objective expression:", model.objective.expr)
        total_distance = 0
        for k in model.batches:
            print(f'Batch {k}:')
            orders = [i for i in model.orders if model.X[i, k].value > 0.5]
            aisles = [m for m in model.aisles if model.Y[m, k].value > 0.5]
            print('Orders:', ', '.join(str(order) for order in orders))
            print('Aisles:', ', '.join(str(aisle) for aisle in aisles))
            print('Last Aisle:', model.LastA[k].value)
            print('Total Aisles:', model.N[k].value)
            print('Total Distance:', model.D[k].value)
            print()
            total_distance += model.D[k].value
        print(f"Total Distance: {total_distance}")
    else:
        print('No solution found!')

if __name__ == '__main__':
    main()
