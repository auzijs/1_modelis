import pandas as pd
import random
import networkx as nx
import numpy as np

def generate_warehouse_dataset(num_items, num_aisles, num_orders, max_items_per_order, max_items_per_aisle, seed):
    random.seed(seed)
    aisle_capacity = {i: 0 for i in range(1, num_aisles + 1)}
    items = {}

    for i in range(1, num_items + 1):
        valid_aisles = [aisle for aisle, count in aisle_capacity.items() if count < max_items_per_aisle]
        if not valid_aisles:
            raise ValueError("All aisles have reached their maximum capacity")
        
        selected_aisle = random.choice(valid_aisles)
        items[i] = selected_aisle
        aisle_capacity[selected_aisle] += 1

    orders = {}
    for order_id in range(1, num_orders + 1):
        order_items = random.sample(list(items.keys()), random.randint(1, max_items_per_order))
        orders[order_id] = {item: items[item] for item in order_items}

    order_list = []
    for order, items_in_order in orders.items():
        for item, aisle in items_in_order.items():
            order_list.append({'Order': order, 'Item': item, 'Aisle': aisle})

    return pd.DataFrame(order_list)

def create_graph_from_data(df):
    G = nx.Graph()

    orders = sorted(df['Order'].unique())
    aisles = sorted(df['Aisle'].unique())
    items = sorted(df['Item'].unique())


    G.add_node("Depot", type='depot', label='')
    
    for order in orders:
        G.add_node(f"Order {order}", type='order', label=order)

    for aisle in aisles:
        G.add_node(f"Aisle {aisle}", type='aisle', label=aisle)
        G.add_node(f"End of Aisle {aisle}", type='end_of_aisle', label='')

    for item in items:
        G.add_node(f"Item {item}", type="item", label=item)


    for _, row in df.iterrows():
        order_node = f"Order {row['Order']}"
        aisle_node = f"Aisle {row['Aisle']}"
        end_of_aisle_node = f"End of Aisle {row['Aisle']}"
        item_node = f"Item {row['Item']}"
        G.add_edge(item_node, aisle_node)
        G.add_edge(item_node, order_node)
        G.add_edge(aisle_node, end_of_aisle_node)

    for i in range(len(aisles) - 1):
        G.add_edge(f"Aisle {aisles[i]}", f"Aisle {aisles[i+1]}")
        G.add_edge(f"End of Aisle {aisles[i]}", f"End of Aisle {aisles[i+1]}")

    return G

def set_node_positions(warehouse_df):
    pos = {}
    aisles = sorted(warehouse_df['Aisle'].unique())
    aisle_positions = {aisle: idx * 10 for idx, aisle in enumerate(aisles)}
    order_positions = {}
    for order in warehouse_df['Order'].unique():
        items_in_order = warehouse_df[warehouse_df['Order'] == order]
        average_aisle = items_in_order['Aisle'].mean()

        if average_aisle > len(aisles) / 2:
            order_positions[order] = max(aisle_positions.values()) + 20
        else:
            order_positions[order] = min(aisle_positions.values()) - 20

    order_y_positions = np.linspace((len(warehouse_df['Order'].unique()) - 1) * 10, 0, len(warehouse_df['Order'].unique()))
    for order, y_pos in zip(sorted(warehouse_df['Order'].unique()), order_y_positions):
        pos[f"Order {order}"] = (order_positions[order], y_pos)

    aisle_height = max(order_y_positions) if order_y_positions.size > 0 else 50
    for aisle in aisles:
        pos[f"Aisle {aisle}"] = (aisle_positions[aisle], 0)
        pos[f"End of Aisle {aisle}"] = (aisle_positions[aisle], aisle_height)

    for aisle in aisles:
        items_in_aisle = warehouse_df[warehouse_df['Aisle'] == aisle]['Item'].unique()
        y_positions = np.linspace(10, aisle_height - 10, len(items_in_aisle) + 2)[1:-1]
        for item, y_pos in zip(sorted(items_in_aisle), y_positions):
            pos[f"Item {item}"] = (aisle_positions[aisle], y_pos)

    depot_x_position = min(aisle_positions.values()) - 5
    depot_y_position = -10
    pos["Depot"] = (depot_x_position, depot_y_position)

    return pos


