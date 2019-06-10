""" 
Takes one argument: a csv of the form
        num_clients,num_servers,drop_rate,failure_code,num_ops,avg_latency
where each of the attributes are ints.

Ingest the data to be in a more easily manipulatable

TODO (jackkhuu): Plotting using the ingested data
"""

import sys 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as ps


CSV = ""
PLOTS_PER_GRAPH = 5

"""
Returns a DataFrame with the Error Bars given each of the configurations

input_file: File

"""
def ingest(input_file):
    df = pd.read_csv(CSV, names=[
        'num_clients', 'num_servers', 'drop_rate', 'failure_code', 'num_ops', 'avg_latency'])

    # Meta Datapoint: Average latency across all iteration of same test
    base_data_query = "SELECT num_clients, num_servers, drop_rate, failure_code, num_ops, \
            MIN(avg_latency) AS min_latency, MAX(avg_latency) AS max_latency, \
            AVG(avg_latency) AS avg_latency \
            FROM df \
            GROUP BY num_clients, num_servers, drop_rate, failure_code, num_ops"
    base_data = ps.sqldf(base_data_query, locals())
    return base_data


"""
Plots a line graph containing the topN configurations with the highest average latency across all num_ops

base_data: DataFrame

"""
def plotTopNLatency(base_data):
    # Line of each setting: Average latency across the same settings across num_ops
    line_query = "SELECT num_clients, num_servers, drop_rate, failure_code, AVG(avg_latency) AS avg \
            FROM base_data \
            GROUP BY num_clients, num_servers, drop_rate, failure_code \
            ORDER BY AVG(avg_latency)"
    line = ps.sqldf(line_query, locals())
    
    # Best lines: The topN average latencies across num_ops
    topN = line.iloc[:PLOTS_PER_GRAPH, :]

    # Find the data points of each of the best lines
    best_line_points = []
    line_label = []

    best_lines = (topN.values.tolist())
    for setup in best_lines:
        line_label.append(" ".join([str(i) for i in setup]))
        clients_, servers_, drop_, failure_, avg_ = setup 
        data_point_query = "SELECT num_ops, min_latency, max_latency, avg_latency \
                            FROM base_data  \
                            WHERE num_clients = \"" + str(clients_)  + "\" \
                            AND num_servers = \"" + str(servers_) + "\" \
                            AND drop_rate = \"" + str(drop_) + "\" \
                            AND failure_code = \"" + str(failure_) + "\" \
                            ORDER BY num_ops"
        data_point = ps.sqldf(data_point_query, locals())
        best_line_points.append(data_point)

    # Plotting set up
    x_axis = best_line_points[0].loc[:, 'num_ops'].values

    y_map = {}
    error_bars = []
    for (name, lp) in zip(line_label, best_line_points):
        y_val = lp.loc[:, 'avg_latency'].values
        y_map[name] = y_val

        y_min = lp.loc[:, 'min_latency'].values
        y_max = lp.loc[:, 'max_latency'].values
        error_bars.append((y_min, y_max))

    # Plotting
    graph = pd.DataFrame(y_map, index= x_axis, columns=line_label)
    plot = graph.plot()
    plot.set_xlabel("Throughput")
    plot.set_ylabel("Latency")
    plot.set_title("Best: Throughput VS Latency")

    # Plotting: Error Bars
    for (name, error) in zip(line_label, error_bars):
        plot.errorbar(x_axis, y_map[name], yerr=[error[0], error[1]])


"""
Plots a bar graph containing the top configurations with the highest average latency across all num_ops for each number of replica for UDP and TCP

tcp_data: DataFrame
udp_data: DataFrame
"""
def plotReplicaComparison(tcp_data, udp_data):
    tcp_data_latency = []
    udp_data_latency = []

    x_axis = ['3', '5']
    bar_labels = ['TCP', 'UDP']

    # TCP
    for replica_count in x_axis:
        # Average latency across the same settings across num_ops
        tcp_table = tcp_data
        replica_query = "SELECT num_clients, num_servers, drop_rate, failure_code, AVG(avg_latency) AS avg \
                FROM tcp_table \
                WHERE num_servers = " + replica_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        replica = ps.sqldf(replica_query, locals())
        best = replica.iloc[0, :]
        tcp_data_latency.append(best.loc['avg'])
    
    # UDP
    for replica_count in x_axis:
        # Average latency across the same settings across num_ops
        udp_table = udp_data
        replica_query = "SELECT num_clients, num_servers, drop_rate, failure_code, AVG(avg_latency) AS avg \
                FROM udp_table \
                WHERE num_servers = " + replica_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        replica = ps.sqldf(replica_query, locals())
        best = replica.iloc[0, :]
        udp_data_latency.append(best.loc['avg'])

    y_map = {bar_labels[0]: tcp_data_latency, bar_labels[1]: udp_data_latency}

    # Plotting
    graph = pd.DataFrame(y_map, index= x_axis, columns=bar_labels)
    plot = graph.plot.bar(rot=0)
    plot.set_xlabel("Number of Replica")
    plot.set_ylabel("Latency")
    plot.set_title("TCP VS UDP: Best Replica Latency")


if __name__ == "__main__":
    if  (len(sys.argv) != 2 and len(sys.argv) != 3):
        print("usage: python3 plot.py <data csv>")
        print("usage: python3 plot.py <TCP csv> <UDP csv>")
    else:
        CSV = sys.argv[1]

        # Meta DataFrame of csv
        base_data = 0
        with open(CSV, mode='r') as input_file:
            base_data = ingest(input_file)

        plotTopNLatency(base_data)
        plotReplicaComparison(base_data, base_data)
        plt.show()
