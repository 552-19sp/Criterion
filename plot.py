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

    plt.show()

    """
    print("X axis: "  + str(x_axis))
    print("Lines: " + str(line_label))
    print("Lines Mapping: " + str(y_map))
    print("Min/Max error: " + str(error_bars))
    """

if __name__ == "__main__":
    if  len(sys.argv) != 2:
        print("usage: python3 plot.py <data csv>")
    else:
        CSV = sys.argv[1]
        with open('results.csv', mode='r') as input_file:
            ingest(input_file)
