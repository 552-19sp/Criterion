""" 
Takes one argument: a csv of the form
        num_clients,num_servers,drop_rate,failure_code,num_ops,avg_latency
where each of the attributes are ints.

Ingest the data to be in a more easily manipulatable

"""

import sys 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as ps

ALGORITHM = 'Mencius'
PLOTS_PER_GRAPH = 5
REPLICA = ['3', '5']
CLIENT = ['1', '2', '3']
DROP = ['0', '2', '5']
FAIL = ['0', '1']

"""
Returns a DataFrame with the Error Bars given each of the configurations

input_file: File

"""
def ingest(input_file):
    df = pd.read_csv(input_file, names=[
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
def plotTopNLatency(base_data, label):
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
    error_bars = [[]]
    for (name, lp) in zip(line_label, best_line_points):
        y_val = lp.loc[:, 'avg_latency'].values
        y_map[name] = y_val

        y_min = lp.loc[:, 'min_latency'].values
        y_max = lp.loc[:, 'max_latency'].values
        error_bars.append((y_val - y_min, y_max - y_val))

    # Plotting
    graph = pd.DataFrame(y_map, index= x_axis, columns=line_label)
    plot = graph.plot(yerr=error_bars[1:])
    plot.set_xlabel("Throughput")
    plot.set_ylabel("Latency")
    plot.set_title("Top Throughput VS Latency: " + label)


"""
Plots a bar graph containing the top configurations with the highest average latency across all num_ops for each number of replica for UDP and TCP

tcp_data: DataFrame
udp_data: DataFrame
"""
def plotReplicaComparison(tcp_data, udp_data, label):
    tcp_data_latency = []
    udp_data_latency = []

    x_axis = REPLICA
    bar_labels = ['TCP', 'UDP']

    tcp_error_bars = []
    # TCP
    for replica_count in x_axis:
        # Average latency across the same settings across num_ops
        tcp_table = tcp_data
        replica_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM tcp_table \
                WHERE num_servers = " + replica_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        replica = ps.sqldf(replica_query, locals())
        best = replica.iloc[0, :]

        tcp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        tcp_error_bars.append((y_min, y_max))

    tcpError = [[error[0] for error in tcp_error_bars], [error[1] for error in tcp_error_bars]]
    
    udp_error_bars = []
    # UDP
    for replica_count in x_axis:
        # Average latency across the same settings across num_ops
        udp_table = udp_data
        replica_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM udp_table \
                WHERE num_servers = " + replica_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        replica = ps.sqldf(replica_query, locals())
        best = replica.iloc[0, :]

        udp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        udp_error_bars.append((y_min, y_max))

    udpError = [[a[0] for a in udp_error_bars], [a[1] for a in udp_error_bars]]

    y_map = {bar_labels[0]: tcp_data_latency, bar_labels[1]: udp_data_latency}

    # Plotting
    graph = pd.DataFrame(y_map, index= x_axis, columns=bar_labels)
    plot = graph.plot.bar(rot=0, yerr= [tcpError, udpError])
    plot.set_xlabel("Number of Replica")
    plot.set_ylabel("Latency")
    plot.set_title(label + " TCP VS UDP: Replica Latency")


"""
Plots a bar graph containing the top configurations with the highest average latency across all num_ops for each number of clients for UDP and TCP

tcp_data: DataFrame
udp_data: DataFrame
"""
def plotClientComparison(tcp_data, udp_data, label):
    tcp_data_latency = []
    udp_data_latency = []

    x_axis = CLIENT 
    bar_labels = ['TCP', 'UDP']

    tcp_error_bars = []
    # TCP
    for client_count in x_axis:
        # Average latency across the same settings across num_ops
        tcp_table = tcp_data
        client_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM tcp_table \
                WHERE num_clients = " + client_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        client = ps.sqldf(client_query, locals())
        best = client.iloc[0, :]

        tcp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        tcp_error_bars.append((y_min, y_max))

    tcpError = [[error[0] for error in tcp_error_bars], [error[1] for error in tcp_error_bars]]
    
    udp_error_bars = []
    # UDP
    for client_count in x_axis:
        # Average latency across the same settings across num_ops
        udp_table = udp_data
        client_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM udp_table \
                WHERE num_clients = " + client_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        client = ps.sqldf(client_query, locals())
        best = client.iloc[0, :]

        udp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        udp_error_bars.append((y_min, y_max))

    udpError = [[a[0] for a in udp_error_bars], [a[1] for a in udp_error_bars]]

    y_map = {bar_labels[0]: tcp_data_latency, bar_labels[1]: udp_data_latency}

    # Plotting
    graph = pd.DataFrame(y_map, index= x_axis, columns=bar_labels)
    plot = graph.plot.bar(rot=0, yerr= [tcpError, udpError])
    plot.set_xlabel("Number of Clients")
    plot.set_ylabel("Latency")
    plot.set_title(label + " TCP VS UDP: Client Latency")


"""
Plots a bar graph containing the top configurations with the highest average latency across all num_ops for each drop rate for UDP and TCP

tcp_data: DataFrame
udp_data: DataFrame
"""
def plotDropComparison(tcp_data, udp_data, label):
    tcp_data_latency = []
    udp_data_latency = []

    x_axis = DROP 
    bar_labels = ['TCP', 'UDP']

    tcp_error_bars = []
    # TCP
    for drop_count in x_axis:
        # Average latency across the same settings across num_ops
        tcp_table = tcp_data
        drop_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM tcp_table \
                WHERE drop_rate = " + drop_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        drop = ps.sqldf(drop_query, locals())
        best = drop.iloc[0, :]

        tcp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        tcp_error_bars.append((y_min, y_max))

    tcpError = [[error[0] for error in tcp_error_bars], [error[1] for error in tcp_error_bars]]
    
    udp_error_bars = []
    # UDP
    for drop_count in x_axis:
        # Average latency across the same settings across num_ops
        udp_table = udp_data
        drop_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM udp_table \
                WHERE drop_rate = " + drop_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        drop = ps.sqldf(drop_query, locals())
        best = drop.iloc[0, :]

        udp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        udp_error_bars.append((y_min, y_max))

    udpError = [[a[0] for a in udp_error_bars], [a[1] for a in udp_error_bars]]

    y_map = {bar_labels[0]: tcp_data_latency, bar_labels[1]: udp_data_latency}

    # Plotting
    graph = pd.DataFrame(y_map, index= x_axis, columns=bar_labels)
    plot = graph.plot.bar(rot=0, yerr= [tcpError, udpError])
    plot.set_xlabel("Drop Rate (Per 1000)")
    plot.set_ylabel("Latency")
    plot.set_title(label + " TCP VS UDP: Drop Rate Latency")
    
    
"""
Plots a bar graph containing the top configurations with the highest average latency across all num_ops for each fail situation for UDP and TCP

tcp_data: DataFrame
udp_data: DataFrame
"""
def plotFailComparison(tcp_data, udp_data, label):
    tcp_data_latency = []
    udp_data_latency = []

    x_axis = FAIL 
    bar_labels = ['TCP', 'UDP']

    tcp_error_bars = []
    # TCP
    for fail_count in x_axis:
        # Average latency across the same settings across num_ops
        tcp_table = tcp_data
        fail_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM tcp_table \
                WHERE failure_code = " + fail_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        fail = ps.sqldf(fail_query, locals())
        best = fail.iloc[0, :]

        tcp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        tcp_error_bars.append((y_min, y_max))

    tcpError = [[error[0] for error in tcp_error_bars], [error[1] for error in tcp_error_bars]]
    
    udp_error_bars = []
    # UDP
    for fail_count in x_axis:
        # Average latency across the same settings across num_ops
        udp_table = udp_data
        fail_query = "SELECT num_clients, num_servers, drop_rate, failure_code, \
                MIN(avg_latency) AS min, MAX(avg_latency) as max, AVG(avg_latency) AS avg \
                FROM udp_table \
                WHERE failure_code = " + fail_count + " GROUP BY \
                num_clients, num_servers, drop_rate, failure_code \
                ORDER BY AVG(avg_latency)"
        fail = ps.sqldf(fail_query, locals())
        best = fail.iloc[0, :]

        udp_data_latency.append(best.loc['avg'])
        y_min = best.loc['avg'] - best.loc['min']
        y_max = best.loc['max'] - best.loc['avg']
        udp_error_bars.append((y_min, y_max))

    udpError = [[a[0] for a in udp_error_bars], [a[1] for a in udp_error_bars]]

    y_map = {bar_labels[0]: tcp_data_latency, bar_labels[1]: udp_data_latency}

    # Plotting
    graph = pd.DataFrame(y_map, index= ['No Failure', 'Random Failure'], columns=bar_labels)
    plot = graph.plot.bar(rot=0, yerr= [tcpError, udpError])
    plot.set_xlabel("Failure Mode")
    plot.set_ylabel("Latency")
    plot.set_title(label + " TCP VS UDP: Failure Mode Latency")


if __name__ == "__main__":
    if  (len(sys.argv) != 2 and len(sys.argv) != 3):
        print("usage: python3 plot.py <data csv>")
        print("usage: python3 plot.py <TCP csv> <UDP csv>")
    else:
        left_data = 0
        with open(sys.argv[1] , mode='r') as input_file:
            left_data = ingest(input_file)
        plotTopNLatency(left_data, 'TCP ' + ALGORITHM)

        if (len(sys.argv) == 3):
            right_data = 0
            with open(sys.argv[2] , mode='r') as input_file:
                right_data = ingest(input_file)
            plotTopNLatency(left_data, 'UCP ' + ALGORITHM)

            # plotReplicaComparison(left_data, right_data, ALGORITHM)
            plotClientComparison(left_data, right_data, ALGORITHM)
            plotDropComparison(left_data, right_data, ALGORITHM)
            plotFailComparison(left_data, right_data, ALGORITHM)
        
        plt.show()
