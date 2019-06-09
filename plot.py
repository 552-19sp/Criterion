""" 
Takes one argument: a csv of the form
        num_clients,num_servers,drop_rate,failure_code,avg_latency
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
        'num_clients', 'num_servers', 'drop_rate', 'failure_code', 'avg_latency'])

    meta = "SELECT num_clients, num_servers, drop_rate, failure_code, \
            MIN(avg_latency) AS min, MAX(avg_latency) AS max, AVG(avg_latency) AS avg \
            FROM df \
            GROUP BY num_clients, num_servers, drop_rate, failure_code \
            ORDER BY AVG(avg_latency)"
    latency = ps.sqldf(meta, locals())
    
    # Return a series with the topN average latencies
    topN = latency.iloc[:PLOTS_PER_GRAPH, :]

    print(topN)


if __name__ == "__main__":
    if  len(sys.argv) != 2:
        print("usage: python3 plot.py <data csv>")
    else:
        CSV = sys.argv[1]
        with open('results.csv', mode='r') as input_file:
            ingest(input_file)
