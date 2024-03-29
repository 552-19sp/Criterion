""" Executes a series of latency benchmarks against TCPCLient. """

import subprocess
import time
import csv
import sys

NUM_CLIENTS = [1, 2, 3] # Number of clients to benchmark with.
NUM_SERVERS = [3, 5] # Number of servers each test cluster should have.
DROP_RATES = [0, 1, 5] # Out of 1000, the server drop rates to test.
NUM_BENCH_OPS = 500 # Number of ops to test per benchmark.
NUM_CLIENT_OPS = 10000 # Number of background client ops.

CMD = "GET A"
CLIENT_PROCS = []

NO_FAILURE_CODE = 0
RANDOM_FAILURE_CODE = 1

NO_DROP_RATE = 0

CLIENT_EXE = ""


def run_background_clients(num_clients, num_servers, drop_rate, failure_code):
    """ Runs extra clients, does not wait for them to exit. """
    for _ in range(num_clients):
        joined_ops = ','.join([CMD] * NUM_CLIENT_OPS)
        CLIENT_PROCS.append(subprocess.Popen(
            (CLIENT_EXE, str(num_servers), str(drop_rate), str(failure_code), joined_ops),
            stdout=subprocess.PIPE))


def run_benchmark_client(num_servers, drop_rate, failure_code):
    """ Runs a new benchmark client as a subprocess. """
    joined_ops = ','.join([CMD] * NUM_BENCH_OPS)
    subprocess.call(
        [CLIENT_EXE, str(num_servers), str(drop_rate), str(failure_code), joined_ops],
        stdout=subprocess.PIPE)


def run_benchmark(num_clients, num_servers, drop_rate, random_failures, writer):
    """ Runs a no failure benchmark with the specified parameters. """
    # Encode if random server failures should be allowed.
    failure_code = RANDOM_FAILURE_CODE if random_failures else NO_FAILURE_CODE

    # Setup background clients, if any.
    num_background_clients = num_clients - 1
    run_background_clients(num_background_clients, num_servers, drop_rate, failure_code)

    # Time benchmarking client.
    start = time.time()
    run_benchmark_client(num_servers, drop_rate, failure_code)
    end = time.time()

    # Kill background clients, if any.
    for proc in CLIENT_PROCS:
        if proc.poll() is None:
            proc.kill()

    avg_latency = round((end - start) * 1000 / NUM_BENCH_OPS)
    print("{} clients/{} servers/{} drop rate/{} random failures/latency: {} ms".format(
        num_clients, num_servers, drop_rate, failure_code, avg_latency))
    writer.writerow([num_clients, num_servers, drop_rate, failure_code, avg_latency])


def run_benchmark_suite(writer):
    """ Runs all the benchmarks for the suite.. """
    for num_clients in NUM_CLIENTS:
        for num_servers in NUM_SERVERS:
            for drop_rate in DROP_RATES:
                run_benchmark(num_clients, num_servers, drop_rate, False, writer)
                run_benchmark(num_clients, num_servers, drop_rate, True, writer)


if __name__ == "__main__":
    if  len(sys.argv) != 2:
        print("usage: python3 benchmarks.py <client exe>")
    else:
        CLIENT_EXE = sys.argv[1]
        with open('results.csv', mode='w') as output_file:
            run_benchmark_suite(
                csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
