""" Executes a series of latency benchmarks against TCPCLient. """

import subprocess
import time
import csv
import sys

NUM_CLIENTS = [1, 2, 3] # Number of clients to benchmark with.
NUM_SERVERS = [3, 5] # Number of servers each test cluster should have.
DROP_RATES = [0, 1, 5] # Out of 1000, the server drop rates to test.
NUM_BENCH_OPS = [100, 500, 1000, 2000] # Number of ops to test per benchmark.
NUM_CLIENT_OPS = 10000 # Number of background client ops.
NUM_ITERATIONS = 5

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
            (CLIENT_EXE, str(num_servers), str(failure_code), str(drop_rate), joined_ops),
            stdout=subprocess.PIPE))


def run_benchmark_client(num_servers, drop_rate, failure_code, num_ops):
    """ Runs a new benchmark client as a subprocess. """
    joined_ops = ','.join([CMD] * num_ops)
    subprocess.call(
        [CLIENT_EXE, str(num_servers), str(drop_rate), str(failure_code), joined_ops],
        stdout=subprocess.PIPE)


def run_benchmark(num_clients, num_servers, drop_rate, num_ops, random_failures, writer):
    """ Runs a no failure benchmark with the specified parameters. """
    # Encode if random server failures should be allowed.
    failure_code = RANDOM_FAILURE_CODE if random_failures else NO_FAILURE_CODE

    # Setup background clients, if any.
    num_background_clients = num_clients - 1
    run_background_clients(num_background_clients, num_servers, drop_rate, failure_code)

    # Time benchmarking client.
    start = time.time()
    run_benchmark_client(num_servers, drop_rate, failure_code, num_ops)
    end = time.time()

    avg_latency = round((end - start) * 1000 / num_ops)
    print("{} clients/{} servers/{} drop rate/{} random failures/latency: {} ms".format(
        num_clients, num_servers, drop_rate, failure_code, avg_latency))
    writer.writerow([num_clients, num_servers, drop_rate, failure_code, num_ops, avg_latency])


    # Kill background clients, if any.
    for proc in CLIENT_PROCS:
        if proc.poll() is None:
            proc.kill()


def run_benchmark_suite(writer):
    """ Runs all the benchmarks for the suite.. """
    for num_iterations in range(NUM_ITERATIONS):
        for num_clients in NUM_CLIENTS:
            for num_servers in NUM_SERVERS:
                for drop_rate in DROP_RATES:
                    for num_ops in NUM_BENCH_OPS:
                        run_benchmark(num_clients, num_servers, drop_rate, num_ops, False, writer)
                        run_benchmark(num_clients, num_servers, drop_rate, num_ops, True, writer)


if __name__ == "__main__":
    if  len(sys.argv) != 2:
        print("usage: python3 benchmarks.py <client exe>")
    else:
        CLIENT_EXE = sys.argv[1]
        with open('results.csv', mode='w') as output_file:
            run_benchmark_suite(
                csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
