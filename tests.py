import ast
import csv
import main
import statistics
import sys
import utils.checker

from itertools import combinations
from scipy.stats import wilcoxon
from time import perf_counter

datasets = {
    "instance_0001": 15.0,
    "instance_0002": 2.0,
    "instance_0003": 12.0,
    "instance_0004": 3.5,
    "instance_0005": 177.88,
    "instance_0006": 691.0,
    "instance_0007": 392.25,
    "instance_0008": 162.94,
    "instance_0009": 4.42,
    "instance_0010": 17.11,
    "instance_0011": 16.85,
    "instance_0012": 11.25,
    "instance_0013": 117.38,
    "instance_0014": 181.64,
    "instance_0015": 149.33,
    "instance_0016": 85.0,
    "instance_0017": 36.5,
    "instance_0018": 117.2,
    "instance_0019": 202.0,
    "instance_0020": 5.0
}

methods = {"MBPSO": 0, "MBPSOzt": 1, "SBPSO": 2}

seeds = [
    10447,
    22022,
    24675,
    35446,
    35476,
    37983,
    39628,
    40694,
    41383,
    42738,
    45786,
    46223,
    48679,
    56429,
    56927,
    58565,
    60163,
    61820,
    61975,
    65439,
    68036,
    68071,
    72427,
    73641,
    78884,
    82757,
    85722,
    87599,
    94860,
    95990
]

def load_csv(file_path):
    data = {}

    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue

            if row[0] == "dataset":
                continue

            id_ = row[0]
            min_value = float(row[1])
            max_value = float(row[2])
            mean = float(row[3])
            stdev = float(row[4])
            gap = float(row[5])
            time = float(row[6])
            data[id_] = (min_value, max_value, mean, gap, stdev, time)
    
    return data

def thirty_executions(method: str) -> None:
    results = {
        method: {dataset: {"objectives": [], "times": []} for dataset in datasets}
    }

    print(f"\nMethod: {method}")
    for dataset, best_obj in datasets.items():
        print(f"Dataset: {dataset}")
        
        start = perf_counter()
        
        for i in seeds:
            solution = main.main(dataset, "solution", methods[method], i)
            is_feasible, objective_value = utils.checker.main(f"datasets/{dataset}.txt", "results/solution.txt")

            if not is_feasible or solution.result["objective"] != objective_value:
                print(f"Something is wrong: {is_feasible}, {solution.result["objective"]}, {objective_value}")
                return
            
            results[method][dataset]["objectives"].append(solution.result["objective"])
            results[method][dataset]["times"].append(solution.result["time"])

        end = perf_counter()
        print(f"Time: {end - start}\n")

        min_obj = min(results[method][dataset]["objectives"])
        max_obj = max(results[method][dataset]["objectives"])
        mean_obj = statistics.mean(results[method][dataset]["objectives"])
        stdev_obj = statistics.stdev(results[method][dataset]["objectives"])

        mean_time = statistics.mean(results[method][dataset]["times"])

        gap = 100 * (best_obj - mean_obj) / best_obj

        with open(f"results/{method}.csv", "+a") as file:
            file.write(f"{dataset},{min_obj},{max_obj},{mean_obj},{stdev_obj},{gap},{mean_time}\n")
            file.close()

        with open(f"execution/data_{dataset}.txt", "+a") as file:
            file.write(str(results[method][dataset]["objectives"]) + f"{method}\n")
            file.close()

def load_results(file_path: str) -> dict[list[str]]:
    data = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue

            idx = line.rfind("]")
            values_str = line[:idx+1]
            method = line[idx+1:]

            values = ast.literal_eval(values_str)
            data[method] = values

    return data

def wilcoxon_test() -> None:
    results = {
        name: {dataset: {"objectives": [], "times": []} for dataset in datasets}
        for name, _ in methods.items()
    }

    for dataset, _ in datasets.items():
        print(f"\nDataset {dataset}")
        with open("results/wilcoxon.txt", "+a") as file:
            file.write(f"\nDataset: {dataset}\n")
            file.close()

        results = load_results(f"execution/data_{dataset}.txt")

        # `thirty_executions` saves the MBPSOzt without the `k` parameter indicator, so you need to manually change.
        for m1, m2 in combinations(["MBPSO", "MBPSOzt_05", "MBPSOzt_2", "SBPSO"], 2):
            A = results[m1]
            B = results[m2]

            _, p = wilcoxon(A, B)

            mean_A = statistics.mean(A)
            mean_B = statistics.mean(B)

            if p < 0.05:
                if mean_A > mean_B:
                    winner = m1
                else:
                    winner = m2
            else:
                winner = "tie"

            print(f"{m1:10} vs {m2:10} -> p={p:22}, winner={winner}")
            with open("results/wilcoxon.txt", "+a") as file:
                file.write(f"{m1:10} vs {m2:10} -> p={p:22}, ")

                if winner == "tie":
                    file.write(f"there was no statistically significant difference.\n")
                else:
                    file.write(f"{winner} was statistically superior.\n")
                    
                file.close()

if __name__ == "__main__":
    test = sys.argv[1]

    if test == "0":
        name = sys.argv[2]
        thirty_executions(name)
    elif test == "1":
        wilcoxon_test()
    elif test == "2":
        d1 = load_csv("results/mbpso.csv")
        d2 = load_csv("results/mbpsozt_05.csv")
        d3 = load_csv("results/sbpso.csv")

        for id_ in d1:
            _, max1, m1, g1, s1, t1 = d1[id_]
            _, max2, m2, g2, s2, t2 = d2[id_]
            _, max3, m3, g3, s3, t3 = d3[id_]

            line = (
                f"{id_} & \n"
                f"{m1:.2f} $\\pm$ {s1:.2f} & {g1:.2f} & {max1:.2f} & {t1:.2f} & \n"
                f"{m2:.2f} $\\pm$ {s2:.2f} & {g2:.2f} & {max2:.2f} & {t2:.2f} & \n"
                f"{m3:.2f} $\\pm$ {s3:.2f} & {g3:.2f} & {max3:.2f} & {t3:.2f} \\\\ \n\n"
            )

            print(line)