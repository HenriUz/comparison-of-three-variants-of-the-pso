import sys

from methods.mbpso import MBPSO, MBPSOzt
from methods.sbpso import SBPSO
from process.dataset import Problem
from random import seed

def main(dataset: str, result_file: str, method: int, seed_number: int) -> Problem:
    seed(seed_number)

    problem = Problem(dataset)

    if method == 0:
        MBPSO(problem, 50, 600, 0.9, 0.4, 2, 2, -6, 6, 0.02)
    elif method == 1:
        MBPSOzt(problem, 50, 600, 0.9, 0.4, 2, 2, -6, 6, 0.02, 0.5)
    elif method == 2:
        SBPSO(problem, 50, 600, 0.9297, 0.2266, 1.3086, 2.1526, 7)
    else:
        print("Invalid method.")
        exit()
    
    print(f"\n\nFinished: {problem.result["objective"]} in time {problem.result["time"]}.")
    
    problem.save_solution(result_file)
    return problem

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Incorrect number of arguments.")
        print("Use: python main.py <dataset> <result_file> <method_number> <seed_number>")
        print("Methods:\n\tMBPSO: 0\n\tMBPSO_zt: 1\n\tSBPSO: 2")
    else:
        try:
            method = int(sys.argv[3])
            seed_number = int(sys.argv[4])
        except (IndexError, ValueError):
            print("Error: method and seed must be numeric integers.")
            exit()
        
        main(sys.argv[1], sys.argv[2], method, seed_number)