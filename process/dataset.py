import os

class Problem():
    """
    Reads and processes the data contained in a dataset file.

    Attributes:
        o (int): Total number of orders in the dataset.
        i (int): Total number of items.
        a (int): Total number of available aisles.
        orders (list[dict[int, int]]): List containing dictionaries that represent the orders. Each dictionary lists an item along with its quantity.
        sorted_orders (list[tuple[int, int]]): List sorted in descending order by the number of items in each order. Each element is a tuple in the format (index, number of items).
        aisles (list[dict[int, int]]): List containing dictionaries that represent the aisles. Each dictionary lists an item along with its quantity in the aisle.
        lb (int): Lower bound.
        ub (int): Upper bound.
        result (dict[str, any]): Dictionary that stores the final results, including the dataset name, lists of selected orders and aisles, the objective function value, and the method's execution time.
    """
    
    def __init__(self, dataset: str) -> None:
        """
        Args:
            dataset (str): Name of the dataset to be processed (there should be a file named "<dataset>.txt" in the "datasets" directory).
        """

        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, "datasets", f"{dataset}.txt")
            
            with open(dataset_path, "rb") as data:
                lines = data.readlines()
        
                first_line = lines[0].strip().split()
                self.o, self.i, self.a = int(first_line[0]), int(first_line[1]), int(first_line[2])

                self.orders = []
                for i in range(self.o):
                    order_line = lines[i + 1].strip().split()
                    order = {int(order_line[1 + k * 2]): int(order_line[2 + k * 2]) for k in range(int(order_line[0]))}
                    self.orders.append(order)

                self.sorted_orders = [(x, sum(d.values())) for x, d in enumerate(self.orders)]
                self.sorted_orders.sort(key = lambda i: i[1], reverse = True)

                self.aisles = []
                for i in range(self.a):
                    aisle_line = lines[i + 1 + self.o].strip().split()
                    aisle = {int(aisle_line[1 + k * 2]): int(aisle_line[2 + k * 2]) for k in range(int(aisle_line[0]))}
                    self.aisles.append(aisle)
                
                last_line = lines[self.o + self.a + 1].strip().split()
                self.lb, self.ub = int(last_line[0]), int(last_line[1])

                self.result = {"dataset": dataset, "orders": [], "aisles": [], "objective": 0, "time": 0}
        
        except FileNotFoundError:
            print(f"Dataset {dataset} not found in the datasets folder.")
            exit()

    def objective_function(self, number_items: int, number_aisles: int) -> float:
        """
        Calculates the value of the objective function and returns it. If the lower bound constraint is violated, or if `number_aisles` is 0, the return value will be 0. 

        Since the methods work directly with the aisles, which have no constraints, and use a greedy function that selects orders while respecting the upper bound and supply constraints (`add_orders`), this function does not check the previous constraints to avoid redundancy.

        Args:
            number_items (int): Number of items in the selected orders.
            number_aisles (int): Number of selected aisles.
        
        Returns:
            Objective (float): Value of the objective function.
        """
        
        # Checking capacity constraints and the number of aisles.
        if number_items < self.lb or number_aisles == 0:
            return 0.0

        return number_items / number_aisles

    def save_solution(self, file: str) -> None:
        """
        Creates or overwrites a file with the specified name in the results folder. This file contains the orders and aisles in the solution in the following format:
        - First line: an integer `o` representing the number of orders in the solution;
        - Next `o` lines: one integer per line, representing the order indices;
        - Next line: an integer `a` representing the number of aisles in the solution;
        - Next `a` lines: one integer per line, representing the aisles indices. 
        
        Args:
            file (str): Name of the file where the solution will be saved (do not include ".txt").
        """
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_path = os.path.join(base_dir, "results", f"{file}.txt")

        with open(result_path, "+w") as file:
            file.write(str(len(self.result["orders"])) + "\n")
            for o in self.result["orders"]:
                file.write(str(o) + "\n")
            file.write(str(len(self.result["aisles"])) + "\n")
            for a in self.result["aisles"]:
                file.write(str(a) + "\n")
            file.close()