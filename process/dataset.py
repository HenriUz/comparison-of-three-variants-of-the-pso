import os

class Problem():
    """
    Reads and processes the data contained in a dataset file.

    Attributes:
        o (int): Total number of orders in the dataset.
        i (int): Total number of items.
        a (int): Total number of available aisles.
        orders (List[Dict[int, int]]): List containing dictionaries that represent the orders. Each dictionary lists an item along with its quantity.
        aisles (List[Dict[int, int]]): List containing dictionaries that represent the aisles. Each dictionary lists an item along with its quantity in the aisle.
        lb (int): Lower bound.
        ub (int): Upper bound.
        result (Dict[str, Any]): Dictionary that stores the final results, including the dataset name, lists of selected orders and aisles, the objective function value, and the method's execution time.
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

                self.aisles = []
                for i in range(self.a):
                    aisle_line = lines[i + 1 + self.o].strip().split()
                    aisle = {int(aisle_line[1 + k * 2]): int(aisle_line[2 + k * 2]) for k in range(int(aisle_line[0]))}
                    self.aisles.append(aisle)
                
                last_line = lines[self.o + self.a + 1].strip().split()
                self.lb, self.ub = int(last_line[0]), int(last_line[1])

                self.result = {"dataset": dataset, "orders": [], "aisles": [], "objective": 0, "time": 0}
        except FileNotFoundError:
            print("Dataset doesn't exist.")
            exit()
    
    def print_dataset(self) -> None:
        print(f"o = {self.o}, i = {self.i}, a = {self.a}")

        print(f"\nOrders: ")
        for i in range(len(self.orders)):
            print(f"{i}: {self.orders[i]}")
        
        print(f"\nAisles: ")
        for i in range(len(self.aisles)):
            print(f"{i}: {self.aisles[i]}")
        
        print(f"\nLower Bound: {self.lb}, upper bound = {self.ub}")