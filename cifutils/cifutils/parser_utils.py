

class PeriodicTable:
    def __init__(self, filename="cifutils/data/elements.txt"):
        self.symbol_to_number = {}
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    atomic_number = int(parts[0])
                    symbol = parts[1].upper()  # Convert symbol to uppercase
                    self.symbol_to_number[symbol] = atomic_number

    def get_atomic_number(self, symbol):
        # Convert the input symbol to uppercase to ensure case-insensitivity
        symbol_upper = symbol.upper()
        return self.symbol_to_number.get(symbol_upper, symbol)
