class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        def fnc(x):
            return x**2
        
        def der(x):
            return 2*x
        

        x = init
        for _ in range(iterations):
            x = x - learning_rate * der(x)
        
        return round(x, ndigits=5)

        pass
