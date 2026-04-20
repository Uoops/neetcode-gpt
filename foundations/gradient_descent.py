class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for _ in range(iterations):
            grad = x * 2
            x -= learning_rate * grad
        
        return round(x, 5)
    