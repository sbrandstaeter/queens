


class Tracer():
    def __int__(self):
        self.evaluate_funcs = []
        self.grad_funcs = []

    def add(self, evaluate_func, grad_func):
        self.evaluate_funcs.append(evaluate_func)
        self.grad_funcs.append(grad_func)

    def evaluate(self, arg):
        pass

    def evaluate_and_grad(self, arg):
        pass





