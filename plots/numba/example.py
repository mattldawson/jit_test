class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print_position(self):
        return "X: {0} Y: {1}\n".format(self.x, self.y)