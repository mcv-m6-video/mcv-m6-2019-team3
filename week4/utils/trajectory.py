class Trajectory(object):
    def __init__(self, x, y, a):
        self.x = x
        self.y = y
        self.a = a # angle


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n x={0}, y={1}, angle={2}'.format(self.x, self.y, self.a)

    def __getitem__(self, key):
        return self.x[key], self.y[key], self.a[key]
