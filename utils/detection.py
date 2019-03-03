class Detection(object):
    def __init__(self, id, label, xtl, ytl, width, height):
        self.id = id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.width = width
        self.height = height

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n id={0}, label={1}, TopLeftXY=({2},{3}), width={4}, height={5}'.format(self.id, self.label, self.xtl, self.ytl, self.width, self.height)
            