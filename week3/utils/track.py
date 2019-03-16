class Track(object):
    def __init__(self, id, bboxes, time_since_update, hits, hit_streak):
        self.id = id
        self.bboxes = bboxes
        self.time_since_update = time_since_update
        self.hits = hits
        self.hit_streak = hit_streak

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n id={0}, bboxes={1}, Time since update={2}, Hits={3}, Hits streak={4}'.format(self.id, self.bboxes, self.time_since_update, self.hits, self.hit_streak)
