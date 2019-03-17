class Track(object):
    def __init__(self, id, detections, time_since_update, hits, hit_streak):
        self.id = id
        self.detections = detections
        self.time_since_update = time_since_update
        self.hits = hits
        self.hit_streak = hit_streak

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n Id={0}, Detections={1}, Time since update={2}, Hits={3}, Hits streak={4}'.format(self.id, self.detections, self.time_since_update, self.hits, self.hit_streak)
