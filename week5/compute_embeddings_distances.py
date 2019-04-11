import os
import pickle
from collections import defaultdict
from scipy.spatial import distance

if __name__ == '__main__':
    for i in range(9):
        print('id:{}'.format(i))
        if os.path.exists('embeddings' + str(i) + '.pkl'):
            with open('embeddings' + str(i) + '.pkl', 'rb') as p:
                print("Reading tracked detections from pkl")
                embeddings = pickle.load(p)
                print("Tracked detections loaded\n")

            tracks_embeddings = defaultdict(list)
            tracks_average_embeddings = []
            for camera in embeddings:
                embed_camera = embeddings[camera]
                for detec_embed in embed_camera:
                    for detec in detec_embed:
                        tracks_embeddings[detec.track_id].append(detec_embed[detec])
                for trackid in tracks_embeddings:
                    tracks_average_embeddings.append([sum(col) / len(col) for col in zip(*tracks_embeddings[trackid])])

            ref_track = tracks_average_embeddings[0]
            print('Distances:')
            for track_emb in tracks_average_embeddings[1:]:
                print(distance.euclidean(ref_track, track_emb))







