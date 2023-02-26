import itertools
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from itertools import groupby, combinations
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from itertools import groupby
import pandas as pd


class AmtDataReader:
    def __init__(
        self,
        file_path="cos_eor/scripts/orm/amt_data/data.npy",
        votes_threshold=5,
        K=5,
        top_k_by_votes=True,
    ):
        data = np.load(file_path, allow_pickle=True).item()
        # ranks: a 3d numpy array: [#room-receps, #objects, #annotations]
        self.ranks = data["ranks"]
        self.objects = data["objects"]
        # room_receptacles: a list of room-receptacle pairs with the room name and receptacle name concatenated with '|'
        self.room_receptacles = data["room_receptacles"]
        self.receptacles = list({rr.split("|")[1] for rr in self.room_receptacles})

        self.obj_to_idx = {obj: idx for idx, obj in enumerate(self.objects)}
        self.room_recep_to_idx = {
            room_rec: idx for idx, room_rec in enumerate(self.room_receptacles)
        }
        self.get_ranks_per_room()
        self.orm_mappings = self.gen_gt_orm_mappings(votes_threshold, K, top_k_by_votes)

    def get_ranks_for_room_recep(self, obj, room, recep):
        return self.ranks[
            self.room_recep_to_idx[room + "|" + recep], self.obj_to_idx[obj], :
        ]

    def get_object_list(self):
        return self.objects

    def get_room_list(self):
        return self.rooms

    def get_room_recep_list(self):
        return self.room_receptacles

    def get_all_orm_mappings(self):
        return self.orm_mappings

    def get_ranks_per_room(self):
        # Assumes receptacles from same rooms are together in self.room_receptacles
        rooms = it.groupby(
            range(len(self.room_receptacles)),
            key=lambda x: self.room_receptacles[x].split("|")[0],
        )
        rooms_dict = {k: list(g) for k, g in rooms}
        self.ranks_per_room = {k: self.ranks[list(rooms_dict[k])] for k in rooms_dict}
        self.receps_per_room = {
            k: [self.room_receptacles[r_idx].split("|")[1] for r_idx in rooms_dict[k]]
            for k in rooms_dict
        }
        self.rooms = self.ranks_per_room.keys()

    def get_ranks_for_obj_room(self, object, room):
        return self.ranks_per_room[room][:, self.obj_to_idx[object]]

    def get_recep_with_k_votes(self, object, room, category="after", k=5):
        ranks = self.get_ranks_for_obj_room(object, room)
        if category == "after":
            votes = ranks > 0
        elif category == "before":
            votes = ranks < 0
        else:
            raise Exception('category should be one of ["before", "after"]')

        votes_per_recep = np.sum(votes, -1)
        return (votes_per_recep >= k).astype(np.int)

    # Restrict to top-K matchings
    # Valid matchings should receive at least min_votes
    # If top_k_by_votes is true, preference is given to receptacles that receive highest number of votes
    # otherwise, preference is given to receptacles based on average ranks
    def gen_gt_orm_mappings(
        self, min_votes, K, top_k_by_votes=False, category="after", return_votes=False
    ):
        mappings = dict()
        for o in range(len(self.objects)):
            obj = self.objects[o]
            mappings[obj] = dict()
            for room in self.ranks_per_room.keys():
                ranks = self.ranks_per_room[room][:, o]
                receps = self.receps_per_room[room]
                if category == "after":
                    recep_ranks = [
                        (receps[j], ranks[j][ranks[j] > 0]) for j in range(len(receps))
                    ]
                elif category == "before":
                    recep_ranks = [
                        (receps[j], ranks[j][ranks[j] < 0]) for j in range(len(receps))
                    ]
                else:
                    raise NotImplementedError
                agg_ranks = sorted(
                    [
                        (recep, rank.shape[0], np.mean(rank) if len(rank) > 0 else 100)
                        for recep, rank in recep_ranks
                    ],
                    key=lambda x: x[1],
                )
                ranks_by_votes = groupby(agg_ranks, lambda x: x[1])
                ranks_by_votes = {
                    votes: [(x[0], x[2]) for x in value]
                    for votes, value in ranks_by_votes
                }
                # filter out the receptacles with less than min_votes
                filtered_receps = {
                    k: v for k, v in ranks_by_votes.items() if k >= min_votes
                }

                # sort rankings based on one of following two criteria:
                # i. number of votes, receptacles with same number votes will be order by avg ranking
                # ii. order by ranking after thresholding on number of votes

                if top_k_by_votes == True:
                    sorted_receps = sorted(
                        [
                            (k, sorted(v, key=lambda x: x[1]))
                            for k, v in filtered_receps.items()
                        ],
                        reverse=True,
                    )
                    # flatten the array
                    sorted_receps = [r for k, v in sorted_receps for r in v]
                else:
                    # flatten
                    flattened = [r for k, v in filtered_receps.items() for r in v]
                    # sort by avg rank
                    sorted_receps = sorted(flattened, key=lambda x: x[1])
                top_k = [recep[0] for recep in sorted_receps[:K]]
                mappings[obj][room] = top_k
        return mappings

    def gen_gt_obj_room_mappings(
        self,
        rooms_criteria=1,
        min_votes=6,
        K=10000,
        top_k_by_votes=False,
        category="after",
        remove_no_matches=False,
    ):
        mappings = self.gen_gt_orm_mappings(
            min_votes=min_votes, K=K, top_k_by_votes=top_k_by_votes, category=category
        )
        if rooms_criteria == 1:
            # best room is the one that has the most matching receptacles
            return {
                o: [
                    max(
                        [(r, len(mappings[o][r])) for r in mappings[o]],
                        key=lambda x: x[1],
                    )[0]
                ]
                for o in mappings
            }
        elif rooms_criteria == 2:
            # best room is the one with highest % of matching receptacles
            return {
                o: [
                    max(
                        [
                            (r, len(mappings[o][r]) / len(self.receps_per_room[r]))
                            for r in mappings[o]
                        ],
                        key=lambda x: x[1],
                    )[0]
                ]
                for o in mappings
            }
        elif rooms_criteria == 3:
            # all rooms
            room_mappings = {
                o: list(set([r for r in mappings[o] if len(mappings[o][r]) > 0]))
                for o in mappings
            }
            if remove_no_matches:
                return {o: r for o, r in room_mappings.items() if len(r) > 0}
            else:
                return room_mappings
        else:
            NotImplementedError
