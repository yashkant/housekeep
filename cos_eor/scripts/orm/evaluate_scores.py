import numpy as np
import yaml
from amt_data.data_reader import AmtDataReader
from sklearn.metrics import average_precision_score as ap_score
import matplotlib.pyplot as plt
import os


class ScoreEvaluator:
    def __init__(
        self,
        scores_file="cos_eor/scripts/orm/mlm_scores.npy",
        splits_file="cos_eor/scripts/orm/amt_data/splits.yaml",
        matching="obj_room_recep",
        random=False,
        seed=0,
    ) -> None:
        self.splits_file = splits_file
        self.scores_file = scores_file
        self.get_scores(matching=matching)
        if random:
            np.random.seed(seed)
            self.scores = self.gen_random_ranks(self.scores)

        self.amt = AmtDataReader()

    def get_objects_list(self, split="trainval"):
        all_objects = yaml.full_load(open(self.splits_file))
        objects_list = all_objects["objects"][split]
        if "guitar" in objects_list:
            objects_list.remove("guitar")
        return objects_list

    def read_scores_file(self):
        scores_dict = np.load(self.scores_file, allow_pickle=True).item()
        return (
            scores_dict["scores"],
            scores_dict["objects"],
            scores_dict["rooms"],
            scores_dict["receptacles"],
        )

    def get_scores(self, matching="obj_room_recep"):
        if matching == "obj_room_recep":
            (
                self.scores,
                self.objects_in_file,
                self.rooms,
                self.receptacles,
            ) = self.read_scores_file()
            self.rooms = [room.replace(" ", "_") for room in self.rooms]
            self.receptacles = [rec.replace(" ", "_") for rec in self.receptacles]
            room_indices = np.argsort(self.rooms)
            self.rooms = np.sort(self.rooms).tolist()
            self.scores = np.take(self.scores, room_indices, axis=1)

            self.obj_to_idx = {
                obj.replace(" ", "_"): idx
                for idx, obj in enumerate(self.objects_in_file)
            }
            self.recep_to_idx = {
                recep.replace(" ", "_"): idx
                for idx, recep in enumerate(self.receptacles)
            }
            self.room_to_idx = {room: idx for idx, room in enumerate(self.rooms)}
            return self.scores
        else:
            scores_dict = np.load(self.scores_file, allow_pickle=True).item()
            if "object_room" in scores_dict:
                scores_dict = scores_dict["object_room"]
            self.scores, self.objects_in_file, self.rooms = (
                scores_dict["scores"],
                scores_dict["objects"],
                scores_dict["rooms"],
            )
            self.rooms = [
                room.lstrip().strip().replace(" ", "_") for room in self.rooms
            ]
            self.objects_in_file = [
                obj.replace(" ", "_") for obj in self.objects_in_file
            ]
            room_indices = np.argsort(self.rooms)
            self.rooms = np.sort(self.rooms).tolist()
            self.scores = np.take(self.scores, room_indices, axis=1)
            self.room_to_idx = {room: idx for idx, room in enumerate(self.rooms)}
            self.obj_to_idx = {
                obj.replace(" ", "_"): idx
                for idx, obj in enumerate(self.objects_in_file)
            }

    def get_scores_for_obj_room(self, object, room, recep_list):
        recep_indices = [self.recep_to_idx[recep] for recep in recep_list]
        return self.scores[
            self.obj_to_idx[object], self.room_to_idx[room], recep_indices
        ]

    def get_scores_for_obj(self, object):
        return self.scores[self.obj_to_idx[object]]

    def mAP(self, gt, preds):
        # scipy function returns nan for this case
        if np.sum(gt) == 0:
            return -1
        return ap_score(gt, preds)

    def evaluate_metrics(self, object, room, drop_receptacles=[], return_all=False):
        recep_list = self.amt.receps_per_room[room]
        filtered_recep_list = [
            recep_list[i]
            for i in range(len(recep_list))
            if recep_list[i] not in drop_receptacles
        ]
        predictions = self.get_scores_for_obj_room(object, room, filtered_recep_list)
        gt = self.amt.get_recep_with_k_votes(object, room, k=6)
        gt = [gt[i] for i in range(len(gt)) if recep_list[i] not in drop_receptacles]
        mAP = self.mAP(gt, predictions)
        if return_all:
            return predictions, gt, recep_list, mAP
        return mAP

    def evaluate(self, split="trainval", drop_receptacles=[]):
        objects = self.get_objects_list(split)
        mAP = np.array(
            [
                [
                    self.evaluate_metrics(o, r, drop_receptacles=drop_receptacles)
                    for o in objects
                ]
                for r in self.rooms
                if r in self.amt.receps_per_room
            ]
        )
        return np.mean(mAP[mAP > -1])

    def evaluate_obj_room_scores_per_obj(
        self, obj, obj_room_mappings, return_all=False
    ):
        room_list = self.rooms
        predictions = self.get_scores_for_obj(obj)
        gt = obj_room_mappings[obj]
        gt = [1 if r in gt else 0 for r in room_list]
        mAP = self.mAP(gt, predictions)
        if return_all:
            return predictions, gt, room_list, mAP
        return mAP

    def evaluate_obj_room_scores(self, split="trainval"):
        mappings = self.amt.gen_gt_obj_room_mappings(rooms_criteria=3, min_votes=6, K=1)
        objects = self.get_objects_list(split)
        mAP = np.array(
            [self.evaluate_obj_room_scores_per_obj(o, mappings) for o in objects]
        )
        return np.mean(mAP[mAP > -1])

    def gen_random_ranks(self, sample_matrix):
        n_items = sample_matrix.shape[-1]
        n_rankings = int(np.prod(sample_matrix.shape) / n_items)
        return np.array(
            [np.random.permutation(n_items) for _ in range(n_rankings)]
        ).reshape(sample_matrix.shape)


base_dir = "cos_eor/scripts/orm/all_scores"

for name, dirname in [
    ("RoBERTa+CM", "roberta-cm"),
    ("GloVE+CM", "glove-cm"),
    ("ZS-MLM", "zs-mlm"),
]:
    print(f"Evaluating {name} scores")
    eval = ScoreEvaluator(
        scores_file=os.path.join(base_dir, dirname, "orr_scores.npy"),
        matching="obj_room_recep",
        splits_file="cos_eor/scripts/orm/amt_data/splits.yaml",
    )
    for split in ["train", "val", "test"]:
        print(f"ORR ({split}): ", eval.evaluate(split=split))
    eval = ScoreEvaluator(
        scores_file=os.path.join(base_dir, dirname, "or_scores.npy"),
        matching="obj_room",
        splits_file="cos_eor/scripts/orm/amt_data/splits.yaml",
    )
    for split in ["train", "val", "test"]:
        print(f"OR ({split}): ", eval.evaluate_obj_room_scores(split=split))
    print()
