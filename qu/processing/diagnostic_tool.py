#   /********************************************************************************
#   * Copyright © 2020-2021, ETH Zurich, D-BSSE, Aaron Ponti
#   * All rights reserved. This program and the accompanying materials
#   * are made available under the terms of the Apache License Version 2.0
#   * which accompanies this distribution, and is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.txt
#   *
#   * Contributors:
#   *     Matteo Jucker Riva - contributor
#   *******************************************************************************/
#   NOTES:
#   this module allows to call a diagnostic tool to evaluate the quality of
#   a segmentation  using ground truth masks and prediction results


import os
from statistics import mean

# object based analysis of results
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import kstest
from sklearn.metrics import classification_report

# batch assessment


class SegmentationDiagnostics:
    """creates a segmentation diagnostics that analyses quality
    of (one or more) prediction batches compared to ground truth"""

    def __init__(self, gt_fold_path, pred_fold_path=None):

        # storage for ground truth attributes
        # "infos" include paths, file list and number of files
        # "metrics" contains:

        #  - cell count per image
        #  - cell size per image

        self.gt = {"infos": None,
                   'img': [],
                   "metrics": None}

        # prediction is a list of dictionary with len== # number of prediction batches to analyse\
        # each dictionary includes path, experiment_name, file_names, file_num
        self.pred_infos = []

        # storage for images
        self.pred_img = []
        # image level metrics for each batch: a list of dictionaries (one per batch) containing

        # - cell count per image
        # - cell size per image

        # - iou per image
        # - confusion matrix per image
        self.pred_metrics = []

        # batch level predictions: a list of dictionaries (one per prediction batch) containing:\
        # - cell count p_value
        # - cell size p_value
        # - Iou average
        # - f1 score total

        # dictionary of subplots for each metric (for all batches)
        self.pred_batch_metrics = []
        # plots
        self.metrics_plots = []
        # load gt infos
        self.gt["infos"] = self.load_batch_infos(gt_fold_path, "ground_truth")

        #  load predictions infos
        if pred_fold_path:
            if not isinstance(pred_fold_path, list):
                pred_fold_path = [pred_fold_path]
            self.pred_infos = [self.load_batch_infos(path) for path in pred_fold_path]

        # Figure
        plt.style.use("dark_background")
        self._fig, self._ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))

    def evaluate_segmentation(self, save_path=None):
        """wrapper to perform all segmentation evaluation and create
        plots"""

        for pred_num in range(len(self.pred_infos)):
            # calc image metrics

            # ----------------- Load images
            # check that gt files and pred files are the same if not filter pred files
            clean_gt_names = sorted([f_name.split("_")[-1] for f_name in self.gt['infos']["file_names"]])
            clean_pred_names = sorted([f_name.split("_")[-1] for f_name in self.pred_infos[pred_num]["file_names"]])

            if (self.gt['infos']["file_num"] != self.pred_infos[pred_num][
                "file_num"]) or clean_gt_names != clean_pred_names:
                selected_gt_files = [gt_file for gt_file in sorted(self.gt['infos']["file_names"]) \
                                     if gt_file.split("_")[-1] in self.pred_infos[pred_num]["file_names"]]

                selected_pred_files = [pr_file for pr_file in sorted(self.pred_infos[pred_num]['file_names']) \
                                       if pr_file.split("_")[-1] in self.gt['infos']["file_name"]]
            else:
                selected_gt_files = self.gt['infos']["file_names"]
                selected_pred_files = self.pred_infos[pred_num]["file_names"]

            gt_img_list = []
            pred_img_list = []
            for n, (gt_file, pred_file) in enumerate(zip(selected_gt_files, selected_pred_files)):

                # load gt image
                gt_img_path = os.path.join(self.gt['infos']['path'], gt_file)
                gt_img_list.insert(n, np.array(Image.open(gt_img_path)))

                # load predictions
                pred_img_path = os.path.join(self.pred_infos[pred_num]['path'], pred_file)
                pred_img_list.insert(n, np.array(Image.open(pred_img_path)))

            # store imgs
            self.gt['img'] = gt_img_list
            self.pred_img.insert(pred_num, pred_img_list)

            # calc basic and advanced image metrics
                # ground truth
            if not self.gt['metrics']:
                self.gt["metrics"] = self.calc_image_metrics(-1)
                # prediction
            self.pred_metrics.insert(pred_num, self.calc_image_metrics(pred_num))
                # advanced
            ious, img_f1 = self.calc_advanced_image_metrics(pred_num)
            self.pred_metrics[pred_num]['ious'] = ious
            self.pred_metrics[pred_num]['image_f1'] = img_f1

            self.calc_advanced_image_metrics(pred_num)

            self.pred_batch_metrics.insert(pred_num, self.calc_batch_metrics(pred_num))

            # plot
            self.plot_metrics(pred_num)

            # Save the figure only if a path was explicitly passed
            if not save_path:
                return "NOT_SAVED"

            output_name = self.pred_infos[pred_num]['exp_name'] + ".png"
            full_save_path = os.path.join(save_path, output_name)
            self._fig.savefig(full_save_path)
            return full_save_path

    def load_batch_infos(self, path, exp_name=None) -> dict:
        """pre_loads batch of images, setting name and number
        uses folder path"""
        new_info_dic = {"path": None,
                        "exp_name": None,
                        "file_names": [],
                        "file_num": None}

        if os.path.exists(path) and (len(os.listdir(path)) > 0):
            info_dic = new_info_dic.copy()
            info_dic["path"] = path
            info_dic["exp_name"] = exp_name if exp_name else os.path.basename(path)
            info_dic["file_names"] = sorted(os.listdir(path))
            info_dic["file_num"] = len(os.listdir(path))
            return info_dic
        else:
            assert f"FileNotFoundError: check the folder {path}"
            return None

    def calc_image_metrics(self, pred_num=0, verbose=False) -> dict:
        """Evaluates segmentation quality in batch of images (one prediction).
        pred_num correspond to the batch number: -1 indicates ground truth

        @returns dictionary with:
        - cell count per image
        - cell sizes per image"""
        metrics_dic = {"cell_count": [],
                       "cell_sizes": []}

        infos_dic = self.pred_infos[pred_num] if pred_num >= 0 else self.gt['infos']
        img_list = self.pred_img[pred_num] if pred_num >= 0 else self.gt['img']

        # check if batch infos are loaded
        if not infos_dic["path"]:
            assert f" FileNotFoundError, no information available. \
            Maybe execute load_batch_infos first? "

        # Scan through ech image

        for n, img in enumerate(img_list):
            if verbose >= 2:
                print(f"\n evaluating image # {n}")

            # Check if image is indeed segmentation
            if (len(np.unique(img)) > 10) or \
                    (img.dtype != int):
                assert "ValueError: image is not a segmentation mask"

            # get contours and dist_transform
            (cnts, _), dist_trans = self._get_contours(img)


            # count distinct objects / cells
            metrics_dic['cell_count'].insert(n, len(cnts))
            if verbose >= 2:
                print(f"number of unique objects: {self._count_cells(cnts)} ")

            # get size distribution
            metrics_dic["cell_sizes"] = metrics_dic["cell_sizes"] + self._get_cell_size(img, cnts, dist_trans)
            if verbose >= 2:
                print(f"cell sizes : {metrics_dic['cell_sizes']}")
        if verbose >= 2:
            print(f"final dictionary {metrics_dic.items()}")
        return metrics_dic

    def calc_advanced_image_metrics(self, pred_num, verbose=False):
        """calculates advanced image level metrics that need comparison between ground truth
        and prediction batches """

        # get imgs + infos + metrics for current prediction and gt
        gt_img_list = self.gt['img']
        pred_img_list = self.pred_img[pred_num]

        # Scan through ech image
        ious = []
        image_f1 = []
        for n, (gt_img, pred_img) in enumerate(zip(gt_img_list, pred_img_list)):
            # calc_int_over_union
            ious.insert(n, self._int_over_union(gt_img, pred_img))
            # calc f1 score per image
            image_f1.insert(n, self._calc_f1_score(gt_img.flatten(), pred_img.flatten()))
        return ious, image_f1

    def calc_batch_metrics(self, pred_num=0):
        """batch level metrics comparing ground truth and prediction"""

        batch_metrics = {"cell_count_RMSE": [],
                         "ks_p": [],
                         "cell_size_p": [],
                         "f1_score": []}

        gt_metrics = self.gt["metrics"]
        pred_metrics = self.pred_metrics[pred_num]
        # prediction image level metrics

        # assess difference in cell count RMSE
        gt_count = gt_metrics['cell_count']
        pr_count = pred_metrics["cell_count"]
        batch_metrics["cell_count_RMSE"] = round(np.sqrt(((np.array(gt_count) - np.array(pr_count))**2).mean()),3)

        # assess difference in cell sizes
        gt_sizes = gt_metrics['cell_sizes']
        pr_sizes = pred_metrics['cell_sizes']
        p_score = kstest(gt_sizes, pr_sizes )[1]
        batch_metrics["ks_p"] = round(p_score,3)

        # calc average iou
        batch_metrics["iou_mean"] = round(mean(pred_metrics['ious']),3)

        # calc f value
        image_f1 = pred_metrics['image_f1']
        if not isinstance(image_f1, list):
            image_f1 = [image_f1]
        f1_score_mean = mean(image_f1)

        batch_metrics["f1_score"] = round(f1_score_mean, 3)

        # store dictionary as attribute
        return batch_metrics

    def plot_metrics(self, pred_num=0, verbose=None):

        # get infos and metrics
        pred_infos = self.pred_infos[pred_num]
        gt_metrics = self.gt['metrics']
        pred_metrics = self.pred_metrics[pred_num]
        pred_batch_metrics = self.pred_batch_metrics[pred_num]

        # plot 1 - cell count
        # get data
        gt_cell_count = gt_metrics['cell_count']
        pred_cell_count = pred_metrics['cell_count']
        max_ax_range = np.quantile(np.array(pred_cell_count).flatten(), 0.95)
        # create plot
        self._ax[0, 0].scatter(np.array(gt_cell_count), np.array(pred_cell_count) )
        self._ax[0, 0].set_xlim([0, max_ax_range])
        self._ax[0, 0].set_ylim([0, max_ax_range])
        self._ax[0, 0].set_title(f"Number of cells per image - RMSE: {pred_batch_metrics['cell_count_RMSE']}")
        self._ax[0, 0].set_xlabel("ground truth")
        self._ax[0, 0].set_ylabel("prediction")

        # plot 2 - cell size
        # get data
        gt_sizes = gt_metrics["cell_sizes"]
        pred_sizes = pred_metrics["cell_sizes"]
        # plot
        self._ax[0, 1].violinplot([gt_sizes, pred_sizes])
        self._ax[0, 1].set_title(f"Size of cells - ks p score: {pred_batch_metrics['ks_p']}")
        self._ax[0, 1].set_xticks([1, 2])
        self._ax[0, 1].set_xticklabels(["ground truth", "prediction"])
        self._ax[0, 1].set_ylim([0, 40])

        # plt 3 intersection over union
        # get data
        ious = pred_metrics['ious']
        # plot
        self._ax[1, 0].boxplot(ious)
        self._ax[1, 0].set_title(f"Cells overlap - Jaccard index: {pred_batch_metrics['iou_mean']}")
        self._ax[1, 0].set_ylim([.50,1])

        # plt 4 f_score
        # get data
        f1_scores = pred_metrics['image_f1']
        # plot
        self._ax[1, 1].hist(f1_scores)
        self._ax[1, 1].set_title(f"Classification accuracy - f1 score: {pred_batch_metrics['f1_score']}")
        self._ax[1, 1].set_xlim([0.5, 1])

        self._fig.suptitle(f"{pred_infos['exp_name']} diagnostic")
        self._fig.show()

    def _get_contours(self, img) -> object:
        """ get contours of distinct objects in mask
        @param: img - 2d array with int values

        @returns tuple with contour information, dist_transform array"""

        if np.max(img) != 255:
            img = np.array(img * (255 / np.max(img)), dtype=np.uint8)

        dist_transform = cv2.distanceTransform(img, cv2.DIST_L1, 5)
        ret, last_image = cv2.threshold(dist_transform, 1e-06 * dist_transform.max(), 255, 0)
        last_image = np.uint8(last_image)
        # get contours
        cnts = cv2.findContours(last_image.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        return cnts, dist_transform

    def _count_cells(self, cnts) -> int:
        """counts distinct objects in mask
        @param: cnts - array with contour information (from _get_contours)
        """
        return len(cnts)

    def _get_cell_size(self, img, cnts, dist_transform) -> list:
        """ get cell size distribution in pixels
        @param : img - 2d array with 0 - 3 values
        @param : cnts - array with contour information (from _get_contours)
        @param : dist_transform - dist transform array( from _get_contours)
        @returns list of cell sizes
        """

        size_list = []
        for cnt in cnts:
            bbox = cv2.boundingRect(cnt)
            bbox_remapped = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
            max_size = np.max(dist_transform[bbox_remapped[0]: bbox_remapped[1],
                              bbox_remapped[2]: bbox_remapped[3]])
            size_list.append(int(max_size))
        return size_list

    def _int_over_union(self, img, img2) -> list:
        """ computes intersection over union metric, excluding areas with false positives
        @ param: img - 2d array with 0 - 3 values (ground truth)
        @ param: img2 - 2d array with 0 - 3 values (prediction)
        """

        inters = img & img2
        union = img + img2

        iou = np.sum(inters > 0) / np.sum(union > 0)

        return iou

    def _run_permutation_test(self, dist1, dist2, num_samples=1000):
        delta = abs(mean(dist1) - mean(dist2))
        pooled = dist1 + dist2
        estimates = []
        for _ in range(num_samples):
            np.random.shuffle(pooled)
            starZ = pooled[:len(dist1)]
            starY = pooled[-len(dist2):]
            estimates.append(abs(mean(starZ) - mean(starY)))
        diffCount = len(np.where(np.array(estimates) <= delta)[0])
        hat_asl_perm = (float(diffCount) / float(num_samples))
        return hat_asl_perm

    def _calc_f1_score(self, y_true, y_pred):
        """Calculates weight-averaged f1 score.

        @param y_true unraveled (1D array) ground truth
        @param y_true unraveled (1D array) prediction
        """

        report = classification_report(y_true, y_pred, digits=3, output_dict=True)
        return report['weighted avg']['f1-score']

    def _calc_f1_score_old(self, conf_matrix) -> np.array:
        """ calculates f1 score using confusion matrix
        :param conf_matrix: confusion matrix calculated with advanced_image_metrics
        :return: float number f1_score
        """
        # from list of arrays to 2d array

        # check the array is suitable for f1 score
        if (conf_matrix.shape[0] != conf_matrix.shape[1]) or \
                (np.sum(conf_matrix) <= 0):
            assert "ShapeError: problem with conf_matrix"
        f1_score = 0
        for i in range(conf_matrix.shape[0]):
            tp = conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i,]) - fp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            class_f1_score = 2 * (precision * recall) / (precision + recall)
            f1_score += class_f1_score
        return f1_score / 3
