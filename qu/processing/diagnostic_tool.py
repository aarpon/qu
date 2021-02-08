#   /********************************************************************************
#   * Copyright © 2020-2021, ETH Zurich, D-BSSE, Aaron Ponti
#   * All rights reserved. This program and the accompanying materials
#   * are made available under the terms of the Apache License Version 2.0
#   * which accompanies this distribution, and is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.txt
#   *
#   * Contributors:
#   *     Matteo Jucker Riva - contriobutor
#   *******************************************************************************/
#   NOTES:
#   this module allows to call a diagnostic tool to evaluate the quality of
#   a segmentation  using ground truth maasks and prediciton results


# object based analysis of results
import cv2
import os
from statistics import mean, stdev
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from pathlib import PurePath
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



# batch assessment

class SegmentationDiagnostic():
    """creates a segmentation diagnostic that analyses quality
    of (one or more) prediction batches compared to ground truth"""

    def __init__(self, gt_fold_path, pred_fold_path=None):

        # storage for ground truth attributes
        # "infos" include paths, file list and number of files
        # "metrics" contains:
        #  - cell contours per image
        #  - dist_transform per image
        #  - cell count per image
        #  - cell size per image

        self.gt = {"infos": {},
                   "metrics": {}}

        # prediction is a list of dictionary with len== # number of prediction batches to analyse\
        # each dictionary includes path, experiment_name, file_names, file_num
        self.pred_infos = []

        # image level metrics for each batch: a list of dictionaries (one per batch) containing
        # - contours
        # - dist_transforms

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
        self.pred_batch_metrics = {"cell_count_plot": None,
                                   "cell_size_plot": None,
                                   "iou_plot": None,
                                   "f-score_plot": None}

        # load gt infos
        self.gt["infos"] = self.load_batch_infos(gt_fold_path, "ground_truth")

        #  load predictions infos
        if pred_fold_path:
            if not isinstance(pred_fold_path, list):
                pred_fold_path = [pred_fold_path]
            self.pred_infos = [self.load_batch_infos(path) for path in pred_fold_path]

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
        - cell contours per image,
        - dist_transform of foreground
        - cell count per image
        - cell sizes per image"""
        new_metric_dic = {"cell_contours": [None],
                          "dist_transforms": [None],
                          "cell_count": [None],
                          "cell_sizes": [None]}

        infos_dic = self.pred_infos[pred_num] if pred_num >= 0 else self.gt['infos']

        # check if batch infos are loaded
        if not infos_dic["path"]:
            assert f" FileNotFoundError, no information available. \
            Maybe execute load_batch_infos first? "

        # Scan through ech image
        metric_dic = new_metric_dic.copy()
        for n, f in range(infos_dic['file_names']):
            if verbose >= 2:
                print(f"evaluating image {f}")

            # Load img
            img_path = os.path.join(infos_dic['path'], f)
            img = np.array(Image.open(img_path))

            # Check if image is indeed segmentation
            if (len(np.unique(img)) > 10) or \
                    (img.dtype != int):
                assert "ValueError: image is not a segmentation mask"

            # get contours and dist_transform
            (cnts, _), dist_trans = self._get_contours(img)
            metric_dic["cell_contours"][n] = cnts
            metric_dic["dist_trasnforms"][n] = dist_trans

            # count distinct objects / cells
            metric_dic['cell_count'][n] = self._count_cells(cnts)
            if verbose >= 2:
                print(f"number of unique objects: {self._count_cells(cnts)} \n")

            # get size distribution
            metric_dic["sizes"][n] = self._get_cell_size(img, cnts, dist_trans)

            # check coherence
            if (len(metric_dic["cell_sizes"]) != infos_dic["file_num"]) and \
                    (metric_dic["cell_contours"][0] == None):
                assert "Problem with analysis, missing metrics"
            return None
        else:
            return metric_dic

    def calc_advanced_image_metrics(self, pred_num, verbose=False):
        """calculates advanced image level metrics that need comparison between ground truth
        and prediction batches """
        new_advanced_metric_dic = {
            "ious": [None],
            "conf_matrix": [None]
        }
        # get infos + metrics for current prediction and gt
        gt_infos = self.gt['infos']
        gt_metrics = self.gt["metrics"]
        pred_infos = self.pred_infos[pred_num]
        pred_metrics = self.pred_metrics[pred_num]

        # check that ground truth metrics are available
        if (not gt_metrics) or (not gt_metrics["cell_contours"]):
            gt_metrics = self.calc_image_metrics(-1)

        # check that gt files and pred files are the same if not filter gt files
        clean_gt_names = sorted([f_name.split("_")[-1] for f_name in gt_infos["file_names"]])
        clean_pred_names = sorted([f_name.split("_")[-1] for f_name in pred_infos["file_names"]])

        if (gt_infos["file_num"] != pred_infos["file_num"]) or clean_gt_names != clean_pred_names:
            selected_gt_files = [gt_file for gt_file in sorted(gt_infos["file_names"]) \
                                 if gt_file.split("_")[-1] in pred_infos["file_names"]]

            selected_pred_files = [pr_file for pr_file in sorted(pred_infos['file_names']) \
                                   if pr_file.split("_")[-1] in gt_infos["file_name"]]
        else:
            selected_gt_files = gt_infos["file_names"]
            selected_pred_files = pred_infos["file_names"]

        # Scan through ech image
        advanced_metric = new_advanced_metric_dic.copy()
        for n, (gt_file, pred_file) in enumerate(zip(selected_gt_files, selected_pred_files)):
            # Load images
            gt_img_path = os.path.join(gt_infos['path'], gt_file)
            gt_img = np.array(Image.open(gt_img_path))
            pred_img_path = os.path.join(pred_infos['path'], pred_file)
            pred_img = np.array(Image.open(pred_img_path))

            # calc_int_over_union
            advanced_metric['ious'].append(self._int_over_union(gt_img, pred_img))
            # get confusion matrix
            single_conf = confusion_matrix(gt_img.flatten(), pred_img.flatten())
            advanced_metric['conf_matrix'].append(single_conf)
            return advanced_metric

    def calc_batch_metrics(self, pred_num=0):
        """batch level metrics comparing ground truth and prediction"""

        batch_metrics = {"cell_count_p": [None],
                                "cell_size_p": [None],
                                "iou_mean": [None],
                                "f1_score": [None]}

        gt_metrics = self.gt["metrics"]
        # check that all ground truth metrics have been calculated:
        if not gt_metrics["cell_count"] or (not gt_metrics["cell_count"][0]) or:
            gt_metrics = self.calc_image_metrics(-1, "gt") }

        # prediction image level metrics
        try:
            pred_metrics = self.pred_metrics[pred_num]
        except IndexError:
            pred_metrics={**self.calc_image_metrics(pred_num), **self.calc_advanced_image_metrics(pred_num)}

        # assess difference in cell count
        gt_count = gt_metrics['cell_count']
        pr_count = pred_metrics["cell_count"]
        batch_metrics["cell_count_p"] = self.run_permutation_test(gt_count, pr_count)
        print("cell_count_p_value", batch_metrics["cell_count_p"],"\n")

        # assess difference in cell sizes
        gt_sizes = gt_metrics['cell_size'].flatten()
        pr_sizes = pred_metrics['cell_size'].flatten()
        batch_metrics["cell_size_p"] = self.run_permutation_test(gt_sizes, pr_sizes)
        print("cell_size_p_value:",batch_metrics["cell_size_p"], "\n")

        # calc average iou
        batch_metrics["mean_iou"] = pred_metrics['iou'].mean()

        # calc f value
        batch_metrics["f1_score"] = self._calc_batch_f1score(pred_num)

        # if verbose:
        #     # create plots on errors
        #     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
        # 
        #     # plot 1 - cell count
        #     max_plot = np.quantile(np.array.gt_cell_count).flatten(), 0.95)
        #     axs[0, 0].scatter(np.array.gt_cell_count)[:, 0], np.array.gt_cell_count)[:, 1])
        #     axs[0, 0].set_xlim([0, max_plot])
        #     axs[0, 0].set_ylim([0, max_plot])
        #     axs[0, 0].set_title("Cell count")
        #     axs[0, 0].set_xlabel("ground truth")
        #     axs[0, 0].set_ylabel("prediction")
        # 
        #     # plot 2 - cell size
        #     avg_size_diff = np.array([[mean(x[0]), mean(x[1])] for x in cell_size])
        #     axs[0, 1].violinplot(avg_size_diff)
        #     axs[0, 1].set_title("cell size in pixels")
        #     axs[0, 1].set_xticks([1, 2])
        #     axs[0, 1].set_xticklabels(["prediction", "ground truth"])
        # 
        #     # plt 3 intersection over union
        #     axs[1, 0].boxplot([ious])
        #     axs[1, 0].set_title("intersection over union")
        # 
        #     # plt 4 confusion matrix
        #     cum_conf_matrix = np.sum(np.array(conf_matrix), axis=0)
        #     sns.heatmap(cum_conf_matrix, ax=axs[1, 1],
        #                 cmap="PiYG",
        #                 annot=True,
        #                 xticklabels=["pred BG", "pred CELL", "pred CONTOUR"],
        #                 yticklabels=["true BG", "true CELL", "true CONTOUR"])
        #     axs[1, 1].set_title("classification accuracy")
        # 
        # segmentation_errors = {.gt_cell_count":.gt_cell_count,
        #                        "cell_size": cell_size,
        #                        "confusion_matrix": conf_matrix}
        # if verbose == 2:
        #     print(segmentation_errors)
        # 
        # return segmentation_errors

    # internal methods
    def _load_gt_images(self):
        """
        load and check
        gt
        images
        """
        gt_file_list = sorted(os.listdir(self.gt_folder))
        for file in gt_file_list:
            image_full_path = os.path.join(self.gt_folder, file)
            image = np.array(Image.open(image_full_path))
            if len(np.unique(image)) > 2:
                self.gt_images.append(image)
            else:
                print(f"mask {file} has a problem, discarding!!")
                gt_file_list.remove(file)
                # add good images to class variables
        self.gt_file_list = gt_file_list
        self.gt_file_num = len(gt_file_list)
        print(f"imported {self.gt_file_num} images out of {len(os.listdir(self.gt_folder))}")

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
        return len(cnts[0])


    def _get_cell_size(self, img, cnts, dist_transform) -> list:
        """ get cell size distribution in pixels
        @param : img - 2d array with 0 - 3 values
        @param : cnts - array with contour information (from _get_contours)
        @param : dist_transform - dist transform array( from _get_contours)
        @returns list of cell sizes
        """

        size_list = []
        for cnt in cnts[0]:
            bbox = cv2.boundingRect(cnt)
            bbox_remapped = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
            max_size = np.max(dist_transform[bbox_remapped[0]: bbox_remapped[1],
                              bbox_remapped[2]: bbox_remapped[3]])
            size_list.append(int(max_size))
        return size_list

        def _detect_partial_segmentation(self, img, cnts, dist_transform) -> list:
            """ evaluate prediction quality of contour (class 2) vs actual cell ( class 1)
                @ param: img - 2d array with 0 - 3 values
                @ param: cnts - array with contour information (from _get_contours)
                @ param: dist_transform - dist array (from _get_contours)
                @returns count of wrong cells
            """
            size_list = []
            problem_list = []
            for cnt in cnts[0]:
                bbox = cv2.boundingRect(cnt)
                bbox_remapped = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
                aoi_dist = dist_transform[bbox_remapped[0]: bbox_remapped[1],
                           bbox_remapped[2]: bbox_remapped[3]]
                aoi = img[bbox_remapped[0]: bbox_remapped[1],
                      bbox_remapped[2]: bbox_remapped[3]]
                contour = np.ma.masked_where(aoi_dist > 2, aoi)
                inner = np.ma.masked_where(aoi_dist <= 2, aoi)
            if (1 in np.ma.unique(contour)) or (1 not in np.ma.unique(inner)):
                problem_list.append(cnt)

            return problem_list

        def _int_over_union(self, img, img2) -> list:
            """ computes intersection over union metric, excluding areas with false positives
            @ param: img - 2d array with 0 - 3 values (ground truth)
            @ param: img2 - 2d array with 0 - 3 values (prediction)
            """

            inters = img & img2
            union = img + img2

            iou = np.sum(inters > 0) / np.sum(union > 0)

            return (iou)

        def _run_permutation_test(self, dist1, dist2, num_samples=1000):
            delta = abs(dist1.mean() - dist2.mean())
            pooled = dist1 + dist2
            estimates = []
            for _ in range(num_samples):
                np.random.shuffle(pooled)
                starZ = pooled[:len(dist1)]
                starY = pooled[-len(dist2):]
                estimates.append(abs(starZ.mean() - starY.mean()))
            diffCount = len(np.where(estimates <= delta)[0])
            hat_asl_perm = 1.0 - (float(diffCount) / float(num_samples))
            return hat_asl_perm


if __name__ == "__main__":
    # running assessment on UNET predictions

    gt_fold_path = "/home/matt/.qu/data/demo_segmentation/masks"

    pred_fold_path = "/home/matt/.qu/data/demo_segmentation/UNET_preds"
    dig = SegmentationDiagnostic(gt_fold_path, pred_fold_path)
    print(dig.gt)
    print(dig.pred_infos)
    print(dig.gt['infos'])
    # for k,v in dig.metrics_per_image[0].items():
    #     print(k)
    #     print(v)
    #     print("\n")
    #
    # print("checking cell count")
    # print()
