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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import PurePath
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# batch assessment

class SegmentationDiagnostic():
    """creates a segmentation dignostic that analyses quality
    of (one or more) predictions compared to ground truth"""

    def __init__(self, gt_fold_path, pred_fold_path=None):

        # ground truth folder path
        self.gt_folder = gt_fold_path
        # ground truth file list
        self.gt_file_list = []
        self.gt_file_num = len(self.gt_file_list)
        self.gt_images = []

        # storage of metrics for ground truth (useful when comparing multiple predictions)
        self.gt_cnts = []
        self.gt_distance_transform = []
        self.gt_cell_count = []
        self.gt_cell_size = []

        # prediction(s) folder path & name
        self.prediction_paths = []
        self.prediction_names = []
        self.prediction_file_names = []

        # metrics evaluation output
        self.metrics_per_image = []

        # add predictions
        if pred_fold_path:
            if isinstance(pred_fold_path, list):
                print("adding predictions")
                self.add_predictions(*pred_fold_path)
            else:
                self.add_predictions(pred_fold_path)

    def add_predictions(self, *args) -> None:
        """loads prediction(s) folder and sets names"""
        for n, fold in enumerate(args):
            print(f"adding predictions folder {fold}")
            if os.path.exists(fold) and (len(os.listdir(fold)) > 0):
                self.prediction_paths.append(fold)
                self.prediction_names.append(PurePath(fold).name)
                self.prediction_file_names.append(sorted(os.listdir(fold)))
            else:
                assert f"FileNotFoundError: check the prediction folder {fold}"
            print(f"prediction folder {fold} has {len(self.prediction_file_names[0])} images")

    def calc_image_level_metrics(self, pred_num=0, verbose=False) -> dict:
        """Evaluates segmentation quality in batch of images (one prediction).
        @returns dictionary with:
        - cell count per image,
        - cell sizes per image,
        - intersection over union score per image,
        - confusion matrix per image"""
        # load gt images if needed
        if not self.gt_images:
            self._load_gt_images()
        # check for missing files in prediction

        # evaluate contours and distance transform for ground truth
        if (not self.gt_cnts) or (not self.gt_distance_transform):
            print("evaluating contours")
            for img in self.gt_images:
                cnts, dist = self._get_contours(img)
                self.gt_cnts.append(cnts[0])
                self.gt_distance_transform.append(dist)
                self.gt_cell_count.append(self._count_cells(cnts[0]))
                self.gt_cell_size.append(self._get_cell_size(img, cnts, dist))
        # load predictions with the same id as gt images
        missing_files = []
        clean_gt_fnames = sorted([f_name.split("_")[-1] for f_name in self.gt_file_list])
        clean_pr_fnames = sorted([f_name.split("_")[-1] for f_name in self.prediction_file_names[pred_num]])

        missing_files = [f for f in clean_gt_fnames if f not in clean_pr_fnames]
        if len(missing_files) > 0:
            print(f"{len(missing_files)} missing images found: {missing_files} ")

        # detect segmentation errors per image
        cell_count = []
        cell_size = []
        ious = []
        conf_matrix = []

        # The following cycle extracts image or object level metrics from the prediction batch
        for n in range(self.gt_file_num):
            if verbose >= 2:
                print(f"number of unique objects: GT - {len(self.gt_cnts[n])}, PRED - {len(pr_cnts[0])} \n")

            tr_img = self.gt_images[n]
            pred_img_path = os.path.join(self.prediction_paths[pred_num], self.prediction_file_names[pred_num][n])
            pr_img = np.array(Image.open(pred_img_path))

            # get contours and dist_transform
            pr_cnts, pr_dist_trans = self._get_contours(pr_img)

            # count distinct objects / cells
            cell_count.append(self._count_cells(pr_cnts[0]))

            # get size distribution
            pr_sizes = self._get_cell_size(pr_img, pr_cnts, pr_dist_trans)
            cell_size.append(pr_sizes)
            # compute intersection over union
            ious.append(self._int_over_union(tr_img, pr_img))
            # get confusion matrix
            single_conf = confusion_matrix(tr_img.flatten(), pr_img.flatten())
            conf_matrix.append(single_conf)
            if pred_num == 0:
                self.metrics_per_image = [{"cell_count": cell_count, "cell_size": cell_size, "ious": ious,
                                           "conf_matrix": conf_matrix}]
            else:
                self.metrics_per_image[pred_num] = ({"cell_count": cell_count, "cell_size": cell_size, "ious": ious,
                                                     "conf_matrix": conf_matrix})
            return None
    def batch_level_metrics(self, pred_num=0):
        try:
            batch_metrics = self.metrics_per_image[pred_num]
        except IndexError:
            self.calc_image_level_metrics(pred_num)
            batch_metrics = self.metrics_per_image[pred_num]

        # assess difference in cell count
        gt_count = self.gt_cell_count
        pr_count = batch_metrics["cell_count"]
        cell_count_p_value = self.run_permutation_test(gt_count, pr_count)
        print("cell_count_p_value", cell_count_p_value)

        # assess difference in cell sizes
        gt_sizes = self.gt_cell_size.flatten()
        pr_sizes = batch_metrics['cell_size'].flatten()
        cell_size_p_value = self.run_permutation_test(gt_sizes, pr_sizes)
        print(cell_size_p_value)

        # calc average iou
        mean_iou = batch_metrics['ious'].mean()

        # calc f value
        conf_array = batch_metrics["conf_matrix"].shape

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
        """load and check gt images"""
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
        """get contours of  distinct objects in mask (ground truth or presentation)
        @param: img - 2d array with 0-3 values
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
        """get cell size distribution in pixels
        @param: img - 2d array with 0-3 values
        @param: cnts - array with contour information (from _get_contours)
        @param: dist_transform - dist transform array (from _get_contours)
        @returns list of cell sizes  """

        size_list = []
        for cnt in cnts[0]:
            bbox = cv2.boundingRect(cnt)
            bbox_remapped = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
            max_size = np.max(dist_transform[bbox_remapped[0]: bbox_remapped[1],
                              bbox_remapped[2]: bbox_remapped[3]])
            size_list.append(int(max_size))
        return size_list

    def _detect_partial_segmentation(self, img, cnts, dist_transform) -> list:
        """ evaluate prediction quality of contour (class 2) vs actual cell (class 1)
        @param: img - 2d array with 0-3 values
        @param: cnts - array with contour information (from _get_contours)
        @param: dist_transform - dist transform array (from _get_contours)
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
        """computes intersection over union metric, exclusing areas with false positives
        @param: img - 2d array with 0-3 values (ground truth)
        @param: img2 - 2d array with 0-3 values (prediction)
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
    dig._load_gt_images()
    dig.add_predictions()
    print(dig.gt_folder, dig.gt_file_list)
    print(np.array(dig.gt_images).shape)
    print(dig.prediction_names)
    print(dig.prediction_file_names)
    dig.calc_image_level_metrics()
