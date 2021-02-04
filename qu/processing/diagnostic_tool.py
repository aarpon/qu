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


def get_contours(img) -> object:
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


def count_cells(cnts) -> int:
    """counts distinct objects in mask
    @param: cnts - array with contour information (from get_contours)
    """
    return len(cnts[0])


def get_cell_size(img, cnts, dist_transform) -> list:
    """get cell size distribution in pixels
    @param: img - 2d array with 0-3 values
    @param: cnts - array with contour information (from get_contours)
    @param: dist_transform - dist transform array (from get_contours)
    @returns list of cell sizes  """

    size_list = []
    for cnt in cnts[0]:
        bbox = cv2.boundingRect(cnt)
        bbox_remapped = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
        max_size = np.max(dist_transform[bbox_remapped[0]: bbox_remapped[1],
                          bbox_remapped[2]: bbox_remapped[3]])
        size_list.append(int(max_size))
    return size_list


def detect_partial_segmentation(img, cnts, dist_transform) -> list:
    """ evaluate prediction quality of contour (class 2) vs actual cell (class 1)
    @param: img - 2d array with 0-3 values
    @param: cnts - array with contour information (from get_contours)
    @param: dist_transform - dist transform array (from get_contours)
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


def int_over_union(img, img2) -> list:
    """computes intersection over union metric, exclusing areas with false positives
    @param: img - 2d array with 0-3 values (ground truth)
    @param: img2 - 2d array with 0-3 values (prediction)
    """

    inters = img & img2
    union = img + img2

    iou = np.sum(inters > 0) / np.sum(union > 0)

    return (iou)




# batch assessment

class SegmentationDiagnostic():
"""creates a segmentation dignostic that analyses quality
of (one or more) predictions compared to ground truth"""
    def __init__(self, gt_fold_path, pred_fold_path=None):

        # ground truth folder path
        self.gt_folder: str = gt_fold_path

        # ground truth file list
        self.gt_file_list: list = os.listdir(self.gt_folder)
        self.gt_file_num: int = len(self.gt_file_list)

        # prediction(s) folder path & name
        self.prediction_paths:list = []
        self.prediction_names:list = []
        self.prediction_file_names:list = []

        # add predictions
        if pred_fold_path:
            if isinstance(pred_fold_path, list):
                self.add_prediction(*pred_fold_path)
            else:
                self.add_prediction(pred_fold_path)



    def add_prediction(self, *args) -> None:
        """loads prediction(s) folder and sets names"""
        for fold in args:
            if os.path.exists(fold) or len(os.listdir(fold) > 0):
                self.pred_path_list.append(fold)
                self.prediction_names = self.prediction_names.append(PurePath(fold).name)
                self.prediction_file_names = self.prediction_file_names.append( os.listdir(fold)
            else:
                assert f"FileNotFoundError: check the prediction folder {fold}"



    def check_segmentation_predictions(self, pred_num=0, verbose=False) -> dict:
        """Evaluates segmentation quality in batch of images.
        All errors are based on comparing ground truth and prediction(s)
        @param: gt_fold_path - path to ground truth folder
        @param: pred_fold_path - path to prediction folder
        @param: verbose - Bool or int between 0 and 2
        @ returns None"""
        # check for missing files in prediction

        missing_files = []

        if verbose:
            print(f"found {len(self.gt_file_num)} ground truth and {len(self.prediction_file_names[pred_num])} predictions")

        clean_fnames = sorted([f_name.split("_")[-1] for f_name in f])
        clean_pr_fnames = sorted([f_name.split("_")[-1] for f_name in pr_f])

        missing_files = [f for f in clean_fnames if f not in clean_pr_fnames]
        if len(missing_files) > 0:
            print(f"{len(missing_files)} missing images found: {missing_files} ")

        # detect segmentation errors per image
        cell_count = []
        cell_size = []
        ious = []
        conf_matrix = []

        # cycle through all images in batch
        for n in range(len(f)):
            # open images
            true_img_path = os.path.join(d, f[n])
            pred_img_path = os.path.join(d1, pr_f[n])
            if verbose > 1:
                print(f"evaluating img {f[n]} and prediction {pr_f[n]}")

            tr_img = np.array(Image.open(true_img_path))
            pr_img = np.array(Image.open(pred_img_path))

            # get contours and dist_transform
            tr_cnts, tr_dist_trans = get_contours(tr_img)
            pr_cnts, pr_dist_trans = get_contours(pr_img)

            # count distinct objects / cells
            cell_count.append([count_cells(tr_cnts), count_cells(pr_cnts)])

            if verbose >= 2:
                print(f"number of unique objects: {len(tr_cnts[0])}, {len(pr_cnts[0])} \n")

            # get size distribution
            tr_sizes = get_cell_size(tr_img, tr_cnts, tr_dist_trans)
            pr_sizes = get_cell_size(pr_img, pr_cnts, pr_dist_trans)
            cell_size.append([pr_sizes, tr_sizes])

            # compute intersection over union
            ious.append(int_over_union(tr_img, pr_img))
            # get confusion matrix
            single_conf = confusion_matrix(tr_img.flatten(), pr_img.flatten())
            conf_matrix.append(single_conf)

        if verbose:
            # create plots on errors
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))

            # plot 1 - cell count
            max_plot = np.quantile(np.array(cell_count).flatten(), 0.95)
            axs[0, 0].scatter(np.array(cell_count)[:, 0], np.array(cell_count)[:, 1])
            axs[0, 0].set_xlim([0, max_plot])
            axs[0, 0].set_ylim([0, max_plot])
            axs[0, 0].set_title("Cell count")
            axs[0, 0].set_xlabel("ground truth")
            axs[0, 0].set_ylabel("prediction")

            # plot 2 - cell size
            avg_size_diff = np.array([[mean(x[0]), mean(x[1])] for x in cell_size])
            axs[0, 1].violinplot(avg_size_diff)
            axs[0, 1].set_title("cell size in pixels")
            axs[0, 1].set_xticks([1, 2])
            axs[0, 1].set_xticklabels(["prediction", "ground truth"])

            # plt 3 intersection over union
            axs[1, 0].boxplot([ious])
            axs[1, 0].set_title("intersection over union")

            # plt 4 confusion matrix
            cum_conf_matrix = np.sum(np.array(conf_matrix), axis=0)
            sns.heatmap(cum_conf_matrix, ax=axs[1, 1],
                        cmap="PiYG",
                        annot=True,
                        xticklabels=["pred BG", "pred CELL", "pred CONTOUR"],
                        yticklabels=["true BG", "true CELL", "true CONTOUR"])
            axs[1, 1].set_title("classification accuracy")

        segmentation_errors = {"cell_count": cell_count,
                               "cell_size": cell_size,
                               "confusion_matrix": conf_matrix}
        if verbose == 2:
            print(segmentation_errors)

        return segmentation_errors




# running assessment on UNET predictions

gt_fold_path = "/home/matt/.qu/data/demo_segmentation/masks"

pred_fold_path = "/home/matt/.qu/data/demo_segmentation/UNET_preds"
_= check_segmentation_predictions(gt_fold_path, pred_fold_path, 1)