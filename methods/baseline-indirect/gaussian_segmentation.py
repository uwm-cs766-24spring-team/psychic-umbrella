"""
Gaussian mixture model + segmentation
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sklearn.metrics

"""
Methods for scoring binary classification

All methods share a common signature:

Args:
    preds: list of predictions (presence of planar deviation) as a binary array
    labels: list of labels, in the format provided by lib.data_loaders.load_labeled_data
    negative_classes: list of classes to treat as negative. Default is ["none"].
Returns:
    float: score in range [0, 1]
"""
def no_fp_fn_score(preds, labels, negative_classes=["none"], threshold=0.05):
    """
    No false positives; false negative score. In the object detection case, it might be more
    important to avoid any false positives (cases where an object is detected when there is nothing)
    than to catch every true positive. In this score, we count the rate of false negatives, but if
    there is a false positive rate over threshold, the score is 0.
    """
    y_true = np.array([l["object"] not in negative_classes for l in labels])
    y_pred = np.array(preds)

    fp_rate = y_pred[~y_true].mean()
    if fp_rate > threshold:
        return 0
    else:
        # return false negative rate
        return y_pred[y_true].mean()


def f1_score(preds, labels, negative_classes=["none"]):
    y_true = np.array([l["object"] not in negative_classes for l in labels])
    y_pred = np.array(preds)
    return sklearn.metrics.f1_score(y_true, y_pred)


def precision_score(preds, labels, negative_classes=["none"]):
    y_true = np.array([l["object"] not in negative_classes for l in labels])
    y_pred = np.array(preds)
    return sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)


def recall_score(preds, labels, negative_classes=["none"]):
    y_true = np.array([l["object"] not in negative_classes for l in labels])
    y_pred = np.array(preds)
    return sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)


def jaccard_score(preds, labels, negative_classes=["none"]):
    y_true = np.array([l["object"] not in negative_classes for l in labels])
    y_pred = np.array(preds)
    # average=micro means average over both classes as the positive label, not only one
    return sklearn.metrics.jaccard_score(y_true, y_pred, average="binary", zero_division=0)


def accuracy_score(preds, labels, negative_classes=["none"]):
    y_true = np.array([l["object"] not in negative_classes for l in labels])
    y_pred = np.array(preds)
    return sklearn.metrics.accuracy_score(y_true, y_pred)


def balanced_accuracy_score(preds, labels, negative_classes=["none"]):
    y_true = np.array([l["object"] not in negative_classes for l in labels])
    y_pred = np.array(preds)
    if len(set(y_pred)) > len(set(y_true)):
        return sklearn.metrics.accuracy_score(y_true, y_pred)
    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred)


"""
Methods for scoring distance prediction

All methods share a common signature:

Args:
    preds: list of predictions (distance to nearest planar deviation) as a float array
    labels: list of labels, in the format provided by lib.data_loaders.load_labeled_data

Returns:
    float: some accuracy score
"""

def tp_mean_absolute_error(dist_preds, binary_preds, labels):
    """
    Mean absolute error, but only over samples which were predicted as true positives by the binary
    classifier
    """
    try:
        # get only the first distance prediction for each sample
        dist_preds = [p[0] for p in dist_preds]
    except IndexError: # if there are no distance predictions for one of the samples
        return np.nan
    tp_idxs = np.logical_and(np.array(binary_preds).astype(bool), np.array([l["object"] != "none" for l in labels]))
    
    # filter dist_preds and labels to only the true positive samples
    y_true = np.array([l["distance"] if "distance" in l.keys() else None for l in labels])[tp_idxs]
    y_pred = np.array(dist_preds)[tp_idxs]

    assert len(y_true) == len(y_pred)
    if len(y_true) == 0:
        return np.nan

    return np.abs(y_true - y_pred).mean()

def mean_absolute_error(dist_preds, labels):
    """
    Mean absolute error over all samples which have a true distance that is non-zero
    """
    pass

    try:
        # get only the first distance prediction for each sample
        dist_preds = [p[0] for p in dist_preds]
    except IndexError: # if there are no distance predictions for one of the samples
        return np.nan

    # filter dist_preds and labels to only the samples with a gt distance
    y_true = np.array([l["distance"] if "distance" in l.keys() else None for l in labels])

    assert len(dist_preds) == len(y_true)

    # if all the distance labels are "none" (meaning there are no objects in any observation)
    if len(y_true) == 0:
        return np.nan

    y_pred = np.array(dist_preds)[y_true != None]
    y_true = y_true[y_true != None]
    return np.abs(y_true - y_pred).mean()


class ThresholdClassifier:
    def __init__(self, metric=balanced_accuracy_score):
        self.metric = metric
        self.threshold = None

    def fit(self, datapoints, labels, threshold_range=None, threshold_points=1000, verbose=False):
        """
        Args:
            datapoints: (n_datapoints, 1): we keep this format to be consistent with the arguments
                used by scipy LinearRegression.
            labels: list of length n_datapoints. Should be raw labels

        Returns:
            float, the best accuracy achieved
        """
        datapoints = datapoints[:, 0]  # convert back to 1d

        if threshold_range == None:
            # start at the 10th percentile smallest of the data
            datapoints_nan = np.ma.filled(datapoints, np.nan)
            thresholds = np.linspace(
                np.nanpercentile(datapoints_nan, 10), datapoints.max(), threshold_points
            )
        else:
            thresholds = np.linspace(*threshold_range, threshold_points)

        accuracies = []
        for threshold in thresholds:
            preds = datapoints < threshold
            accuracies.append(self.metric(preds, labels))

        self.threshold = thresholds[np.argmax(accuracies)]

        if verbose:
            fig, ax = plt.subplots()
            ax.plot(thresholds, accuracies)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Accuracy")
            plt.show()

        return np.max(accuracies)

    def predict(self, datapoints):
        datapoints = datapoints[:, 0]
        return datapoints < self.threshold


class DepthImageGaussianWrapper:
    """
    Wrap common functionality of per-bin gaussian based methods for images.
    """

    valid_distribution_types = ["Gaussian", "GMM"]

    def __init__(self, pdf_percentile=100.0, distribution_type="Gaussian"):
        """
        Args:
            pdf_percentile: float, the percentile of the least probable pixels to use to calculate the
                joint pdf. 100% is equivalent to summing all pdfs. 0% is equivalent to taking the
                max pdf.
            distribution_type: str, one of DepthImageGaussianWrapper.valid_distribution_types
        """

        if distribution_type not in DepthImageGaussianWrapper.valid_distribution_types:
            raise ValueError(
                f"distribution_type must be one of {DepthImageGaussianWrapper.valid_distribution_types}"
            )
        self.pdf_percentile = pdf_percentile
        self.distribution_type = distribution_type

    def fit(self, samples, labels, verbose=False):
        """
        Args:
            samples: (n_samples, img_height, img_width)
            labels: (n_samples,): binary labels
        """

        assert type(samples) == np.ma.masked_array

        bg_samples = samples[~labels]  # (n_bg_samples, img_height, img_width)
        bg_samples_flat = bg_samples.reshape(bg_samples.shape[0], -1)  # (n_bg_samples, n_pixels)

        if self.distribution_type == "Gaussian":

            # because of distance clipping, there will almost certainly be some pixels are masked
            # in some samples but not in others. This creates a problem when e.g. the pixel is
            # unmasked in only 3 samples, as we can't fit a good Gaussian to it. So we need to
            # flatten the mask across the samples dimension, so that if a pixel is masked in any
            # one sample, it is masked in all of them
            bg_samples_flat.mask = bg_samples_flat.mask.any(axis=0, keepdims=True)

            # mask out any pixels that are zero with a mask array, in addition to existing mask
            bg_samples_flat.mask = bg_samples_flat.mask | (bg_samples_flat == 0.0)

            self.bg_means = bg_samples_flat.mean(axis=0)  # (n_pixels,)
            self.bg_stds = bg_samples_flat.std(axis=0)  # (n_pixels,)

            if verbose:
                fig, ax = plt.subplots()
                ax.imshow(self.bg_means.reshape(samples.shape[1:]), cmap="gray")
                ax.set_title("Mean depth image")
                plt.show()

        elif self.distribution_type == "GMM":
            raise ValueError("GMM not implemented for images")
            # n_gaussians_range = np.arange(1, 11)
            # self.bin_classifiers = []
            # for bin_idx in tqdm(range(bg_hists.shape[2])):
            #     curr_models = []
            #     curr = bg_hists[:, :, bin_idx]
            #     for i in range(curr.shape[1]):
            #         models = [None for _ in range(len(n_gaussians_range))]
            #         for j in range(len(n_gaussians_range)):
            #             models[j] = GaussianMixture(n_components=n_gaussians_range[j]).fit(
            #                 curr[:, i].reshape(-1, 1)
            #             )
            #         AIC = [m.aic(curr[:, i].reshape(-1, 1)) for m in models]
            #         curr_models.append(models[np.argmin(AIC)])
            #     self.bin_classifiers.append(curr_models)

    def predict(self, samples, return_raw=False):
        samples_flat = samples.reshape(samples.shape[0], -1)  # (n_samples, n_pixels)

        if self.distribution_type == "Gaussian":
            per_pixel_pdfs = (
                (1 / (self.bg_stds * np.sqrt(2 * np.pi)))[None, :]
                * np.exp(
                    -0.5
                    * (((samples_flat - self.bg_means[None, :]) / (self.bg_stds[None, :])) ** 2)
                )
                * self.bg_stds[None, :]
            )  # (n_samples, n_pixels)

            # print("depth at 16149", samples_flat[0][16149])
            # print("prob at 16149 (before mask)", per_pixel_pdfs[0][16149])

            # mask out the pdfs for pixels that are zero, so that they won't affect the joint pdf
            # calculation later
            per_pixel_pdfs.mask = per_pixel_pdfs.mask | (samples_flat == 0.0)

            # print("prob at 16149 (after mask)", per_pixel_pdfs[0][16149])

            # numpy has a strange (and maybe undocumented?) feature where if you generate a NaN
            # in a masked array, it automatically masks out that value. This is not what we want,
            # as we want pixels with a prob of 0.0 to remain in the joint PDF calculation.
            not_masked_and_zero = np.logical_and(~per_pixel_pdfs.mask, per_pixel_pdfs == 0.0)
            with np.errstate(divide="ignore"):
                log_per_pixel_pdfs = np.log(per_pixel_pdfs)  # (n_samples, n_pixels)
            log_per_pixel_pdfs.mask = per_pixel_pdfs.mask

            # using -inf or nan causes problems with e.g. scipy LogisticRegression, so set to a
            # large negative number instead
            log_per_pixel_pdfs[not_masked_and_zero] = -1e100

            # print("log prob at 16149", log_per_pixel_pdfs[0][16149])
        elif self.distribution_type == "GMM":
            raise ValueError("GMM not implemented for images")
            # log_per_bin_pdfs = np.zeros(samples.shape)

            # for sample_idx in tqdm(range(samples.shape[0])):
            #     for bin_idx in range(samples.shape[2]):
            #         for zone_idx in range(samples.shape[1]):
            #             model = self.bin_classifiers[bin_idx][zone_idx]
            #             log_per_bin_pdfs[sample_idx, zone_idx, bin_idx] = model.score_samples(
            #                 samples[sample_idx, zone_idx, bin_idx].reshape(-1, 1)
            #             )[0]

        # percentile ignores masks on masked arrays, so create a version of the log_per_pixel_pdfs
        # where masked values are set to nan, then use nanpercentile
        log_per_pixel_pdfs_nan = np.ma.filled(log_per_pixel_pdfs, np.nan)  # (n_samples, n_pixels)
        cutoff = np.nanpercentile(
            log_per_pixel_pdfs_nan, self.pdf_percentile, axis=1
        )  # (n_samples,)

        # cutoff log per pixel pdfs to those higher than the cutoff (i.e. more probable than the cutoff)
        # because a value of 1 in the mask means to remove it, we want to apply the
        # mask to all values greater than the cutoff, hince the > comparison
        cutoff_log_per_pixel_pdfs = np.ma.masked_array(
            log_per_pixel_pdfs, log_per_pixel_pdfs > cutoff[:, None]
        )  # (n_samples, n_pixels)

        probs = cutoff_log_per_pixel_pdfs.sum(axis=1)  # (n_samples,)

        if return_raw:
            return probs, log_per_pixel_pdfs
        else:
            return probs


class DepthImagePerPixelGaussian:
    def __init__(self):
        self.classifier = None

    def train(self, samples, raw_labels, metric=balanced_accuracy_score, verbose=False):
        remove_surfaces = ["foamboard", "chipboard"]
        # samples on some surfaces give trouble because the depth image is
        # poor quality due to the uniform surface texture. Filter those images out
        samples = [s for s, l in zip(samples, raw_labels) if l["surface"] not in remove_surfaces]
        raw_labels = [l for l in raw_labels if l["surface"] not in remove_surfaces]

        # (n_samples, img_height, img_width)
        images = np.ma.masked_array([s["depth_image"] for s in samples])
        # (n_samples,)
        labels = np.array([False if l["object"] == "none" else True for l in raw_labels])

        accuracies = []
        # pdf_percentiles = np.linspace(0, 100, 10, endpoint=True)
        pdf_percentiles = np.linspace(0, 1, 25, endpoint=True)
        # pdf_percentiles = [94.0]
        for pdf_percentile in tqdm(pdf_percentiles, desc=f"Training {self.__class__.__name__}"):
            # we can use the Gaussian wrapper if we treat each pixel as a separate histogram zone,
            # and think of each histogram as only having one bin
            self.wrapper = DepthImageGaussianWrapper(
                pdf_percentile=pdf_percentile, distribution_type="Gaussian"
            )
            self.wrapper.fit(images, labels, verbose=False)

            per_sample_pdfs = self.wrapper.predict(images)

            self.classifier = ThresholdClassifier(metric=metric)
            accuracies.append(self.classifier.fit(per_sample_pdfs[:, None], raw_labels))

        best_pdf_percentile = pdf_percentiles[np.argmax(accuracies)]
        self.wrapper = DepthImageGaussianWrapper(
            pdf_percentile=best_pdf_percentile, distribution_type="Gaussian"
        )
        self.wrapper.fit(images, labels, verbose=verbose)

        per_sample_pdfs = self.wrapper.predict(images)

        self.classifier.fit(per_sample_pdfs[:, None], raw_labels)

        if verbose:
            print(f"Trained {self.__class__.__name__}")
            print(f"Best percentile: {best_pdf_percentile}")
            for pdf_percentile, training_loss in zip(pdf_percentiles, accuracies):
                print(f"Training loss for {pdf_percentile}: {training_loss:.3f}")
            fig, ax = plt.subplots()
            ax.plot(pdf_percentiles, accuracies)
            ax.set_xlabel("Percentile")
            ax.set_ylabel("Training Loss")
            fig.suptitle(f"Training Loss for {self.__class__.__name__}")
            plt.show()

    def predict(self, samples, return_raw=False, return_dists=False, force_distance=False):
        if return_raw and return_dists:
            raise ValueError("Can't return both raw and dists")

        images = np.ma.masked_array([s["depth_image"] for s in samples])

        per_sample_pdfs, per_pixel_pdfs = self.wrapper.predict(images, return_raw=True)

        if return_raw:
            return self.classifier.predict(per_sample_pdfs[:, None]), per_pixel_pdfs.reshape(*images.shape)
        elif return_dists:
            return self.classifier.predict(per_sample_pdfs[:, None]), [[0.0]] * len(samples) # TODO: implement
        else:
            return self.classifier.predict(per_sample_pdfs[:, None])