# Copyright (c) Northwestern Argonne Institute of Science and Engineering (NAISE)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
def get_args_parser():
    parser = argparse.ArgumentParser('Features post processing using dimensionality reduction and clustering', add_help=False)
    parser.add_argument('--features_path', default='', type=str, help="Path to the features to be processed.")

    # For the dim red method
    parser.add_argument('--dimensions', default=2, type=int, help='Reduce to this number of dimensions (Default 2).')
    parser.add_argument('--dim_red_method', default='PCA', type=str, help="Dimensionality reduction method (Default: PCA).")


    # For the clustering method
    parser.add_argument('--clustering_method', default='DBSCAN', type=str, help="Clustering method (Default: DBSCAN).")
    parser.add_argument("--dbscan_eps", type=float, default=0.5, help="""This is for the DBSCAN algorithm.
            The maximum distance between two samples for one to be considered as in the neighborhood of the other (default=0.5).""")
    parser.add_argument('--dbscan_min_samples', default=5, type=int, help="""This is for the DBSCAN algorithm.
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself (Default 5).""")

    # General arguments
    parser.add_argument('--output_dir', default='.', help='Path where to save clustering results.')

    return parser








def process_features(args):
    x = bring_features(args.features_path)
    x = reduce_dim(x, args.dim_red_method, args.dimensions)
    y = cluster_data(x, args)

    clusters = {'x': x, 'y': y}
    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    clusters_fname = os.path.join(args.output_dir, "clusters")
    np.save(clusters_fname, clusters)
    print(f"{clusters_fname} saved.")




def bring_features(path):
    feats = torch.load(path)
    feats = StandardScaler().fit_transform(feats)
    return feats


def reduce_dim(feats, method='PCA', dimensions=2):
    if method == 'PCA':
        pca = PCA(n_components=dimensions, svd_solver='full')
        pca.fit(feats)
        return pca.transform(feats)
    elif method == 'SVD':
        svd = SVD(n_components=dimensions)
        svd.fit(feats)
        return svd.transform(feats)
    else:
        raise NameError(f"Unknow dimensionality reduction method: {method}")



def cluster_data(data, args):
    if args.clustering_method == 'DBSCAN':
        # Compute Density-based spatial clustering of applications with noise (DBSCAN)
        labels = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit_predict(data)
    else:
        raise NameError(f"Unknow clustering algorithm method: {args.clustering_method}")

    return labels







if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cloud-DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    process_features(args)
