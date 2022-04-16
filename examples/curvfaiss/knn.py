# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
# ==============================================================================

from absl import app
from absl import flags

import numpy as np
import curvfaiss
from curvlearn import manifolds
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_float("curvature", 0.0, "Curvature of stereographic model, default is 0")
flags.DEFINE_integer("batch", 16, "Embedding batch size")
flags.DEFINE_integer("dim", 8, "Embedding dimension")
flags.DEFINE_integer("topk", 10, "Find the top k nearest neighbors, default is 10")


def generate_sample():
    np.random.seed(2022)
    embedding = np.random.random((FLAGS.batch, FLAGS.dim)).astype('float32')
    id = np.arange(FLAGS.batch)

    def curvlearn_op(emb):
        manifold = manifolds.Stereographic()
        emb = manifold.to_manifold(tf.constant(emb, dtype=tf.float32), FLAGS.curvature)

        # src = [emb[0],..,emb[0],emb[1],...,emb[1],...]
        # dst = [emb[0],emb[1],...,emb[batch],emb[0],...]
        src = tf.reshape(tf.tile(emb, [1, FLAGS.batch]), [-1, FLAGS.dim])
        dst = tf.tile(emb, [FLAGS.batch, 1])
        distance = tf.reshape(manifold.distance(src, dst, FLAGS.curvature), [FLAGS.batch, FLAGS.batch])

        sess = tf.Session()
        emb, distance = sess.run([emb, distance])

        return emb, distance

    embedding, distance = curvlearn_op(embedding)

    knn_index = np.argsort(distance, axis=-1)
    knn_id = np.array([[id[col] for col in row[:FLAGS.topk]] for row in knn_index])

    return embedding, id, knn_id


def faiss_knn(emb, id, query=None):
    def build_index(embedding):
        """
        Stereographic distance only supports brute-force indexing method for now.
        """
        n, d = embedding.shape[0], embedding.shape[1]

        index = curvfaiss.IndexFlatStereographic(d, FLAGS.curvature)
        index.add(embedding)

        return index

    if query is None:
        query = emb

    index = build_index(emb)
    D, I = index.search(query, FLAGS.topk)
    I = np.array([[id[col] for col in row] for row in I])

    return I


def main(argv):
    del argv

    embedding, id, golden_knn = generate_sample()
    knn = faiss_knn(embedding, id)

    print("curvfaiss sanity check: {}!".format("passed" if np.array_equal(golden_knn, knn) else "failed"))


if __name__ == "__main__":
    app.run(main)