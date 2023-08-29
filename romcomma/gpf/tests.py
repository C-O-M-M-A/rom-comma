#  BSD 3-Clause License.
# 
#  Copyright (c) 2019-2023 Robert A. Milton. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
# 
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Contains tests of the gpf package."""


from romcomma.gpf import base, kernels, likelihoods, models
import numpy as np
import gpflow as gf
import tensorflow as tf
from romcomma.user import contexts

def covariance():
    a = np.array([[0.9, -0.5], [-0.5, 0.75]])
    b = base.Variance(a)
    print(b.value)
    print(b.value)
    b._cholesky_diagonal.assign([1.0, 1.0])
    print(b.value)
    print(b.value)


def regression_data():
    data = np.linspace(start=1, stop=50, num=50).reshape(5, 10).transpose()
    return data[:, :3], data[:, 3:]


def kernel():
    lengthscales = [0.01 * np.ones(3), 0.03 * np.ones(3)]
    variance = 0.5 * np.eye(2)
    return kernels.RBF(variance, lengthscales)


def likelihood():
    variance = 0.0001 * np.eye(2)
    return likelihoods.MOGaussian(variance)


@tf.function
def increment(x: tf.Tensor) -> tf.Tensor:
    x = x + tf.constant(1.0)
    return {'x': x}

if __name__ == '__main__':
    with contexts.Environment('Test', float='float64'):
        lh = likelihood()
        X, Y = regression_data()
        print(X)
        print(Y)
        gp = models.MOGPR((X, Y), kernel(), noise_variance=lh.variance.value)
        results = gp.predict_f(X, full_cov=False, full_output_cov=False)
        print(results)
        results = gp.predict_y(X, full_cov=False, full_output_cov=False)
        print(results)
        results = gp.log_marginal_likelihood()
        print(results)
        gp.kernel.is_lengthscales_trainable = True
        opt = gf.optimizers.Scipy()
        opt.minimize(closure=gp.training_loss, variables=gp.trainable_variables)
        results = gp.predict_y(X, full_cov=False, full_output_cov=False)
        print(gp.log_marginal_likelihood())
        print(gp.kernel.variance.value)
        print(gp.likelihood.variance.value)
        print(results)

