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

""" Create a Repository from a csv file or a DataFrame, optionally adding Gaussian noise to the output """

from __future__ import annotations

from romcomma.base.definitions import *
from romcomma.run import context, sample
import romcomma.data


ROOT: Path = Path('C:/Users/fc1ram/Documents/Research/dat/SoftwareTest/root')     #: The root folder to house all data repositories.
CSV: Path = Path('C:/Users/fc1ram/Documents/Research/dat/SoftwareTest/2.0/all.M.7.c.v.25.00.N.400.0/data.csv')      #: The csv file to read.
K: int = 5   #: The number of Folds in the new repository.
ADD_NOISE: bool = False     #: Whether to add Gaussian Noise to the outputs being read.
NOISE_MAGNITUDES: Tuple[float] = (0.25,)   #: The noise-to-signal ratio, which is equal to the StdDev of the noise added to the normalised function output.
IS_NOISE_COVARIANT: Tuple[bool] = (False,)   #: Whether the Gaussian noise applied to the outputs is statistically independent between outputs.
IS_NOISE_VARIANCE_RANDOM: Tuple[bool] = (False,)    #: Whether the noise variance is stochastic or fixed.

if __name__ == '__main__':
    with context.Environment('Test', device='CPU'):
        repo = romcomma.data.storage.Repository.from_csv(ROOT / 'repository', CSV).into_K_folds(K, shuffle_before_folding=False)
        if ADD_NOISE:
            ext = 0
            for noise_magnitude in NOISE_MAGNITUDES:
                for is_noise_covariant in IS_NOISE_COVARIANT:
                    for is_noise_variance_random in IS_NOISE_VARIANCE_RANDOM:
                        noisy_repo = romcomma.data.storage.Repository.from_df(repo.folder.with_suffix(f'.{ext}'), repo.data.df)
                        noise_variance = sample.GaussianNoise.Variance(repo.L, noise_magnitude, is_noise_covariant, is_noise_variance_random)
                        with context.Timer(f'noise={noise_magnitude} is_noise_covariant={is_noise_covariant} ' +
                                           f'is_noise_variance_random={is_noise_variance_random} ext={ext} ... ', is_inline=True):
                            noise = sample.GaussianNoise(repo.N, noise_variance())
                            noise(repo)
                        ext += 1

