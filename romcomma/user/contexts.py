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

""" **Context managers** """

from __future__ import annotations

from romcomma.base.definitions import *
from time import time
from datetime import timedelta
from contextlib import contextmanager


@contextmanager
def Timer(name: str = '', is_inline: bool = True):
    """ Context Manager for timing operations.

    Args:
        name: The name of this context, ``print``ed as what is being timed. The (default) empty string will not be timed.
        is_inline: Whether to report timing inline (the default), or with linebreaks to top and tail a paragraph.
    """
    _enter = time()
    if name != '':
        if is_inline:
            print(f'Running {name}', end='', flush=True)
        else:
            print(f'Running {name}...')
    yield
    if name != '':
        _exit = time()
        if is_inline:
            print(f' took {timedelta(seconds=int(_exit-_enter))}.')
        else:
            print(f'...took {timedelta(seconds=int(_exit-_enter))}.')


@contextmanager
def Environment(name: str = '', device: str = '', **kwargs):
    """ Context Manager setting up the environment to run operations.

    Args:
        name: The name of this context, ``print``ed as what is being run. The (default) empty string will not be timed.
        device: The device to run on. If this ends in the regex ``[C,G]PU*`` then the logical device ``/[C,G]PU*`` is used,
            otherwise device allocation is automatic.
        **kwargs: Is passed straight to the implementation GPFlow manager. Note, however, that ``float=float32`` is inoperative due to SciPy.
            ``eager=bool`` is passed to `tf.config.run_functions_eagerly <https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly>`_.
    """
    with Timer(name):
        kwargs = kwargs | {'float': 'float64'}
        eager = kwargs.pop('eager', None)
        tf.config.run_functions_eagerly(eager)
        print(' using GPFlow(' + ', '.join([f'{k}={v!r}' for k, v in kwargs.items()]), end=')')
        device = '/' + device[max(device.rfind('CPU'), device.rfind('GPU')):]
        if len(device) > 3:
            device_manager = tf.device(device)
            print(f' on {device}', end='')
        else:
            device_manager = Timer()
        implementation_manager = gf.config.as_context(gf.config.Config(**kwargs))
        print('...')
        with device_manager:
            with implementation_manager:
                yield
        print('...Running ' + name, end='')
