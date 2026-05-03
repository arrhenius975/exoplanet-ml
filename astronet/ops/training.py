# Copyright 2018 The Exoplanet ML Authors.
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

"""Functions for training an AstroNet model."""

import functools
import logging

import tensorflow as tf


def _polynomial_decay_schedule(initial_value, decay_steps, end_factor, power):
  """Creates a polynomial decay learning rate schedule.
  
  Args:
    initial_value: Initial learning rate.
    decay_steps: Number of steps to decay over.
    end_factor: Factor to multiply initial_value by at end of decay.
    power: Power of the polynomial decay.
    
  Returns:
    A tf.keras.optimizers.schedules.LearningRateSchedule object.
  """=None):
  """Creates the learning rate and weight decay, both with the same schedule.

  Args:
    hparams: ConfigDict containing the learning rate and weight decay
      configurations.
    global_step: Optional global step Tensor (for backwards compatibility,
      not used in TF2 optimizers).

  Returns:
    (learning_rate, weight_decay): The learning rate (float or schedule) and 
    weight decay (float or None). If a decay schedule is configured, returns
    a schedule object; otherwise returns a float.
  """
  learning_rate = hparams.learning_rate
  weight_decay = hparams.get("weight_decay", 0.0)

  if hparams.get("learning_rate_decay_steps"):
    lr_schedule = _polynomial_decay_schedule(
        initial_value=learning_rate,
        decay_steps=hparams.learning_rate_decay_steps,
        end_factor=hparams.learning_rate_end_factor,
        power=hparams.learning_rate_decay_power)
    learning_rate = lr_schedule
    # For weight decay, we also apply the schedule if it exists
    if weight_decay:
      wd_schedule = _polynomial_decay_schedule(
          initial_value=weight_decay,
          decay_steps=hparams.learning_rate_decay_steps,
          end_factor=hparams.learning_rate_end_factor,
          power=hparams.learning_rate_decay_power)
      weight_decay = wd_schedule
          _polynomial_decay,
        global_step=global_step,
        decay_steps=hparams.learning_rate_decay_steps,
        end_factor=hparams.learning_rate_end_factor,
        power=hparams.learning_rate_decay_power)
    learning_rate = decay_schedule(learning_rate)
    if weight_decay:
      weight_decay = decay_schedule(weight_decay)

  return learning_rate, weight_decay


def create_optimizer(hparams, global_step=None, use_tpu=False):
  """Creates a TensorFlow Optimizer.

  Args:
    hparams: ConfigDict containing the optimizer configuration.
    global_step: Optional global step Tensor (for backwards compatibility,
      not used in TF2 optimizers).
    use_tpu: If True, the returned optimizer is wrapped in a
      CrossShardOptimizer (for backwards compatibility).

  Returns:
    A tf.keras.optimizers.Optimizer instance.

  Raises:
    ValueError: If hparams.optimizer is unrecognized.
  """
  optimizer_name = hparams.optimizer.lower()
  
  # Get learning rate and weight decay configurations
  learning_rate, weight_decay = create_learning_rate_and_weight_decay(
      hparams, global_step)
  
  # Convert weight decay schedule to initial value if needed (Keras optimizers
  # may not support schedule objects for weight_decay in all cases)
  if hasattr(weight_decay, '__call__') and not isinstance(
      weight_decay, tf.keras.optimizers.schedules.LearningRateSchedule):
    # It's a schedule object, extract initial value for now
    weight_decay_value = float(hparams.get("weight_decay", 0.0))
  else:
    weight_decay_value = weight_decay if isinstance(weight_decay, (int, float)) else 0.0
  
  # Create optimizer with appropriate parameters
  if optimizer_name == "momentum":
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=hparams.get("momentum", 0.9),
        nesterov=hparams.get("use_nesterov", False),
        weight_decay=weight_decay_value,
        name="MomentumOptimizer")
  elif optimizer_name == "sgd":
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        weight_decay=weight_decay_value,
        name="SGD")
  elif optimizer_name == "adagrad":
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate,
        weight_decay=weight_decay_value,
        name="Adagrad")
  elif optimizer_name == "adam":
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=weight_decay_value,
        name="Adam")
  elif optimizer_name == "rmsprop":
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        weight_decay=weight_decay_value,
        name="RMSprop")
  else:
    raise ValueError("Unknown optimizer: {}".format(hparams.optimizer))

  # Wrap for TPU if needed (for backwards compatibility)
  if use_tpu:
    try:
      optimizer = tf.distribute.experimental.TPUShardingOptimizer(optimizer)
    except AttributeError:
      # Fall back to contrib version if available
      try:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
      except AttributeError:
        logging.warning("TPU optimizers not available, using regular optimizer")

  return optimizer


def create_train_op(model, optimizer):
  """Creates a Tensor to train the model.

  Args:
    model: Instance of AstroModel.
    optimizer: Instance of tf.keras.optimizers.Optimizer.

  Returns:
    A Tensor that runs a single training step and returns model.total_loss.
  """
  # Compute gradients
  with tf.GradientTape() as tape:
    loss = model.total_loss
  
  # Get trainable variables
  trainable_vars = model.trainable_variables if hasattr(model, 'trainable_variables') else tf.trainable_variables()
  
  # Compute gradients
  gradients = tape.gradient(loss, trainable_vars)
  
  # Maybe clip gradient norms
  if model.hparams.get("clip_gradient_norm"):
    clip_norm = model.hparams.clip_gradient_norm
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
  
  # Apply gradients
  optimizer.apply_gradients(zip(gradients, trainable_vars))
  
  # Increment global step manually if needed
  if hasattr(model, 'global_step') and hasattr(model.global_step, 'assign_add'):
    model.global_step.assign_add(1)
  
  return loss
