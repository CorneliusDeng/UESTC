**Reference：https://github.com/google-research/tuning_playbook**



### Guide for starting a new project

------

#### Choosing the model architecture

**Summary: When starting a new project, try to reuse a model that already works.**

- Choose a well established, commonly used model architecture to get working first. It is always possible to build a custom model later.
- Model architectures typically have various hyperparameters that determine the model's size and other details (e.g. number of layers, layer width, type of activation function).
- When possible, try to find a paper that tackles something as close as possible to the problem at hand and reproduce that model as a starting point.

#### Choosing the optimizer

**Summary: Start with the most popular optimizer for the type of problem at hand.**

- No optimizer is the "best" across all types of machine learning problems and model architectures.
- Stick with well-established, popular optimizers, especially when starting a new project.
- Be prepared to give attention to all hyperparameters of the chosen optimizer.
  - Optimizers with more hyperparameters may require more tuning effort to find the best configuration.
  - This is particularly relevant in the beginning stages of a project when we are trying to find the best values of various other hyperparameters while treating optimizer hyperparameters as nuisance parameters.
  - It may be preferable to start with a simpler optimizer (e.g. SGD with fixed momentum or Adam with fixed $\epsilon, \beta_1,\beta_2$) in the initial stages of the project and switch to a more general optimizer later.
- Recommend optimizers (but are not limited to):
  - SGD with momentum (the Nesterov variant)
  - Adam and NAdam, which are more general than SGD with momentum. Note that Adam has 4 tunable hyperparameters and they can all matter!

#### Choosing the batch size

**Summary: The batch size governs the training speed and shouldn't be used to directly tune the validation set performance. Often, the ideal batch size will be the largest batch size supported by the available hardware.**

- The batch size is a key factor in determining the training time and computing resource consumption.

- Increasing the batch size will often reduce the training time. This can be highly beneficial because it, e.g.:

  - Allows hyperparameters to be tuned more thoroughly within a fixed time interval, potentially resulting in a better final model.
  - Reduces the latency of the development cycle, allowing new ideas to be tested more frequently.

- Increasing the batch size may either decrease, increase, or not change the resource consumption.

- The batch size should not be treated as a tunable hyperparameter for validation set performance.

  - As long as all hyperparameters are well-tuned (especially the learning rate and regularization hyperparameters) and the number of training steps is sufficient, the same final performance should be attainable using any batch size.

- Determining the feasible batch sizes and estimating training throughput

  - For a given model and optimizer, there will typically be a range of batch sizes supported by the available hardware. The limiting factor is usually accelerator memory.

  - Unfortunately, it can be difficult to calculate which batch sizes will fit in memory without running, or at least compiling, the full training program. The easiest solution is usually to run training jobs at different batch sizes (e.g. increasing powers of 2) for a small number of steps until one of the jobs exceeds the available memory.

  - For each batch size, we should train for long enough to get a reliable estimate of the training throughput 

    training throughput = or, equivalently, the time per step.

    time per step = (batch size) / (training throughput)

  - When the accelerators aren't yet saturated, if the batch size doubles, the training throughput should also double (or at least nearly double). Equivalently, the time per step should be constant (or at least nearly constant) as the batch size increases.

  - If this is not the case then the training pipeline has a bottleneck such as I/O or synchronization between compute nodes. This may be worth diagnosing and correcting before proceeding.

  - If the training throughput increases only up to some maximum batch size, then we should only consider batch sizes up to that maximum batch size, even if a larger batch size is supported by the hardware.

    - All benefits of using a larger batch size assume the training throughput increases. If it doesn't, fix the bottleneck or use the smaller batch size.
    - Gradient accumulation simulates a larger batch size than the hardware can support and therefore does not provide any throughput benefits. It should generally be avoided in applied work.

  - These steps may need to be repeated every time the model or optimizer is changed (e.g. a different model architecture may allow a larger batch size to fit in memory).

- Choosing the batch size to minimize training time

  - Training time = (time per step) x (total number of steps)
  - We can often consider the time per step to be approximately constant for all feasible batch sizes. This is true when there is no overhead from parallel computations and all training bottlenecks have been diagnosed and corrected. In practice, there is usually at least some overhead from increasing the batch size.
  - As the batch size increases, the total number of steps needed to reach a fixed performance goal typically decreases. E.g. Doubling the batch size might halve the total number of steps required. This is called perfect scaling. Perfect scaling holds for all batch sizes up to a critical batch size, beyond which one achieves diminishing returns. Eventually, increasing the batch size no longer reduces the number of training steps (but never increases it).
  - Therefore, the batch size that minimizes training time is usually the largest batch size that still provides a reduction in the number of training steps required.

- Changing the batch size requires re-tuning most hyperparameters

  - The optimal values of most hyperparameters are sensitive to the batch size. Therefore, changing the batch size typically requires starting the tuning process all over again.
  - The hyperparameters that interact most strongly with the batch size, and therefore are most important to tune separately for each batch size, are the optimizer hyperparameters (e.g. learning rate, momentum) and the regularization hyperparameters.
  - Keep this in mind when choosing the batch size at the start of a project. If you need to switch to a different batch size later on, it might be difficult, time consuming, and expensive to re-tune everything for the new batch size.

#### Choosing the initial configuration

- Before beginning hyperparameter tuning we must determine the starting point. This includes specifying (1) the model configuration (e.g. number of layers), (2) the optimizer hyperparameters (e.g. learning rate), and (3) the number of training steps.
- Determining this initial configuration will require some manually configured training runs and trial-and-error.
- Choosing the number of training steps involves balancing the following tension:
  - On the one hand, training for more steps can improve performance and makes hyperparameter tuning easier.
  - On the other hand, training for fewer steps means that each training run is faster and uses fewer resources, boosting tuning efficiency by reducing the time between cycles and allowing more experiments to be run in parallel. Moreover, if an unnecessarily large step budget is chosen initially, it might be hard to change it down the road, e.g. once the learning rate schedule is tuned for that number of steps.



### A scientific approach to improving model performance

------

#### The incremental tuning strategy

#### Exploration vs exploitation

#### Choosing the goal for the next round of experiments

#### Designing the next round of experiments

#### Determining whether to adopt a training pipeline change or hyperparameter configuration

#### After exploration concludes



### Determining the number of steps for each training run

------

#### Deciding how long to train when training is not compute-bound

#### Deciding how long to train when training is compute-bound



### Determining the number of steps for each training run

------

#### Optimizing the input pipeline

#### Evaluating model performance

#### Saving checkpoints and retrospectively selecting the best checkpoint

#### Setting up experiment tracking

#### Batch normalization implementation details

#### Considerations for multi-host pipelines