# Myriad

A JAX-native platform for massively parallel system identification and control.

Myriad is built around a simple idea: if you can simulate thousands of experiments in parallel on a GPU, you can learn a lot faster about uncertain, stochastic systems. It combines system identification, control, and population-level learning in a single experimental framework.

```{warning}
Myriad is in early active development. APIs will change, documentation has large gaps, and some features are still taking shape. Things will improve over time — contributions and feedback are welcome!
```

Check out [Motivation & Philosophy](introduction/motivation_philosophy.md) to learn more about Myriad, the [tutorials](tutorials/basics/index) to get started, or head to the [GitHub repo](https://github.com/robinhenry/myriad-jax) for the source code and latest updates.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Introduction

introduction/motivation_philosophy
introduction/installation
introduction/quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

user_guide/core_concepts
user_guide/basic_usage
user_guide/running_experiments
user_guide/tips
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Advanced

advanced/custom_task
advanced/auto_tuning
advanced/benchmarking
```


```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api/overview
api/envs/index
api/agents/index
api/platform
api/core
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tutorials

tutorials/basics/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Development

Github <https://github.com/robinhenry/myriad-jax>
development/guidelines
```
