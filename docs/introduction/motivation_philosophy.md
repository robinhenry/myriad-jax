# Motivation & Philosophy

## The Scientific Challenge
Scientific systems — from gene circuits to chemical reactors — present a unique set of obstacles that make standard analysis difficult:

* **Inherent stochasticity:** dynamics are often driven by random (sometimes very rare) events (e.g., molecular noise), requiring thousands or millions of samples to characterize properly.

* **Parameter uncertainty:** physical constants (like reaction rates) are often unknown or vary across the population.

* **Time bottlenecks:** biology and chemistry are often slow. Gathering sufficient data sequentially or with a few samples in parallel is often impossible within a reasonable timeframe.

## The Opportunity: High-Throughput Control

Modern experimental hardware has evolved to meet these challenges. For example, platforms like [microfluidic mother machines](https://www.youtube.com/watch?v=yrJzMW5jcbM) now allow us to observe and interact in real-time with 100,000+ or even 1M+ of cells simultaneously and independently.

**Myriad is built to explore and leverage this experimental paradigm.**

We need algorithms that can act on this massive, parallel data stream in real-time. Myriad provides the computational backend to develop and test these algorithms before deploying them to real hardware, enabling scientists to:

1. **Iterate quickly:** want to test a new algorithm, experimental setup, or system? Try it out *in-silico* in minutes, by-passing days/weeks/months of expensive hardware and experimental time on a potential dead-end.

2.  **Accelerate discovery:** when sim-to-real transfer is viable, train policies on 100k+ or millions of simulated variants and deploy them directly to hardware, potentially saving days or weeks of data gathering required for training.

3.  **Standardize & reproduce experiments:** reproducing Myriad experiments *exactly* is as simple as providing the Myriad version number you used and the experiment configuration file (automatically saved).

## The Myriad Philosophy

Our goal is not to replace physical experiments. On the contrary, we strongly believe something only "works" once it's been demonstrated in the lab.

However, we think we can make the whole process more efficient by filtering out bad ideas and validating good ones computationally first.

And we'd like to make it super easy, so that many people can join in on the fun.

With this in mind, here are a few principles we'd like to adhere to as we keep developing Myriad.

#### 1. Optimize for real-world experimental time (& embrace parallelism)

The rate of scientific progress is constrained by how long it takes to test and validate new ideas. As such, we want to develop Myriad with the end goal of minimizing experimental time and facilitating quick iterations.

One way to tackle this time bottleneck is to do more experiments (and therefore gather more data) in parallel. With new massively parallel experimental setups enabling those experiments in the physical world, Myriad should aim to facilitate the design of algorithms that complement these experimental platforms.

#### 2. Don't get stuck on toy problems

Toy problems play a very important role in the development of new methods and algorithms, and they should always be our first port of call in such cases (the first Myriad task we implemented was the classic RL Cartpole problem!).

However, we try to remember that the real world is often much more complex than our toy problems. And if we stop at toy problems, our methods and algorithms are unlikely to get used in the lab.

For this reason, although we recognize this isn't always possible, we aim to focus most of our efforts on the implementation of more rigorous simulations that illustrate the chaos and complexity of physical systems, so that new algorithms can be battle-tested in more realistic conditions.

#### 3. Make it really smooth

Ultimately, we want Myriad to be used by scientists. If one needs a CS degree and extensive knowledge of JAX, python, GPU optimization, etc. to use it, this vision is unlikely to come to life.

Instead, we make Myriad as user-friendly as possible, so that you can focus on what's most important: the science.
