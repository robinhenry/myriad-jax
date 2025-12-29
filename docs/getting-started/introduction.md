# Introduction to Myriad

## The Motivation

### The Challenge
Scientific systems — from gene circuits to chemical reactors — present a unique set of obstacles that make standard analysis difficult:

* **Inherent stochasticity:** dynamics are often driven by random events (e.g., molecular noise), requiring thousands or millions of samples to characterize properly.
* **Parameter uncertainty:** physical constants (like reaction rates) are often unknown or vary across the population.
* **Time bottleneck:** biology and chemistry are often slow. Gathering sufficient data sequentially is often impossible within a reasonable timeframe.

### The Opportunity: High-Throughput Control

Modern experimental hardware has evolved to meet these challenges. For example, platforms like [microfluidic mother machines](https://www.youtube.com/watch?v=yrJzMW5jcbM) now allow us to observe and interact in real-time with 100,000+ or even 1M+ of cells simultaneously.

**Myriad is built to leverage this experimental paradigm.**

We need algorithms that can act on this massive, parallel data stream in real-time. Myriad provides the computational backend to develop and test these algorithms before deploying them to real hardware, enabling you to:

1.  **Test experimental designs in silico:** Iterate on active learning strategies, experimental protocols, and control algorithms in simulation—avoiding wasted lab time on approaches that don't work.
2.  **Validate algorithmic ideas:** Verify that your method works in principle on realistic stochastic systems before committing expensive hardware time.
3.  **Accelerate discovery:** When sim-to-real transfer is viable, train policies on 100k+ simulated variants and deploy them directly to hardware.

The goal is not to replace physical experiments, but to make them more efficient by filtering out bad ideas and validating good ones computationally first.

## Philosophy

Our goal is provide the scientific community with the tools to run in-silico simulations and experiments that accelerate discovery in the physical lab. For this reason, we try to embrace the messy reality of the lab by prioritizing:

* **Rigorous stochasticity:** we aim to model exact stochasticity (e.g., Gillespie/SSA) and population heterogeneity because, in the lab, noise is the signal.
* **Glass-box dynamics:** we expose the underlying differential equations, enabling gradient-based analysis, physics-informed learning, and intelligent experiment design when useful.
* **Lab relevance:** we target problems where simulation results are intended to transfer to real experimental hardware.
