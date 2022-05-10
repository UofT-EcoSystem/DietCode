# [RFC] DietCode

## Motivation

Auto-scheduler (Ansor) uses code sketch and optimization rules to generate a
large search space. The search space defined by Ansor has shown great
opportunities and therefore the search quality and the search efficiency are
determined by how we search the space.

Ansor utilizes improved cost model and task scheduler to help explore the search
space. The cost model analyzes and finds high-performance code transformations
in the search space and the task scheduler allocates the time budget to
different computation graphs. However, we find serval drawbacks to this
approach:

The accuracy of the cost model determines the search quality, but Ansor uses
monolithic cost model to predict different computation graphs (subgraphs),
resulting in an accuracy loss during tuning.

The task scheduler allocates most of the time budget to subgraphs with most
improving potential (i.e., those with the highest latency). This approach works
well at the beginning of the autotuning. However, as the potential subgraph
gradually reaches its peak performance with adequate time budget, other
subgraphs have little time budget to reach its peak performance.

The search process will at the end take a dozen of hours. This motivates us to
find better way to explore the search space.


## Solution

