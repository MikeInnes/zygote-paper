# Variational Quantum Circuit

This folder contains a **proof of concept** demonstration of getting the gradients of rotations from the classical simulation of variational quantum circuits.

## Related Work

[Pennylane: The qubit rotation example](https://pennylane.readthedocs.io/en/latest/tutorials/qubit_rotation.html)

## Introduction

Varitional quantum circuits are one important research topic in the field of quantum computation which requires one to calculate the gradients
numerically very often, in classical simiulation, although the AD (automatic differentiation) in theory is the same as other numerical programs, however
it requires **complex value** AD and some mutation support to save the memory, which most AD package/framework haven't considered seriously.

Moreover, in the framework for quantum computation [Yao.jl](https://github.com/QuantumBFS/Yao.jl), a lot custom data structure (special matrices like general permutation matrices, special type for quantum circuits) are defined for convenience and performance, which makes it almost impossible to directly use most operator overloaded based AD: the framework itself is not able to infer the gradient through custom data types and will result in a large modification to insert tracked types.

As a **proof of concept** example, we demonstrate how Zygote successfully infers the gradient of two rotation block on **X**, **Y** direction so we could optimize these two parameter to make the quantum circuit produce a target quantum state.

To check out about the concept related to quantum computation with Yao, please read the documentation of [Yao](https://quantumbfs.github.io/Yao.jl/latest/).

## Run the script

All the dependencies are recorded in `Project.toml` and `Manifest.toml`, run the `demo.jl` directly with

```
julia --project=variational_quantum_circuit demo.jl
```