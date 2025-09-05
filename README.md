# Extended Coupled Cluster approach to Twisted Graphene Layers

## Introduction

This is a simulation code written as part of my masters project for developing the extended coupled cluster framework and applying it to characterise monolayer and twisted bilayer graphene via the continuum model. This project was built using Python 3.13. The associated pre-print is [available on arXiv](https://doi.org/10.48550/arXiv.2405.10863).

## Funcionality

The current funcionality includes:

- Extended coupled cluster implementation, up to doubles truncation with CUDA support,
- Application to monolayer and twisted bilayer graphene systems with varying fillings,
- Calculation of the energy, band structure and superconducting gap,
- Added singular value decomposition for optimised calculations.

## Structure

There are two main programs, built on the same codebase:

- The monolayer program in ```src/monolayer/``` which contains code for simulating simple monolayer graphene,
- The bilayer program in ```src/bilayer/``` which contains the CUDA supported simulation code of twister bilayer graphene.

## Changelog

This program has had the following releases:


### 1.0 Stable - 5th of September 2025

- Initial public release of the software.