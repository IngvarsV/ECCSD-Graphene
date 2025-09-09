# Extended Coupled Cluster approach to Twisted Graphene Layers

## Introduction

This is a simulation code written as part of my masters project for developing the extended coupled cluster framework and applying it to characterise monolayer and twisted bilayer graphene via the continuum model. This project was built using Python 3.12 and PyTorch 2.2.1 versions.

## Funcionality

The current funcionality includes:

- Extended coupled cluster implementation, up to doubles truncation with CUDA support,
- Application to monolayer and twisted bilayer graphene systems with varying fillings,
- Calculation of the energy, band structure and superconducting gap,
- Added singular value decomposition for optimised calculations.

## Structure

There are two main programs, built on the same codebase:

- The monolayer program in ```src/monolayer/``` which contains code for simulating simple monolayer graphene,
- The bilayer program in ```src/bilayer/``` which contains the CUDA supported simulation code of twisted bilayer graphene.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
