# Genetic Algorithm (GA) Application: Playlist & Timetable Generator

## Overview

This project implements a Genetic Algorithm (GA) to solve two problems:

1.  Musical Playlist Generation
2.  Musical Timetable Generation (day/slot based)

The application allows experimenting with different GA operators (selection, crossover, mutation) and provides weighted fitness functions for playlist and timetable optimization.

## Features

- Choice of selection operators: roulette, tournament, rank
- Choice of crossover operators: one-point, two-point, uniform, order1 (OX for permutations)
- Choice of mutation operators: swap, scramble, inversion, replace
- Two fitness functions (playlist and timetable) that balance hard and soft constraints using weighted terms
- Command-line interface (CLI) to select mode, operators, and GA parameters
- Includes a synthetic demo dataset of songs (can be replaced with your own CSV)

## Usage

Basic example: Generate a 20-song playlist

```bash
python main.py --mode playlist --pop-size 120 --gens 150 \
  --selection tournament --crossover order1 --mutation swap \
  --playlist-size 20

Weekly timetable generation (5 days × 4 slots)

```bash
python main.py --mode timetable --days 5 --slots-per-day 4 \--selection rank --crossover order1 \--mutation inversion

```bash
Show available options

python main.py --help

## Data

- The project includes a small synthetic SONGS list for demo purposes.
- You can replace this with your own dataset (CSV) if desired.

## Author

(Write your name and roll number here)
`````
