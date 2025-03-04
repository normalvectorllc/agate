# Pokemon Dataset

## Overview

This dataset contains comprehensive information about Pokemon from multiple generations, including their types, stats, abilities, and other attributes. It is used in the AI/ML coding interview assessment to evaluate candidates' data analysis and model building skills.

## Source

The dataset is derived from the [Pokemon Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon/data) on Kaggle by Rounak Banik, which provides a comprehensive collection of Pokemon data.

## Features

The dataset includes the following features:

- **pokedex_number**: The unique identification number of the Pokemon
- **name**: Name of the Pokemon
- **japanese_name**: Japanese name of the Pokemon
- **type1**: Primary type of the Pokemon
- **type2**: Secondary type of the Pokemon (if any)
- **hp**: Base hit points (health)
- **attack**: Base attack stat
- **defense**: Base defense stat
- **sp_attack**: Base special attack stat
- **sp_defense**: Base special defense stat
- **speed**: Base speed stat
- **generation**: Generation the Pokemon was introduced in
- **is_legendary**: Boolean indicating whether the Pokemon is legendary (1) or not (0)
- **height_m**: Height of the Pokemon in meters
- **weight_kg**: Weight of the Pokemon in kilograms
- **abilities**: List of abilities the Pokemon can have
- **against_bug**: Effectiveness multiplier of Bug-type moves against this Pokemon
- **against_dark**: Effectiveness multiplier of Dark-type moves against this Pokemon
- **against_dragon**: Effectiveness multiplier of Dragon-type moves against this Pokemon
- **against_electric**: Effectiveness multiplier of Electric-type moves against this Pokemon
- **against_fairy**: Effectiveness multiplier of Fairy-type moves against this Pokemon
- **against_fight**: Effectiveness multiplier of Fighting-type moves against this Pokemon
- **against_fire**: Effectiveness multiplier of Fire-type moves against this Pokemon
- **against_flying**: Effectiveness multiplier of Flying-type moves against this Pokemon
- **against_ghost**: Effectiveness multiplier of Ghost-type moves against this Pokemon
- **against_grass**: Effectiveness multiplier of Grass-type moves against this Pokemon
- **against_ground**: Effectiveness multiplier of Ground-type moves against this Pokemon
- **against_ice**: Effectiveness multiplier of Ice-type moves against this Pokemon
- **against_normal**: Effectiveness multiplier of Normal-type moves against this Pokemon
- **against_poison**: Effectiveness multiplier of Poison-type moves against this Pokemon
- **against_psychic**: Effectiveness multiplier of Psychic-type moves against this Pokemon
- **against_rock**: Effectiveness multiplier of Rock-type moves against this Pokemon
- **against_steel**: Effectiveness multiplier of Steel-type moves against this Pokemon
- **against_water**: Effectiveness multiplier of Water-type moves against this Pokemon
- **base_egg_steps**: Number of steps required to hatch an egg of this Pokemon
- **base_happiness**: Base happiness value of the Pokemon
- **base_total**: Sum of all the base stats
- **capture_rate**: Catch rate of the Pokemon
- **classfication**: Category of the Pokemon (e.g., "Seed Pokémon", "Flame Pokémon")
- **experience_growth**: The amount of experience points gained
- **percentage_male**: Percentage of the species that are male (some species are genderless)

## Notes

- Some Pokemon may have missing values for certain attributes, which candidates should handle appropriately.
- The dataset includes Pokemon from multiple generations, which may have different stat distributions.
- Legendary Pokemon are rare and typically have higher stats than non-legendary Pokemon.
- The dataset may contain more features than are needed for the assessment, providing candidates with the opportunity to select relevant features for their analysis.