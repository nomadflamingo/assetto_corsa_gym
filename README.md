<h1>Assetto Corsa Gym</span></h1>

This repository provides the version of [Assetto Corsa Gym](https://github.com/dasGringuen/assetto_corsa_gym), extended with fuel consumption metrics. This code was used for the experiments for a Bachelor's thesis conducted at the <i>National University of Kyiv-Mohyla Academy</i>.

## Reward Function Modification

The reward function has been updated to penalize fuel consumption. It uses the immediate fuel change since the previous step, calculated as `fuel_diff`. The new reward is computed as:
```
r_new = r_old - fuel_diff * penalize_fuel_consumption_coef
```

The `penalize_fuel_consumption_coef` is a configurable coefficient defined in the config file. In our experiments, we adjusted this coefficient to reduce the average reward by 2%, 5%, 10%, and 20%.

This behavior can be toggled via the `AssettoCorsa.penalize_fuel_consumption` flag in the config file.

## Added Metrics

The following metrics have been added in the simulation `summary.csv` file:
- `BestLapFuel`: Fuel used on the lap with the lowest fuel consumption
- `LapNo_i_Fuel`: Fuel consumed during lap `i` of the episode

The following metrics have been added in the simulation `.csv` output files for each episode:
- `fuel`: Remaining fuel in the car at the current simulation step
- `fuel_diff`: Change in fuel level since the previous step