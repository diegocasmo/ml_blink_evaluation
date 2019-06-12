# ml_blink

Code for evaluating the ML-Blink algorithm in isolation

## Usage
  - Install [pipenv](https://docs.pipenv.org/en/latest/) by running:
``` bash
  brew install pipenv
```
  - Run `pipenv install` to install dependencies
  - Unzip the beta-pack dataset images (`PanSTARRS_ltd` and `USNO1001`) in `/images` in their corresponding directories
  - Run `pipenv shell` to activate the environment
  - Run evaluation script (replace `<num_projections>` and `<num_time_stpes>` as desired)
``` bash
  python ml_blink.py <num_projections> <num_time_steps>
```
