import numpy as np

SEED = 2137
GENOME_LENGTH = 96
POPULATION_SIZE = 50
GENERATIONS = 500
MUTATION_RATE = 0.02 # Do 5%
NO_PROGRESS_LIMIT = 10
RESULTS_DIR = "venv/results"


rng = np.random.default_rng(SEED)