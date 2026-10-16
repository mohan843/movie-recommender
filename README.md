# Movie Recommender

A small, explainable, content-based movie recommendation project. It ranks movies by cosine similarity over their genres and descriptive tags, using a smoothed TF-IDF-style representation. It uses only Python's standard library, so it runs offline without API keys or model downloads.

## Features

- Search for titles case-insensitively and recommend the closest matches.
- Human-readable sample catalog in `data/movies.csv`.
- Deterministic ordering and similarity scores from 0 to 1.
- Command-line interface and unit tests.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/mohan843/movie-recommender.git
cd movie-recommender
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

No external packages are needed; `requirements.txt` documents this.

## Usage

Run the default example:

```bash
python movie_recommender.py
```

Choose a seed title and number of results:

```bash
python movie_recommender.py "Arrival" --limit 3
```

Use a different CSV file with columns `title,genres,tags,year`:

```bash
python movie_recommender.py "The Matrix" --data data/movies.csv --limit 5
```

The Python API is also available:

```python
from movie_recommender import MovieRecommender, load_movies

movies = load_movies("data/movies.csv")
model = MovieRecommender(movies)
for movie, score in model.recommend("The Matrix", limit=3):
    print(movie["title"], round(score, 3))
```

## How it works

Each title's genres and tags are tokenized, weighted with a smoothed TF-IDF-style formula, and L2-normalized. The dot product of normalized vectors gives cosine similarity. Recommendations exclude the seed title and sort by descending similarity, then title for stable ties. This is a content-based baseline: it does not use personal ratings or collaborative filtering.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project files

- `movie_recommender.py`: recommendation engine and CLI.
- `data/movies.csv`: sample catalog.
- `tests/test_movie_recommender.py`: behavior and edge-case tests.
- `requirements.txt`: dependency declaration (standard library only).
