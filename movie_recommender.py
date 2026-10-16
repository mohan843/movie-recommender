"""A lightweight, dependency-free content-based movie recommender."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


class MovieRecommender:
    """Recommend movies by cosine similarity over genres and descriptive tags."""

    def __init__(self, movies: Iterable[dict[str, str]]) -> None:
        self.movies = [self._normalize(movie) for movie in movies]
        self.by_title = {movie["title"].casefold(): movie for movie in self.movies}
        if len(self.by_title) != len(self.movies):
            raise ValueError("Movie titles must be unique")
        self.vocab = sorted({token for movie in self.movies for token in self._tokens(movie)})
        self.vectors = {movie["title"]: self._vector(movie) for movie in self.movies}

    @staticmethod
    def _normalize(movie: dict[str, str]) -> dict[str, str]:
        if not movie.get("title", "").strip():
            raise ValueError("Every movie needs a title")
        return {
            "title": movie["title"].strip(),
            "genres": movie.get("genres", "").replace("|", " ").strip(),
            "tags": movie.get("tags", "").strip(),
            "year": movie.get("year", "").strip(),
        }

    @staticmethod
    def _tokens(movie: dict[str, str]) -> list[str]:
        return [token.casefold() for token in (movie["genres"] + " " + movie["tags"]).split() if token]

    def _vector(self, movie: dict[str, str]) -> dict[str, float]:
        tokens = self._tokens(movie)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        n_docs = max(len(self.movies), 1)
        # Smoothed inverse document frequency prevents ubiquitous terms dominating.
        vector = {
            term: (1 + math.log(count)) * (math.log((1 + n_docs) / (1 + sum(term in self._tokens(m) for m in self.movies))) + 1)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {term: value / norm for term, value in vector.items()} if norm else {}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(term, 0.0) for term, value in left.items())

    def recommend(self, title: str, limit: int = 5) -> list[tuple[dict[str, str], float]]:
        """Return up to limit (movie, similarity) pairs, excluding the input movie."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        movie = self.by_title.get(title.casefold())
        if movie is None:
            raise KeyError(f"Unknown movie: {title}")
        candidates = [m for m in self.movies if m["title"] != movie["title"]]
        ranked = [(m, self._cosine(self.vectors[movie["title"]], self.vectors[m["title"]])) for m in candidates]
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]["title"].casefold()))[:limit]


def load_movies(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find similar movies from genres and tags.")
    parser.add_argument("title", nargs="?", default="The Matrix", help="movie title to use as a seed")
    parser.add_argument("--data", default=str(Path(__file__).parent / "data" / "movies.csv"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    recommender = MovieRecommender(load_movies(args.data))
    print(f"Because you liked {args.title}:")
    for movie, score in recommender.recommend(args.title, args.limit):
        year = f" ({movie['year']})" if movie["year"] else ""
        print(f"  {movie['title']}{year}  [similarity: {score:.3f}]")


if __name__ == "__main__":
    main()
