import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movie_recommender import MovieRecommender, load_movies


class MovieRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies = load_movies(Path(__file__).resolve().parents[1] / "data" / "movies.csv")
        cls.model = MovieRecommender(cls.movies)

    def test_sample_data_loads(self):
        self.assertEqual(len(self.movies), 10)

    def test_recommendations_are_ranked_and_exclude_seed(self):
        results = self.model.recommend("The Matrix", 4)
        self.assertEqual(len(results), 4)
        self.assertNotIn("The Matrix", [movie["title"] for movie, _ in results])
        self.assertGreaterEqual(results[0][1], results[-1][1])
        self.assertEqual(results[0][0]["title"], "Blade Runner 2049")

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(self.model.recommend("the matrix", 1)[0][0]["title"], "Blade Runner 2049")

    def test_unknown_seed_raises(self):
        with self.assertRaises(KeyError):
            self.model.recommend("Missing Movie")

    def test_negative_limit_raises(self):
        with self.assertRaises(ValueError):
            self.model.recommend("The Matrix", -1)

    def testlzero_limit_returns_empty(self):
        self.assertEqual(self.model.recommend("The Matrix", 0), [])

    def test_duplicate_titles_are_rejected(self):
        with self.assertRaises(ValueError):
            MovieRecommender([{"title": "Same"}, {"title": "Same"}])


if __name__ == "__main__":
    unittest.main()
