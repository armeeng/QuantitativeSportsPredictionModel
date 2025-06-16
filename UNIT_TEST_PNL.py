import unittest
import numpy as np
import pandas as pd
import io
from unittest.mock import patch
from TestModel import TestModel  # adjust import as needed

class TestCalculateProfit(unittest.TestCase):
    def test_positive_odds(self):
        odds = np.array([150, 200])
        profit = TestModel._calculate_profit(odds, bet_amount=2.0)
        expected = np.array([(150/100)*2.0, (200/100)*2.0])
        np.testing.assert_allclose(profit, expected, rtol=1e-6)

    def test_negative_odds(self):
        odds = np.array([-150, -200])
        profit = TestModel._calculate_profit(odds, bet_amount=2.0)
        expected = np.array([(100/150)*2.0, (100/200)*2.0])
        np.testing.assert_allclose(profit, expected, rtol=1e-6)

    def test_mixed_and_zero_odds(self):
        odds = np.array([150, -200, 0])
        profit = TestModel._calculate_profit(odds, bet_amount=1.0)
        expected = np.array([1.5, 0.5, 0.0])
        np.testing.assert_allclose(profit, expected, rtol=1e-6)

    def test_nan_and_non_numeric(self):
        odds = np.array(['a', np.nan])
        profit = TestModel._calculate_profit(odds, bet_amount=1.0)
        np.testing.assert_allclose(profit, [0.0, 0.0], rtol=1e-6)

class TestGetOutcomes(unittest.TestCase):
    def test_wrong_shape_raises(self):
        predictions = np.zeros((1,2))
        y_test = np.array([5])  # wrong shape
        odds = pd.DataFrame({'team1_spread': [0], 'total_score': [0]})
        model = TestModel(predictions, y_test, odds)
        with self.assertRaises(ValueError):
            model._get_outcomes()

    def test_regressor_outcomes_correct(self):
        preds = np.array([[10,5], [3,7]])
        y_test = np.array([[10,5], [3,7]])
        odds = pd.DataFrame({
            'team1_spread': [0, 0], 'total_score': [13, 13]
        })
        model = TestModel(preds, y_test, odds)
        outcomes = model._get_outcomes()
        np.testing.assert_array_equal(outcomes['actual_winner_is_t1'], [True, False])
        np.testing.assert_array_equal(outcomes['pred_winner_is_t1'], [True, False])
        assert not outcomes['spread_pushes'].any()
        assert not outcomes['ou_pushes'].any()

class TestAccuracyAndPnlRegressor(unittest.TestCase):
    def setUp(self):
        self.predictions = np.array([[10,5], [3,7]])
        self.y_test = np.array([[10,5], [3,7]])
        self.test_odds = pd.DataFrame({
            'team1_ml': [100, -120], 'team2_ml': [-110, 105],
            'team1_spread': [-3, -3], 'total_score': [13, 13],
            'team1_spread_odds': [100, 100], 'team2_spread_odds': [100, 100],
            'over_odds': [100, 100], 'under_odds': [100, 100]
        })
        self.model = TestModel(self.predictions, self.y_test, self.test_odds)

    def test_calculate_accuracy_all_correct(self):
        acc = self.model.calculate_accuracy()
        self.assertEqual(acc['win_accuracy'], 1.0)
        self.assertEqual(acc['spread_accuracy'], 1.0)
        self.assertEqual(acc['ou_accuracy'], 1.0)

    def test_calculate_pnl_of_all_games(self):
        pnl = self.model.calculate_pnl_of_all_games(bet_amount=1.0)
        self.assertAlmostEqual(pnl['moneyline_pnl'], 2.05, places=6)
        self.assertAlmostEqual(pnl['spread_pnl'], 2.0, places=6)
        self.assertAlmostEqual(pnl['ou_pnl'], 2.0, places=6)

    def test_ev_threshold_regressor_returns_nan(self):
        ev = self.model.calculate_pnl_of_game_above_ev_threshold(ev_threshold=0, bet_amount=1.0)
        for key in ['moneyline', 'spread', 'ou']:
            self.assertTrue(np.isnan(ev[key]['pnl']))
            self.assertEqual(ev[key]['count'], 0)

class TestPnLThresholdClassifier(unittest.TestCase):
    def setUp(self):
        self.predictions = {
            'win': np.array([[0.4, 0.6], [0.6, 0.4]]),
            'spread': np.array([[0.5, 0.5], [0.5, 0.5]]),
            'over': np.array([[0.7, 0.3], [0.2, 0.8]])
        }
        self.y_test = np.array([[10,5], [5,10]])
        self.test_odds = pd.DataFrame({
            'team1_ml': [100, 100], 'team2_ml': [100, 100],
            'team1_spread': [0, 0], 'total_score': [15, 15],
            'team1_spread_odds': [100, 100], 'team2_spread_odds': [100, 100],
            'over_odds': [100, 100], 'under_odds': [100, 100]
        })
        self.model = TestModel(self.predictions, self.y_test, self.test_odds)

    def test_pnl_threshold_default(self):
        ev = self.model.calculate_pnl_of_game_above_ev_threshold(ev_threshold=0, bet_amount=1.0)
        for key in ['moneyline', 'spread', 'ou']:
            self.assertIn(key, ev)
            self.assertIsInstance(ev[key]['pnl'], float)
            self.assertIsInstance(ev[key]['count'], (int, np.integer))

    def test_pnl_threshold_high(self):
        ev = self.model.calculate_pnl_of_game_above_ev_threshold(ev_threshold=1e6, bet_amount=1.0)
        for key in ['moneyline', 'spread', 'ou']:
            self.assertEqual(ev[key]['count'], 0)
            self.assertEqual(ev[key]['pnl'], 0)

class TestPushHandling(unittest.TestCase):
    def test_spread_push_excluded(self):
        preds = np.array([[10,10], [5,0]])  # first game margin zero -> push
        y_test = np.array([[10,10], [5,0]])
        test_odds = pd.DataFrame({
            'team1_ml': [100,100], 'team2_ml': [100,100],
            'team1_spread': [0,0], 'total_score': [20,5],
            'team1_spread_odds': [100,100], 'team2_spread_odds': [100,100],
            'over_odds': [100,100], 'under_odds': [100,100]
        })
        model = TestModel(preds, y_test, test_odds)
        acc = model.calculate_accuracy()
        self.assertEqual(acc['spread_accuracy'], 1.0)

    def test_ou_push_excluded(self):
        preds = np.array([[10,5], [7,8]])  # first game non-push, second push
        y_test = np.array([[10,5], [8,7]])
        test_odds = pd.DataFrame({
            'team1_ml': [100,100], 'team2_ml': [100,100],
            'team1_spread': [0,0], 'total_score': [14,15],
            'team1_spread_odds': [100,100], 'team2_spread_odds': [100,100],
            'over_odds': [100,100], 'under_odds': [100,100]
        })
        model = TestModel(preds, y_test, test_odds)
        acc = model.calculate_accuracy()
        # OU accuracy computed only on game1
        self.assertEqual(acc['ou_accuracy'], 1.0)

class TestDisplayResults(unittest.TestCase):
    def setUp(self):
        self.predictions = np.array([[10,5]])
        self.y_test = np.array([[10,5]])
        self.test_odds = pd.DataFrame({
            'team1_ml': [100], 'team2_ml': [100],
            'team1_spread': [0], 'total_score': [15],
            'team1_spread_odds': [100], 'team2_spread_odds': [100],
            'over_odds': [100], 'under_odds': [100]
        })
        self.model = TestModel(self.predictions, self.y_test, self.test_odds)

    def test_display_results_output(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            self.model.display_results()
            output = fake_out.getvalue()
            self.assertIn("Betting Accuracies", output)
            self.assertIn("Profit & Loss", output)

if __name__ == '__main__':
    unittest.main()
