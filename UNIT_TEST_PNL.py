import unittest
import numpy as np

# You must save your main class in a file named `ml_model.py` for this import to work
from Model import MLModel 

class TestPnlCalculation(unittest.TestCase):
    """
    Unit tests for the static method _calculate_pnl in the MLModel class.
    
    These tests verify the core logic for calculating profit and loss from
    American odds across various betting scenarios.
    """

    def test_single_win_positive_odds(self):
        """Test a single winning bet on an underdog (+150)."""
        # We bet on Team 1 (pick_condition = True)
        # Team 1 wins by 10 (actual_vs_line = 10, which is > 0), so the bet wins.
        # Odds are +150, so a $1 bet should profit $1.50.
        pnl = MLModel._calculate_pnl(
            pick_condition=np.array([True]),
            actual_vs_line=np.array([10]),
            odds=np.array([150])
        )
        self.assertAlmostEqual(pnl, 1.5)

    def test_single_win_negative_odds(self):
        """Test a single winning bet on a favorite (-200)."""
        # We bet on Team 1 (pick_condition = True)
        # Team 1 wins by 5 (actual_vs_line = 5, which is > 0), so the bet wins.
        # Odds are -200, so a $1 bet should profit $0.50 (risk $2 to win $1).
        pnl = MLModel._calculate_pnl(
            pick_condition=np.array([True]),
            actual_vs_line=np.array([5]),
            odds=np.array([-200])
        )
        self.assertAlmostEqual(pnl, 0.5)

    def test_single_loss(self):
        """Test a single losing bet."""
        # We bet on Team 1 (pick_condition = True)
        # Team 1 loses by 3 (actual_vs_line = -3, which is < 0), so the bet loses.
        # Any loss is -1 unit.
        pnl = MLModel._calculate_pnl(
            pick_condition=np.array([True]),
            actual_vs_line=np.array([-3]),
            odds=np.array([-110])
        )
        self.assertAlmostEqual(pnl, -1.0)

    def test_single_push(self):
        """Test a bet that results in a push."""
        # The actual outcome vs the line is exactly 0. This is a push.
        # P/L for a push should be 0.
        pnl = MLModel._calculate_pnl(
            pick_condition=np.array([True]),
            actual_vs_line=np.array([0]),
            odds=np.array([-110])
        )
        self.assertAlmostEqual(pnl, 0.0)

    def test_ignore_invalid_odds(self):
        """Test that bets with invalid odds (0 or nan) are ignored."""
        # Game 1 is a win, but odds are 0, so it shouldn't be counted.
        # Game 2 is a win, but odds are nan, so it also shouldn't be counted.
        # Total P/L should be 0.
        pnl = MLModel._calculate_pnl(
            pick_condition=np.array([True, True]),
            actual_vs_line=np.array([10, 5]),
            odds=np.array([0, np.nan])
        )
        self.assertAlmostEqual(pnl, 0.0)
        
    def test_betting_against_team1(self):
        """Test betting on Team 2 (pick_condition = False)."""
        # We bet on Team 2 (pick_condition = False)
        # Team 1 loses by 7 (actual_vs_line = -7, which is < 0), so our bet on Team 2 wins.
        # Odds are +120, so profit should be $1.20.
        pnl = MLModel._calculate_pnl(
            pick_condition=np.array([False]),
            actual_vs_line=np.array([-7]),
            odds=np.array([120])
        )
        self.assertAlmostEqual(pnl, 1.20)

    def test_vectorized_mixed_scenario(self):
        """Test a complex scenario with multiple bets to verify vectorization."""
        # Game 1: Bet Team 1, Team 1 wins. Odds +100. P/L = +1.0
        # Game 2: Bet Team 1, Team 1 loses. Odds -110. P/L = -1.0
        # Game 3: Bet Team 2, Team 2 wins. Odds -150. P/L = +0.666... (100/150)
        # Game 4: Bet Team 1, it's a push. Odds -110. P/L = 0.0
        # Game 5: Bet Team 1, Team 1 wins. Odds are 0. P/L = 0.0 (ignored)
        # Game 6: Bet Team 2, Team 2 loses. Odds +130. P/L = -1.0
        
        pick_condition = np.array([True, True, False, True, True, False])
        actual_vs_line = np.array([10,  -5,   -3,    0,    7,    2])
        odds           = np.array([100, -110, -150, -110,  0,    130])

        # Total Expected P/L = 1.0 - 1.0 + (100/150) + 0.0 + 0.0 - 1.0 = -0.333...
        expected_pnl = 1.0 - 1.0 + (100/150.0) - 1.0
        
        actual_pnl = MLModel._calculate_pnl(pick_condition, actual_vs_line, odds)
        
        self.assertAlmostEqual(actual_pnl, expected_pnl)

# This allows running the tests directly from the command line
if __name__ == '__main__':
    unittest.main()