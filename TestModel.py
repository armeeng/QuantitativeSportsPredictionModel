import pandas as pd
import numpy as np

class TestModel:
    """
    A simple class to display the results from an MLModel's test set.
    """
    def __init__(self, predictions, y_test, test_odds):
        """
        Initializes the TestModel with the data from an MLModel instance.

        Args:
            predictions: The output from model.predictions.
            y_test: The true values from model.y_test.
            test_odds: The betting odds from model.test_odds.
        """
        self.predictions = predictions
        self.y_test = y_test
        self.test_odds = test_odds
        # Cache outcomes to avoid recalculating
        self._outcomes = None

    @staticmethod
    def _calculate_profit(odds, bet_amount=1.0):
        """Calculates profit from American odds for a given bet amount."""
        odds = pd.to_numeric(odds, errors='coerce')
        profit = np.zeros_like(odds, dtype=float)
        
        # Where odds are positive
        positive_mask = odds > 0
        profit[positive_mask] = (odds[positive_mask] / 100.0) * bet_amount
        
        # Where odds are negative
        negative_mask = odds < 0
        profit[negative_mask] = (100.0 / np.abs(odds[negative_mask])) * bet_amount
        
        return profit

    def _get_outcomes(self):
        """
        A private helper to determine actual and predicted outcomes.
        This avoids redundant calculations between accuracy and PnL methods.
        """
        if self._outcomes is not None:
            return self._outcomes

        # --- 1. Determine Actual Outcomes ---
        y_test_arr = self.y_test.to_numpy() if isinstance(self.y_test, pd.DataFrame) else np.array(self.y_test)
        if y_test_arr.ndim != 2 or y_test_arr.shape[1] != 2:
            raise ValueError(f"y_test must be a 2D array with shape (n_samples, 2), but got {y_test_arr.shape}")

        actual_t1_score, actual_t2_score = y_test_arr[:, 0], y_test_arr[:, 1]
        actual_winner_is_t1 = actual_t1_score > actual_t2_score

        actual_margin = actual_t1_score - actual_t2_score
        team1_spread = pd.to_numeric(self.test_odds['team1_spread'], errors='coerce')
        spread_outcome = actual_margin + team1_spread
        actual_spread_is_t1_cover = spread_outcome > 0
        spread_pushes = spread_outcome == 0

        actual_total = actual_t1_score + actual_t2_score
        total_line = pd.to_numeric(self.test_odds['total_score'], errors='coerce')
        total_outcome = actual_total - total_line
        actual_is_over = total_outcome > 0
        ou_pushes = total_outcome == 0

        # --- 2. Determine Predicted Outcomes ---
        if isinstance(self.predictions, dict): # Classifier
            pred_winner_is_t1 = self.predictions['win'][:, 1] > 0.5
            pred_spread_is_t1_cover = self.predictions['spread'][:, 1] > 0.5
            pred_is_over = self.predictions['over'][:, 1] > 0.5
        else: # Regressor
            pred_t1_score, pred_t2_score = self.predictions[:, 0], self.predictions[:, 1]
            pred_winner_is_t1 = pred_t1_score > pred_t2_score
            predicted_margin = pred_t1_score - pred_t2_score
            pred_spread_is_t1_cover = (predicted_margin + team1_spread) > 0
            predicted_total = pred_t1_score + pred_t2_score
            pred_is_over = predicted_total > total_line

        self._outcomes = {
            'actual_winner_is_t1': actual_winner_is_t1,
            'actual_spread_is_t1_cover': actual_spread_is_t1_cover,
            'spread_pushes': spread_pushes,
            'actual_is_over': actual_is_over,
            'ou_pushes': ou_pushes,
            'pred_winner_is_t1': pred_winner_is_t1,
            'pred_spread_is_t1_cover': pred_spread_is_t1_cover,
            'pred_is_over': pred_is_over
        }
        return self._outcomes

    def calculate_accuracy(self):
        """Calculates the accuracy of the model's predictions."""
        o = self._get_outcomes()
        
        win_accuracy = np.mean(o['pred_winner_is_t1'] == o['actual_winner_is_t1'])
        
        non_push_spread = ~o['spread_pushes']
        spread_accuracy = np.mean(o['pred_spread_is_t1_cover'][non_push_spread] == o['actual_spread_is_t1_cover'][non_push_spread])
        
        non_push_ou = ~o['ou_pushes']
        ou_accuracy = np.mean(o['pred_is_over'][non_push_ou] == o['actual_is_over'][non_push_ou])

        return {
            'win_accuracy': win_accuracy,
            'spread_accuracy': spread_accuracy,
            'ou_accuracy': ou_accuracy
        }

    def calculate_pnl_of_all_games(self, bet_amount=1.0):
        """
        Calculates the Profit and Loss (PnL) for $1 bets on the model's predictions,
        skipping any games where the odds are missing or zero.
        """
        o = self._get_outcomes()
        
        # --- 1. Moneyline PnL ---
        bet_on_t1 = o['pred_winner_is_t1']
        actual_t1 = o['actual_winner_is_t1']
        ml_odds = np.where(bet_on_t1,
                        pd.to_numeric(self.test_odds['team1_ml'], errors='coerce'),
                        pd.to_numeric(self.test_odds['team2_ml'], errors='coerce'))
        # valid if not NaN and not zero
        valid_ml = (~np.isnan(ml_odds)) & (ml_odds != 0)
        ml_profits = self._calculate_profit(ml_odds, bet_amount)
        
        # only consider valid games
        ml_wins = bet_on_t1 == actual_t1
        ml_pnl = np.sum(
            np.where(ml_wins[valid_ml], ml_profits[valid_ml], -bet_amount)
        )

        # --- 2. Spread PnL (excluding pushes) ---
        non_push_spread = ~o['spread_pushes']
        bet_on_spread = o['pred_spread_is_t1_cover'][non_push_spread]
        actual_spread = o['actual_spread_is_t1_cover'][non_push_spread]
        # grab odds
        t1_sp_odds = pd.to_numeric(self.test_odds['team1_spread_odds'], errors='coerce')[non_push_spread]
        t2_sp_odds = pd.to_numeric(self.test_odds['team2_spread_odds'], errors='coerce')[non_push_spread]
        sp_odds = np.where(bet_on_spread, t1_sp_odds, t2_sp_odds)
        valid_sp = (~np.isnan(sp_odds)) & (sp_odds != 0)
        sp_profits = self._calculate_profit(sp_odds, bet_amount)
        
        sp_wins = bet_on_spread == actual_spread
        sp_pnl = np.sum(
            np.where(sp_wins[valid_sp], sp_profits[valid_sp], -bet_amount)
        )

        # --- 3. Over/Under PnL (excluding pushes) ---
        non_push_ou = ~o['ou_pushes']
        bet_over = o['pred_is_over'][non_push_ou]
        actual_over = o['actual_is_over'][non_push_ou]
        # grab odds
        over_odds = pd.to_numeric(self.test_odds['over_odds'], errors='coerce')[non_push_ou]
        under_odds = pd.to_numeric(self.test_odds['under_odds'], errors='coerce')[non_push_ou]
        ou_odds = np.where(bet_over, over_odds, under_odds)
        valid_ou = (~np.isnan(ou_odds)) & (ou_odds != 0)
        ou_profits = self._calculate_profit(ou_odds, bet_amount)
        
        ou_wins = bet_over == actual_over
        ou_pnl = np.sum(
            np.where(ou_wins[valid_ou], ou_profits[valid_ou], -bet_amount)
        )

        return {
            'moneyline_pnl': ml_pnl,
            'spread_pnl':    sp_pnl,
            'ou_pnl':        ou_pnl
        }

    
    def calculate_pnl_of_game_above_ev_threshold(self, ev_threshold=0, bet_amount=1.0):
        """
        For classifiers, calculates PnL by only betting on games where the 
        Expected Value (EV) is above a given threshold.

        Returns:
            A dictionary containing PnL and the count of bets placed for each bet type.
            Returns NaNs if the model is not a classifier.
        """
        # This function only works for classifiers that provide probabilities.
        if not isinstance(self.predictions, dict):
            nan_result = {'pnl': np.nan, 'count': 0}
            return {'moneyline': nan_result, 'spread': nan_result, 'ou': nan_result}
            
        o = self._get_outcomes()

        # --- Calculate EV for all 6 possible outcomes ---
        # Probabilities from the model
        prob_t1_win = self.predictions['win'][:, 1]
        prob_t1_cover = self.predictions['spread'][:, 1]
        prob_over = self.predictions['over'][:, 1]

        # Potential profits for each outcome
        profit_t1_win = self._calculate_profit(self.test_odds['team1_ml'], bet_amount)
        profit_t2_win = self._calculate_profit(self.test_odds['team2_ml'], bet_amount)
        profit_t1_cover = self._calculate_profit(self.test_odds['team1_spread_odds'], bet_amount)
        profit_t2_cover = self._calculate_profit(self.test_odds['team2_spread_odds'], bet_amount)
        profit_over = self._calculate_profit(self.test_odds['over_odds'], bet_amount)
        profit_under = self._calculate_profit(self.test_odds['under_odds'], bet_amount)

        # Expected Value = (Prob_Win * Profit) - (Prob_Loss * Stake)
        ev_t1_win = (prob_t1_win * profit_t1_win) - ((1 - prob_t1_win) * bet_amount)
        ev_t2_win = ((1 - prob_t1_win) * profit_t2_win) - (prob_t1_win * bet_amount)
        ev_t1_cover = (prob_t1_cover * profit_t1_cover) - ((1 - prob_t1_cover) * bet_amount)
        ev_t2_cover = ((1 - prob_t1_cover) * profit_t2_cover) - (prob_t1_cover * bet_amount)
        ev_over = (prob_over * profit_over) - ((1 - prob_over) * bet_amount)
        ev_under = ((1 - prob_over) * profit_under) - (prob_over * bet_amount)

        # --- Process Moneyline Bets ---
        bet_on_t1_ml = ev_t1_win > ev_t2_win
        best_ml_ev = np.where(bet_on_t1_ml, ev_t1_win, ev_t2_win)
        place_ml_bet = best_ml_ev > ev_threshold
        
        ml_bet_won = (bet_on_t1_ml == o['actual_winner_is_t1'])
        ml_profit_to_use = np.where(bet_on_t1_ml, profit_t1_win, profit_t2_win)
        ml_pnl_per_game = np.where(ml_bet_won, ml_profit_to_use, -bet_amount)
        
        moneyline_pnl = np.sum(ml_pnl_per_game[place_ml_bet])
        moneyline_bet_count = np.sum(place_ml_bet)

        # --- Process Spread Bets (excluding pushes) ---
        non_push_spread = ~o['spread_pushes']
        bet_on_t1_spread = ev_t1_cover > ev_t2_cover
        best_spread_ev = np.where(bet_on_t1_spread, ev_t1_cover, ev_t2_cover)
        place_spread_bet = (best_spread_ev > ev_threshold) & non_push_spread

        spread_bet_won = (bet_on_t1_spread == o['actual_spread_is_t1_cover'])
        spread_profit_to_use = np.where(bet_on_t1_spread, profit_t1_cover, profit_t2_cover)
        spread_pnl_per_game = np.where(spread_bet_won, spread_profit_to_use, -bet_amount)
        
        spread_pnl = np.sum(spread_pnl_per_game[place_spread_bet])
        spread_bet_count = np.sum(place_spread_bet)
        
        # --- Process Over/Under Bets (excluding pushes) ---
        non_push_ou = ~o['ou_pushes']
        bet_on_over = ev_over > ev_under
        best_ou_ev = np.where(bet_on_over, ev_over, ev_under)
        place_ou_bet = (best_ou_ev > ev_threshold) & non_push_ou

        ou_bet_won = (bet_on_over == o['actual_is_over'])
        ou_profit_to_use = np.where(bet_on_over, profit_over, profit_under)
        ou_pnl_per_game = np.where(ou_bet_won, ou_profit_to_use, -bet_amount)
        
        ou_pnl = np.sum(ou_pnl_per_game[place_ou_bet])
        ou_bet_count = np.sum(place_ou_bet)
        
        return {
            'moneyline': {'pnl': moneyline_pnl, 'count': moneyline_bet_count},
            'spread': {'pnl': spread_pnl, 'count': spread_bet_count},
            'ou': {'pnl': ou_pnl, 'count': ou_bet_count}
        }

    def display_results(self):
        """Displays the calculated accuracies and PnL."""
        num_games = len(self.y_test)
        
        # --- Display Calculated Accuracies ---
        accuracies = self.calculate_accuracy()
        print("\n[4] Betting Accuracies:")
        print(f"  - Win/Loss Accuracy:    {accuracies['win_accuracy']:.2%}")
        print(f"  - Spread Accuracy:      {accuracies['spread_accuracy']:.2%}")
        print(f"  - Over/Under Accuracy:  {accuracies['ou_accuracy']:.2%}")

        # --- Display Flat Bet PnL ---
        pnl = self.calculate_pnl_of_all_games()
        print(f"\n[5] Profit & Loss (flat $1 bets on all {num_games} games):")
        print(f"  - Moneyline PnL:      ${pnl['moneyline_pnl']:.2f}")
        print(f"  - Spread PnL:         ${pnl['spread_pnl']:.2f}")
        print(f"  - Over/Under PnL:     ${pnl['ou_pnl']:.2f}")

        # --- Display EV-Based PnL (only for classifiers) ---
        if isinstance(self.predictions, dict):
            ev_pnl = self.calculate_pnl_of_game_above_ev_threshold()
            ml_info = ev_pnl['moneyline']
            spread_info = ev_pnl['spread']
            ou_info = ev_pnl['ou']
            
            print("\n[6] PnL on +EV Bets (Classifier Only):")
            print(f"  - Moneyline:  ${ml_info['pnl']:.2f} from {ml_info['count']} bets ({ml_info['count']/num_games:.1%})")
            print(f"  - Spread:     ${spread_info['pnl']:.2f} from {spread_info['count']} bets ({spread_info['count']/num_games:.1%})")
            print(f"  - Over/Under: ${ou_info['pnl']:.2f} from {ou_info['count']} bets ({ou_info['count']/num_games:.1%})")

        print("\n------------------------------------")