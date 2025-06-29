import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibrationDisplay

class TestModel:
    """
    A simple class to display the results from an MLModel's test set.
    This version includes corrections to properly handle missing or invalid
    betting odds and adds accuracy calculations, along with Kelly Criterion betting simulation.
    NEW: Adds calibration analysis for both the model and the vig-free market odds.
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

    @staticmethod
    def _american_to_decimal(american_odds):
        """Converts American odds to decimal odds."""
        american_odds = pd.to_numeric(american_odds, errors='coerce')
        decimal_odds = np.full_like(american_odds, np.nan, dtype=float)

        positive_mask = american_odds > 0
        decimal_odds[positive_mask] = (american_odds[positive_mask] / 100) + 1

        negative_mask = american_odds < 0
        decimal_odds[negative_mask] = (100 / np.abs(american_odds[negative_mask])) + 1

        return decimal_odds

    # --- NEW: Helper method to convert American odds to implied probability ---
    @staticmethod
    def _american_to_implied_prob(american_odds):
        """Converts American odds to their implied probability."""
        american_odds = pd.to_numeric(american_odds, errors='coerce')
        prob = np.full_like(american_odds, np.nan, dtype=float)
        
        positive_mask = american_odds > 0
        prob[positive_mask] = 100 / (american_odds[positive_mask] + 100)

        negative_mask = american_odds < 0
        prob[negative_mask] = np.abs(american_odds[negative_mask]) / (np.abs(american_odds[negative_mask]) + 100)
        
        return prob

    # --- NEW: Helper method to calculate vig-free probabilities ---
    @staticmethod
    def _get_vig_free_probs(odds1, odds2):
        """
        Calculates the vig-free (fair) probability for the first outcome.
        
        Args:
            odds1: The American odds for the first outcome (e.g., Team 1 ML).
            odds2: The American odds for the second outcome (e.g., Team 2 ML).

        Returns:
            A numpy array of the fair probability for the first outcome.
        """
        prob1 = TestModel._american_to_implied_prob(odds1)
        prob2 = TestModel._american_to_implied_prob(odds2)
        
        market_sum = prob1 + prob2
        
        # Where the market sum is valid for vig removal ( > 1)
        valid_market = (market_sum > 1)
        
        fair_prob1 = np.full_like(prob1, np.nan)
        fair_prob1[valid_market] = prob1[valid_market] / market_sum[valid_market]
        
        return fair_prob1

    @staticmethod
    def _calculate_kelly_fraction(prob_win, decimal_odds):
        """
        Calculates the Kelly Criterion fraction.
        Args:
            prob_win (float): The model's estimated probability of winning the bet.
            decimal_odds (float): The decimal odds for the bet.
        Returns:
            The fraction of the bankroll to bet.
        """
        b = decimal_odds - 1
        q = 1 - prob_win
        kelly_fraction = (b * prob_win - q) / b
        return np.maximum(0, kelly_fraction)


    def _get_outcomes(self):
        """A private helper to determine actual and predicted outcomes."""
        if self._outcomes is not None:
            return self._outcomes
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

    def calculate_accuracies(self):
        """Calculates the accuracy of the model's predictions for win, spread, and total."""
        o = self._get_outcomes()
        
        correct_winner_preds = np.sum(o['pred_winner_is_t1'] == o['actual_winner_is_t1'])
        total_games = len(o['actual_winner_is_t1'])
        win_accuracy = (correct_winner_preds / total_games) if total_games > 0 else 0
        
        non_push_spread = ~o['spread_pushes']
        correct_spread_preds = np.sum((o['pred_spread_is_t1_cover'] == o['actual_spread_is_t1_cover'])[non_push_spread])
        num_spread_outcomes = np.sum(non_push_spread)
        spread_accuracy = (correct_spread_preds / num_spread_outcomes) if num_spread_outcomes > 0 else 0
        
        non_push_ou = ~o['ou_pushes']
        correct_ou_preds = np.sum((o['pred_is_over'] == o['actual_is_over'])[non_push_ou])
        num_ou_outcomes = np.sum(non_push_ou)
        total_accuracy = (correct_ou_preds / num_ou_outcomes) if num_ou_outcomes > 0 else 0
        
        return {
            'win_accuracy': win_accuracy, 'spread_accuracy': spread_accuracy, 'total_accuracy': total_accuracy,
            'correct_winner_preds': correct_winner_preds, 'total_games': total_games,
            'correct_spread_preds': correct_spread_preds, 'num_spread_outcomes': num_spread_outcomes,
            'correct_ou_preds': correct_ou_preds, 'num_ou_outcomes': num_ou_outcomes,
        }

    def calculate_pnl_of_all_games(self, bet_amount=1.0):
        """Calculates the Profit and Loss (PnL) for $1 bets on the model's predictions."""
        o = self._get_outcomes()
        bet_on_t1_win = o['pred_winner_is_t1']
        ml_odds = np.where(bet_on_t1_win, self.test_odds['team1_ml'], self.test_odds['team2_ml'])
        ml_odds_numeric = pd.to_numeric(ml_odds, errors='coerce')
        valid_ml_bets = pd.notna(ml_odds_numeric) & (ml_odds_numeric != 0)
        ml_bet_won = (bet_on_t1_win == o['actual_winner_is_t1'])
        ml_profits = self._calculate_profit(ml_odds_numeric, bet_amount)
        ml_pnl_per_game = np.where(ml_bet_won, ml_profits, -bet_amount)
        moneyline_pnl = np.sum(ml_pnl_per_game[valid_ml_bets])

        bet_on_t1_cover = o['pred_spread_is_t1_cover']
        spread_odds = np.where(bet_on_t1_cover, self.test_odds['team1_spread_odds'], self.test_odds['team2_spread_odds'])
        spread_odds_numeric = pd.to_numeric(spread_odds, errors='coerce')
        valid_spread_bets = pd.notna(spread_odds_numeric) & (spread_odds_numeric != 0) & (~o['spread_pushes'])
        spread_bet_won = (bet_on_t1_cover == o['actual_spread_is_t1_cover'])
        spread_profits = self._calculate_profit(spread_odds_numeric, bet_amount)
        spread_pnl_per_game = np.where(spread_bet_won, spread_profits, -bet_amount)
        spread_pnl = np.sum(spread_pnl_per_game[valid_spread_bets])

        bet_on_over = o['pred_is_over']
        ou_odds = np.where(bet_on_over, self.test_odds['over_odds'], self.test_odds['under_odds'])
        ou_odds_numeric = pd.to_numeric(ou_odds, errors='coerce')
        valid_ou_bets = pd.notna(ou_odds_numeric) & (ou_odds_numeric != 0) & (~o['ou_pushes'])
        ou_bet_won = (bet_on_over == o['actual_is_over'])
        ou_profits = self._calculate_profit(ou_odds_numeric, bet_amount)
        ou_pnl_per_game = np.where(ou_bet_won, ou_profits, -bet_amount)
        ou_pnl = np.sum(ou_pnl_per_game[valid_ou_bets])
        
        return {
            'moneyline_pnl': moneyline_pnl, 'spread_pnl': spread_pnl, 'ou_pnl': ou_pnl,
            'moneyline_bets_placed': np.sum(valid_ml_bets),
            'spread_bets_placed': np.sum(valid_spread_bets),
            'ou_bets_placed': np.sum(valid_ou_bets),
        }

    def calculate_pnl_of_game_above_ev_threshold(self, ev_threshold=0, bet_amount=1.0):
        """Calculates PnL by only betting on games where Expected Value (EV) is above a threshold."""
        if not isinstance(self.predictions, dict):
            nan_result = {'pnl': np.nan, 'count': 0}
            return {'moneyline': nan_result, 'spread': nan_result, 'ou': nan_result}

        o = self._get_outcomes()
        def is_valid(odds_series):
            odds_num = pd.to_numeric(odds_series, errors='coerce')
            return pd.notna(odds_num) & (odds_num != 0)
        validity = {
            't1_ml': is_valid(self.test_odds['team1_ml']), 't2_ml': is_valid(self.test_odds['team2_ml']),
            't1_spread': is_valid(self.test_odds['team1_spread_odds']), 't2_spread': is_valid(self.test_odds['team2_spread_odds']),
            'over': is_valid(self.test_odds['over_odds']), 'under': is_valid(self.test_odds['under_odds']),
        }
        profits = {key: self._calculate_profit(self.test_odds[val_key], bet_amount)
                   for key, val_key in [('t1_ml', 'team1_ml'), ('t2_ml', 'team2_ml'),
                                        ('t1_spread', 'team1_spread_odds'), ('t2_spread', 'team2_spread_odds'),
                                        ('over', 'over_odds'), ('under', 'under_odds')]}

        prob_t1_win = self.predictions['win'][:, 1]
        prob_t1_cover = self.predictions['spread'][:, 1]
        prob_over = self.predictions['over'][:, 1]
        ev_t1_win = (prob_t1_win * profits['t1_ml']) - ((1 - prob_t1_win) * bet_amount)
        ev_t2_win = ((1 - prob_t1_win) * profits['t2_ml']) - (prob_t1_win * bet_amount)
        ev_t1_cover = (prob_t1_cover * profits['t1_spread']) - ((1 - prob_t1_cover) * bet_amount)
        ev_t2_cover = ((1 - prob_t1_cover) * profits['t2_spread']) - (prob_t1_cover * bet_amount)
        ev_over = (prob_over * profits['over']) - ((1 - prob_over) * bet_amount)
        ev_under = ((1 - prob_over) * profits['under']) - (prob_over * bet_amount)

        ev_t1_win[~validity['t1_ml']] = -np.inf
        ev_t2_win[~validity['t2_ml']] = -np.inf
        ev_t1_cover[~validity['t1_spread']] = -np.inf
        ev_t2_cover[~validity['t2_spread']] = -np.inf
        ev_over[~validity['over']] = -np.inf
        ev_under[~validity['under']] = -np.inf

        bet_on_t1_ml = ev_t1_win > ev_t2_win
        best_ml_ev = np.where(bet_on_t1_ml, ev_t1_win, ev_t2_win)
        place_ml_bet = best_ml_ev > ev_threshold
        ml_bet_won = (bet_on_t1_ml == o['actual_winner_is_t1'])
        ml_profit_to_use = np.where(bet_on_t1_ml, profits['t1_ml'], profits['t2_ml'])
        ml_pnl_per_game = np.where(ml_bet_won, ml_profit_to_use, -bet_amount)
        moneyline_pnl = np.sum(ml_pnl_per_game[place_ml_bet])

        bet_on_t1_spread = ev_t1_cover > ev_t2_cover
        best_spread_ev = np.where(bet_on_t1_spread, ev_t1_cover, ev_t2_cover)
        place_spread_bet = (best_spread_ev > ev_threshold) & (~o['spread_pushes'])
        spread_bet_won = (bet_on_t1_spread == o['actual_spread_is_t1_cover'])
        spread_profit_to_use = np.where(bet_on_t1_spread, profits['t1_spread'], profits['t2_spread'])
        spread_pnl_per_game = np.where(spread_bet_won, spread_profit_to_use, -bet_amount)
        spread_pnl = np.sum(spread_pnl_per_game[place_spread_bet])

        bet_on_over = ev_over > ev_under
        best_ou_ev = np.where(bet_on_over, ev_over, ev_under)
        place_ou_bet = (best_ou_ev > ev_threshold) & (~o['ou_pushes'])
        ou_bet_won = (bet_on_over == o['actual_is_over'])
        ou_profit_to_use = np.where(bet_on_over, profits['over'], profits['under'])
        ou_pnl_per_game = np.where(ou_bet_won, ou_profit_to_use, -bet_amount)
        ou_pnl = np.sum(ou_pnl_per_game[place_ou_bet])

        return {
            'moneyline': {'pnl': moneyline_pnl, 'count': np.sum(place_ml_bet)},
            'spread': {'pnl': spread_pnl, 'count': np.sum(place_spread_bet)},
            'ou': {'pnl': ou_pnl, 'count': np.sum(place_ou_bet)}
        }

    def simulate_kelly_betting(self, initial_bankroll=1000, max_fraction=0.01):
        """Simulates betting using the Kelly Criterion for classifiers."""
        if not isinstance(self.predictions, dict):
            nan_result = {'final_bankroll': np.nan, 'bets_placed': 0, 'total_wagered': np.nan}
            return {'moneyline': nan_result, 'spread': nan_result, 'ou': nan_result}

        o = self._get_outcomes()
        bankrolls = {'moneyline': initial_bankroll, 'spread': initial_bankroll, 'ou': initial_bankroll}
        bets_placed = {'moneyline': 0, 'spread': 0, 'ou': 0}
        total_wagered = {'moneyline': 0.0, 'spread': 0.0, 'ou': 0.0}

        prob_t1_win = self.predictions['win'][:, 1]
        prob_t1_cover = self.predictions['spread'][:, 1]
        prob_over = self.predictions['over'][:, 1]
        decimal_odds = {key: self._american_to_decimal(val) for key, val in self.test_odds.items() if 'odds' in key or 'ml' in key}

        outcomes_np = {key: np.array(value) for key, value in o.items()}
        decimal_odds_np = {key: np.array(value) for key, value in decimal_odds.items()}

        num_games = len(self.y_test)
        for i in range(num_games):
            # Moneyline
            if not np.isnan(decimal_odds_np['team1_ml'][i]) and not np.isnan(decimal_odds_np['team2_ml'][i]):
                kelly_t1 = self._calculate_kelly_fraction(prob_t1_win[i], decimal_odds_np['team1_ml'][i])
                kelly_t2 = self._calculate_kelly_fraction(1 - prob_t1_win[i], decimal_odds_np['team2_ml'][i])
                if kelly_t1 > 0 or kelly_t2 > 0:
                    bet_on_t1 = kelly_t1 > kelly_t2
                    fraction = min(kelly_t1 if bet_on_t1 else kelly_t2, max_fraction)
                    bet_amount = bankrolls['moneyline'] * fraction
                    if bet_amount > 0:
                        total_wagered['moneyline'] += bet_amount
                        won = (bet_on_t1 == outcomes_np['actual_winner_is_t1'][i])
                        odds_to_use = decimal_odds_np['team1_ml'][i] if bet_on_t1 else decimal_odds_np['team2_ml'][i]
                        bankrolls['moneyline'] += bet_amount * (odds_to_use - 1) if won else -bet_amount
                        bets_placed['moneyline'] += 1
            # Spread
            if not outcomes_np['spread_pushes'][i] and not np.isnan(decimal_odds_np['team1_spread_odds'][i]) and not np.isnan(decimal_odds_np['team2_spread_odds'][i]):
                kelly_t1_cover = self._calculate_kelly_fraction(prob_t1_cover[i], decimal_odds_np['team1_spread_odds'][i])
                kelly_t2_cover = self._calculate_kelly_fraction(1 - prob_t1_cover[i], decimal_odds_np['team2_spread_odds'][i])
                if kelly_t1_cover > 0 or kelly_t2_cover > 0:
                    bet_on_t1 = kelly_t1_cover > kelly_t2_cover
                    fraction = min(kelly_t1_cover if bet_on_t1 else kelly_t2_cover, max_fraction)
                    bet_amount = bankrolls['spread'] * fraction
                    if bet_amount > 0:
                        total_wagered['spread'] += bet_amount
                        won = (bet_on_t1 == outcomes_np['actual_spread_is_t1_cover'][i])
                        odds_to_use = decimal_odds_np['team1_spread_odds'][i] if bet_on_t1 else decimal_odds_np['team2_spread_odds'][i]
                        bankrolls['spread'] += bet_amount * (odds_to_use - 1) if won else -bet_amount
                        bets_placed['spread'] += 1
            # Over/Under
            if not outcomes_np['ou_pushes'][i] and not np.isnan(decimal_odds_np['over_odds'][i]) and not np.isnan(decimal_odds_np['under_odds'][i]):
                kelly_over = self._calculate_kelly_fraction(prob_over[i], decimal_odds_np['over_odds'][i])
                kelly_under = self._calculate_kelly_fraction(1 - prob_over[i], decimal_odds_np['under_odds'][i])
                if kelly_over > 0 or kelly_under > 0:
                    bet_on_over = kelly_over > kelly_under
                    fraction = min(kelly_over if bet_on_over else kelly_under, max_fraction)
                    bet_amount = bankrolls['ou'] * fraction
                    if bet_amount > 0:
                        total_wagered['ou'] += bet_amount
                        won = (bet_on_over == outcomes_np['actual_is_over'][i])
                        odds_to_use = decimal_odds_np['over_odds'][i] if bet_on_over else decimal_odds_np['under_odds'][i]
                        bankrolls['ou'] += bet_amount * (odds_to_use - 1) if won else -bet_amount
                        bets_placed['ou'] += 1
        return {
            'moneyline': {'final_bankroll': bankrolls['moneyline'], 'bets_placed': bets_placed['moneyline'], 'total_wagered': total_wagered['moneyline']},
            'spread': {'final_bankroll': bankrolls['spread'], 'bets_placed': bets_placed['spread'], 'total_wagered': total_wagered['spread']},
            'ou': {'final_bankroll': bankrolls['ou'], 'bets_placed': bets_placed['ou'], 'total_wagered': total_wagered['ou']}
        }

    # --- REWRITTEN: Now compares model calibration against the market's implied probabilities ---
    def check_calibration(self, n_bins=10):
        """
        Calculates Brier scores and generates calibration plots to compare the reliability
        of the model's probabilities against the vig-free market odds. Only for classifiers.
        """
        if not isinstance(self.predictions, dict):
            print("\nCalibration check is only available for classifier models.")
            return

        print("\n--- Model vs. Market Calibration Analysis ---")
        o = self._get_outcomes()
        
        calibration_data = {
            'Moneyline': {
                'y_true': o['actual_winner_is_t1'],
                'model_prob': self.predictions['win'][:, 1],
                'market_odds1': self.test_odds['team1_ml'],
                'market_odds2': self.test_odds['team2_ml'],
                'mask': np.full_like(o['actual_winner_is_t1'], True, dtype=bool)
            },
            'Spread': {
                'y_true': o['actual_spread_is_t1_cover'],
                'model_prob': self.predictions['spread'][:, 1],
                'market_odds1': self.test_odds['team1_spread_odds'],
                'market_odds2': self.test_odds['team2_spread_odds'],
                'mask': ~o['spread_pushes']
            },
            'Over/Under': {
                'y_true': o['actual_is_over'],
                'model_prob': self.predictions['over'][:, 1],
                'market_odds1': self.test_odds['over_odds'],
                'market_odds2': self.test_odds['under_odds'],
                'mask': ~o['ou_pushes']
            }
        }

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle('Calibration Plots: Model vs. Vig-Free Market Odds', fontsize=16)

        for i, (bet_type, data) in enumerate(calibration_data.items()):
            ax = axes[i]
            print(f"\n--- {bet_type} Calibration ---")

            # --- 1. Model Calibration ---
            y_true = data['y_true'][data['mask']]
            model_prob = data['model_prob'][data['mask']]
            
            brier_model = brier_score_loss(y_true, model_prob)
            baseline_brier = np.mean(y_true) * (1 - np.mean(y_true))
            bss_model = 1 - (brier_model / baseline_brier)
            print(f"  Model Brier Score:      {brier_model:.4f} (BSS: {bss_model:.3f})")
            
            CalibrationDisplay.from_predictions(y_true, model_prob, n_bins=n_bins, ax=ax, name='Model')

            # --- 2. Market Odds Calibration ---
            market_prob = self._get_vig_free_probs(data['market_odds1'], data['market_odds2'])
            valid_market_mask = ~np.isnan(market_prob) & data['mask']
            
            if np.any(valid_market_mask):
                y_true_market = data['y_true'][valid_market_mask]
                market_prob_valid = market_prob[valid_market_mask]

                brier_market = brier_score_loss(y_true_market, market_prob_valid)
                bss_market = 1 - (brier_market / baseline_brier) # Use same baseline for fair comparison
                print(f"  Market Brier Score:     {brier_market:.4f} (BSS: {bss_market:.3f})")

                CalibrationDisplay.from_predictions(y_true_market, market_prob_valid, n_bins=n_bins, ax=ax, name='Market (Vig-Free)')
            else:
                print("  Market Brier Score:     N/A (Missing or invalid odds)")

            ax.set_title(f'{bet_type} Calibration')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def display_results(self, initial_bankroll=1000):
        """Displays the calculated accuracies and PnL."""
        acc = self.calculate_accuracies()
        print("\nModel Prediction Accuracy:")
        print(f"  - Winner Accuracy:     {acc['win_accuracy']:.2%} ({acc['correct_winner_preds']}/{acc['total_games']})")
        print(f"  - Spread Accuracy:     {acc['spread_accuracy']:.2%} ({acc['correct_spread_preds']}/{acc['num_spread_outcomes']})")
        print(f"  - Over/Under Accuracy: {acc['total_accuracy']:.2%} ({acc['correct_ou_preds']}/{acc['num_ou_outcomes']})")

        pnl = self.calculate_pnl_of_all_games()
        print(f"\nProfit & Loss (flat $1 bets on all available odds):")
        print(f"  - Moneyline PnL:      ${pnl['moneyline_pnl']:.2f} from {pnl['moneyline_bets_placed']} bets")
        print(f"  - Spread PnL:         ${pnl['spread_pnl']:.2f} from {pnl['spread_bets_placed']} bets")
        print(f"  - Over/Under PnL:     ${pnl['ou_pnl']:.2f} from {pnl['ou_bets_placed']} bets")

        if isinstance(self.predictions, dict):
            ev_pnl = self.calculate_pnl_of_game_above_ev_threshold()
            ml_info, spread_info, ou_info = ev_pnl['moneyline'], ev_pnl['spread'], ev_pnl['ou']
            print("\nPnL on +EV Bets (Classifier Only):")
            print(f"  - Moneyline:  ${ml_info['pnl']:.2f} from {ml_info['count']} bets ({ml_info['count']/acc['total_games']:.1%})")
            print(f"  - Spread:     ${spread_info['pnl']:.2f} from {spread_info['count']} bets ({spread_info['count']/acc['total_games']:.1%})")
            print(f"  - Over/Under: ${ou_info['pnl']:.2f} from {ou_info['count']} bets ({ou_info['count']/acc['total_games']:.1%})")

            kelly_results = self.simulate_kelly_betting(initial_bankroll=initial_bankroll)
            kelly_ml, kelly_spread, kelly_ou = kelly_results['moneyline'], kelly_results['spread'], kelly_results['ou']
            print(f"\nKelly Criterion Simulation (Initial Bankroll: ${initial_bankroll}):")
            print(f"  - Moneyline:  Final Bankroll: ${kelly_ml['final_bankroll']:.2f} (Profit: ${kelly_ml['final_bankroll'] - initial_bankroll:.2f}) from {kelly_ml['bets_placed']} bets")
            print(f"  - Spread:     Final Bankroll: ${kelly_spread['final_bankroll']:.2f} (Profit: ${kelly_spread['final_bankroll'] - initial_bankroll:.2f}) from {kelly_spread['bets_placed']} bets")
            print(f"  - Over/Under: Final Bankroll: ${kelly_ou['final_bankroll']:.2f} (Profit: ${kelly_ou['final_bankroll'] - initial_bankroll:.2f}) from {kelly_ou['bets_placed']} bets")
            
            # This is now the last major output.
            self.check_calibration()

        print("\n------------------------------------")