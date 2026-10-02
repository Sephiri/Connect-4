from __future__ import annotations
import numpy as np
from abc import abstractmethod
from numba import jit
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from board import Board


class Heuristic:
    """Abstract class defining a heuristic
    """
    def __init__(self, game_n: int) -> None:
        """
        Args:
            game_n (int): n in a row required to win
        """
        self.game_n: int = game_n
        self.eval_count: int = 0


    def get_best_action(self, player_id: int, board: Board) -> int:
        """Determines the best column for the next move

        Args:
            player_id (int): the player for which to compute the heuristic value
            board (Board): the board to evaluate

        Returns:
            int: column with the best heuristic value
        """
        min_util: int = -max(board.get_board_state().shape)
        utils: np.ndarray = np.full(board.width, min_util - 1, dtype=int)

        for i in range(board.width):
            if board.is_valid(i):
                self.eval_count += 1
                utils[i] = self.evaluate_board(player_id, board.get_new_board(i, player_id))

        return np.argmax(utils)
    

    def evaluate_board(self, player_id: int, board: Board) -> int:
        """Helper function to assign a utility to a board

        Args:
            player_id (int): the player for which to compute the heuristic value
            board (Board): the board to evaluate

        Returns:
            int: the utility of a board
        """
        self.eval_count += 1
        state: np.ndarray = board.get_board_state()
        return self._evaluate(player_id, state, self.winning(state, self.game_n))
    

    @staticmethod
    def winning(state: np.ndarray, game_n: int) -> int:
        """Determines whether a player has won, and if so, which one

        Args:
            state (np.ndarray): the board to check
            game_n (int): n in a row required to win

        Returns:
            int: 1 or 2 if the respective player won, -1 if the game is a draw, 0 otherwise
        """
        from app import winning as app_winning # imported here to avoid circular imports
        return app_winning(state, game_n)
    

    def __str__(self) -> str:
        """ 
        Returns:
            str: name of the heuristic
        """
        return self._name()


    @abstractmethod
    def _name(self) -> str:
        """Abstract method for naming the heuristic

        Returns:
            str: name of the heuristic
        """
        pass


    @abstractmethod
    def _evaluate(self, player_id: int, state: np.ndarray, winner: int) -> int:
        """Abstract method for evaluating a board state

        Args:
            player_id (int): the player for which to compute the heuristic value
            state (np.ndarray): the board to check
            winner (int): 1 or 2 if the respective player won, -1 if the game is a draw, 0 otherwise

        Returns:
            int: heuristic value for the board state
        """
        pass    


class SimpleHeuristic(Heuristic):
    """A simple heuristic
    Inherits from Heuristic
    """
    def __init__(self, game_n: int) -> None:
        """
        Args:
            game_n (int): n in a row required to win
        """
        super().__init__(game_n)


    def _name(self) -> str:
        """
        Returns:
            str: the name of the heuristic; Simple
        """
        return 'Simple'
    
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _evaluate(player_id: int, state: np.ndarray, winner: int) -> int:
        """Determine utility of a board state

        Args:
            player_id (int): the player for which to compute the heuristic value
            state (np.ndarray): the board to check
            winner (int): 1 or 2 if the respective player won, -1 if the game is a draw, 0 otherwise

        Returns:
            int: heuristic value for the board state
        """
        width: int
        height: int
        width, height = state.shape

        if winner == player_id: # player won
            return max(width, height)
        elif winner < 0: # draw
            return 0
        elif winner > 0: # player lost
            return -max(width, height)
        
        # not winning or losing, return highest number of claimed squares in a row      
        max_in_row: int = 0
        for i in range(width):
            for j in range(height):
                if state[i, j] != player_id:
                    continue

                max_in_row = max(max_in_row, 1)

                for x in range(1, width - i):
                    if state[i + x, j] == player_id:
                        max_in_row = max(max_in_row, x + 1)
                    else:
                        break

                for y in range(1, height - j):
                    if state[i, j + y] == player_id:
                        max_in_row = max(max_in_row, y + 1)
                    else:
                        break

                for d in range(1, min(width - i, height - j)):
                    if state[i + d, j + d] == player_id:
                        max_in_row = max(max_in_row, d + 1)
                    else:
                        break

                for a in range(1, min(width - i, j)):
                    if state[i + a, j - a] == player_id:
                        max_in_row = max(max_in_row, a + 1)
                    else:
                        break

        return max_in_row

class ExpandedHeuristic(Heuristic):
    """
    This heuristic extends SimpleHeuristic by checking whether a consecutive line of the player's stones can be extended immediately at either end.
    If neither end is playable, the line receives a score of 0. Otherwise, its score is the number of consecutive stones. 

    Inherits from Heuristic
    """
    def __init__(self, game_n: int) -> None:
        """
        Args:
            game_n (int): n in a row required to win
        """
        super().__init__(game_n)


    def _name(self) -> str:
        """
        Returns:
            str: the name of the heuristic; Simple
        """
        return 'Gibberish'
    
    
    def _evaluate(self, player_id: int, state: np.ndarray, winner: int) -> int:
        """Determine utility of a board state
        
        Args:
            player_id (int): the player for which to compute the heuristic value
            state (np.ndarray): the board to check
            winner (int): 1 or 2 if the respective player won, -1 if the game is a draw, 0 otherwise

        Returns:
            int: heuristic value for the board state
        """
         
        width: int
        height: int
        width, height = state.shape

        adversary = 2 if player_id == 1 else 1

        # same as SimpleHeuristic, winning or losing are give more points than simple multiple stones in a row
        if winner == player_id: # player won
            return max(width, height)
        elif winner < 0: # draw
            return 0
        elif winner > 0: # player lost
            return -max(width, height)
        
        # not winning or losing, return highest number of claimed squares in a row      
        max_in_row: int = 0
        for i in range(width):
            for j in range(height):
                if state[i, j] != player_id:
                    continue

                # Horizontal
                temp_max_in_row = 1

                for x in range(1, width - i):
                    if state[i + x, j] == player_id:
                        temp_max_in_row += 1
                    elif state[i + x, j] == adversary:
                        break
                    else:
                        break

                if not (self._is_playable(state, i - 1, j) or 
                        self._is_playable(state, i + temp_max_in_row, j)):
                    temp_max_in_row = 0

                max_in_row = max(max_in_row, temp_max_in_row)

                # Vertical
                temp_max_in_row = 1

                for y in range(1, height - j):
                    if state[i, j + y] == player_id:
                        temp_max_in_row += 1
                    elif state[i, j + y] == adversary:
                        break
                    else:
                        break

                if not (self._is_playable(state, i, j - 1) or 
                        self._is_playable(state, i, j + temp_max_in_row)):
                    temp_max_in_row = 0

                max_in_row = max(max_in_row, temp_max_in_row)

                # Diagonally down-right
                temp_max_in_row = 1

                for d in range(1, min(width - i, height - j)):
                    if state[i + d, j + d] == player_id:
                        temp_max_in_row += 1
                    elif state[i + d, j + d] == adversary:
                        break
                    else:
                        break

                if not (self._is_playable(state, i - 1, j - 1) or 
                        self._is_playable(state, i + temp_max_in_row, j + temp_max_in_row)):
                    temp_max_in_row = 0

                max_in_row = max(max_in_row, temp_max_in_row)

                # Diagonally up-right
                temp_max_in_row = 1

                for a in range(1, min(width - i, j + 1)):
                    if state[i + a, j - a] == player_id:
                        temp_max_in_row += 1
                    elif state[i + a, j - a] == adversary:
                        break
                    else:
                        break

                if not (self._is_playable(state, i - 1, j + 1) or self._is_playable(
                        state, i + temp_max_in_row, j - temp_max_in_row)):
                    temp_max_in_row = 0

                max_in_row = max(max_in_row, temp_max_in_row)

        return max_in_row

    @staticmethod
    def _is_playable(state: np.ndarray, col: int, row: int) -> bool:
        width, height = state.shape
        
        if not (0 <= col < width and 0 <= row < height):
            return False
        
        return (
            state[col, row] == 0
            and (row == height - 1 or state[col, row + 1] != 0)
        )