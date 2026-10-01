from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board
import random
from utils import winning

class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass

class Node:
    """
    Tree Data structure for MinMax Tree
    """
    def __init__(self, board: Board, value: int, move: int | None = None) -> None:
        """
        Args:
            board (Board): current board state
            value (int): current position evaluated using the heuristic
            move (int): Which move in which column led to this node
        """
        self.board = board
        self.value = value
        self.move = move
        self.children: list["Node"] = []

class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        # DONE: implement minmax algortihm!
        # HINT: use the functions on the 'board' object to produce a new board given a specific move
        # HINT: use the functions on the 'heuristic' object to produce evaluations for the different board states!
        
        # This is obviously not enough (this is depth 1)
        # Your assignment is to create a data structure (tree) to store the gameboards such that you can evaluate a higher depths.
        # Then, use the minmax algorithm to search through this tree to find the best move/action to take!

        root_value = self.heuristic.evaluate_board(self.player_id, board)
        root = Node(board, root_value)

        self._build_tree(root, 0, self.depth, self.player_id)

        _, max_move = self._minimax(root, True)
        return max_move

    def _build_tree(self, node: Node, current_depth: int, max_depth: int, player_id: int) -> None: 
        board = node.board
        
        if (current_depth == max_depth or winning(node.board.get_board_state(), self.game_n) != 0):
            return 

        for col in range(board.width):
            if board.is_valid(col):
                new_board: Board = board.get_new_board(col, player_id)
                value: int = self.heuristic.evaluate_board(self.player_id, new_board)
                child = Node(new_board, value, col)
                node.children.append(child)
                next_player = 2 if player_id == 1 else 1
                self._build_tree(child, current_depth +1, max_depth, next_player)

    def _minimax(self, node: Node, maximize: bool) -> tuple[int, int | None]:
        """
        Args: 
            node (Node): current node to calculate the minmax value
            maximize (bool): True or False whether minimax should maximize or minimize the children's values 
        Returns: 
            int: best value based on min or max
            int: respective column that led to the node with the best value 
        """
        if not node.children:
            return node.value, None

        values = []

        for child in node.children:
            value, _ = self._minimax(child, not maximize)
            values.append((value, child.move))

        if maximize:
            return max(values, key=lambda pair: pair[0])
        else:
            return min(values, key=lambda pair: pair[0])

    

class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm with alpha-beta pruning
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        # DONE: implement minmax algorithm with alpha beta pruning!

        root_value = self.heuristic.evaluate_board(self.player_id, board)
        root = Node(board, root_value)

        alpha = -np.inf
        beta = np.inf
        
        _, max_move = self._minimax(root, True, alpha, beta, 0, self.depth, self.player_id,)
        return max_move

    def _minimax(self, node: Node, maximize: bool, alpha:int, beta:int,  current_depth: int, max_depth: int, player_id: int) -> tuple[int, int | None]:
        """
        Args: 
            node (Node): current node to calculate the minmax value
            maximize (bool): True or False whether minimax should maximize or minimize the children's values 
            alpha (int): best achievable max value up to this point
            beta (int): best achievable min value up to this point
        Returns: 
            int: best value based on min or max
            int: respective column that led to the node with the best value 
        """
        if (current_depth == max_depth or winning(node.board.get_board_state(), self.game_n) != 0):
            return node.value, None
        
        board = node.board

        if not any(board.is_valid(col) for col in range(board.width)):
            return node.value, None

        if maximize:
            temp_value_pair = (-np.inf, None)

            for col in range(board.width):
                if board.is_valid(col):
                    new_board: Board = board.get_new_board(col, player_id)
                    node_value: int = self.heuristic.evaluate_board(self.player_id, new_board)
                    child = Node(new_board, node_value, col)
                    
                    next_player = 2 if player_id == 1 else 1
                    minimax_value, _ = self._minimax(child, False, alpha, beta, current_depth +1, max_depth, next_player)

                    candidate = (minimax_value, col)
                    temp_value_pair = max(temp_value_pair, candidate, key=lambda pair: pair[0])
    
                    if temp_value_pair[0] >= beta:
                        return temp_value_pair
    
                    alpha = max(alpha, temp_value_pair[0])
                else:
                    continue
            return temp_value_pair

        else:
            temp_value_pair = (np.inf, None)
    
            for col in range(board.width):
                if board.is_valid(col):
                    new_board: Board = board.get_new_board(col, player_id)
                    node_value: int = self.heuristic.evaluate_board(self.player_id, new_board)
                    child = Node(new_board, node_value, col)

                    next_player = 2 if player_id == 1 else 1

                    minimax_value, _ = self._minimax(child, True, alpha, beta, current_depth +1, max_depth, next_player)

                    candidate = (minimax_value, col)
                    temp_value_pair = min(temp_value_pair, candidate, key=lambda pair: pair[0])

                    if temp_value_pair[0] <= alpha:
                        return temp_value_pair

                    beta = min(beta, temp_value_pair[0])
                else:
                    continue
            return temp_value_pair

class MCNode():
    """
    data structure for MonteCarlo player
    """
    def __init__(self, board: Board, total_reward: int = 0, visits: int = 0, parent: MCNode | None = None, move: int | None = None) -> None:
        self.board = board
        self.total_reward = total_reward
        self.visits = visits
        self.parent = parent
        self.children: list["MCNode"] = []
        self.move = move


class MonteCarloPlayer(PlayerController):
    def __init__(self, player_id: int, game_n: int, max_simulations: int, heuristic: Heuristic) -> None:
            """
            Args:
                player_id (int): id of a player, can take values 1 or 2 (0 = empty)
                game_n (int): n in a row required to win
                max_simulations (int): the max number of simulated gameplays
                heuristic (Heuristic): heuristic used by the player
            """
            super().__init__(player_id, game_n, heuristic)
            self.max_simulations: int = max_simulations

    def make_move(self, board: Board) -> int:
        root_node: MCNode = MCNode(board)
        simulations = self.max_simulations 
        
        while simulations != 0:
            self._build_tree(root_node, self.player_id)
            simulations-= 1

        return self._choose_move(root_node)

    def _choose_move(self, node: MCNode) -> int:
        """
        chooses the best move based on the winning rate (total_reward/visits) for each node
        Args:
            node (MCNode): current node from which point onward the functions decide which child to pick 
        Return:
            int: which column to choose for next move
        """
        result = -np.inf
        move = 0
        for child in node.children:
            temp = child.total_reward / child.visits
            if temp >= result: 
                result = temp
                move = child.move
        return move

    def _build_tree(self, node: MCNode, player_id: int) -> None:
        """
        creating the tree structure for MC
        Args:
            node (MCNode): starting node, from which MonteCarlo algorithm starts
            player_id (int): the player which turn it is
        """
        board = node.board

        if winning(board.get_board_state(), self.game_n) != 0:
            result = self._simulate_game(board, player_id)
            self._set_values(node, result)
            return

        visited_moves = {child.move for child in node.children}
        legal_moves = [col for col in range(board.width) if board.is_valid(col)]
        unvisited_moves = [move for move in legal_moves if move not in visited_moves]

        if unvisited_moves: 
            # choose random unvisited child
            col = random.choice(unvisited_moves)

            # create child
            child_board = board.get_new_board(col, player_id)
            child: MCNode = MCNode(child_board, parent=node, move=col)
            node.children.append(child)

            # gameplay simulation
            result = self._simulate_game(child_board, player_id)

            # backprop
            self._set_values(child, result) 
            return   

        # select one of the children, if all have been simulated once
        uct_selected: MCNode = self._select_move(node)
        next_player = 2 if player_id == 1 else 1
        self._build_tree(uct_selected, next_player)  

    def _simulate_game(self, node_board: Board, player_id: int) -> int:
        """
        Args:
            node_board (Board): current board from which the gameplay is simulated with random moves
            player_id (int): the player who just made a move
        Return:
            int: to which result does this simulation lead to 
        """
        winner = winning(node_board.get_board_state(), self.game_n)
        # as long as there is no winner, keep on doing random moves
        if winner == 0:
            next_player = 2 if player_id == 1 else 1
    
            # random move of adversary player
            random_move = None
            while random_move is None:
                random_choice = random.choice(range(node_board.width))
                if node_board.is_valid(random_choice):
                    random_move = random_choice
                    next_child_board = node_board.get_new_board(random_move, next_player)
            return self._simulate_game(next_child_board, next_player)
        
        # return 1 for win, 0 for draw and -1 for loss
        if winner == self.player_id:
            return 1
        if winner == -1: # draw
            return 0
        else:
            return -1

    def _set_values(self, child: MCNode, result: int) -> None:
        """
        include the results in the nodes variables and backpropagate it upwards on the tree
        Args:
            child (MCNode): the node on which a gameplay was simulated
            result (int): result of the last simulation
        """
        child.visits += 1
        if result == 1: 
            child.total_reward += 1
        if result == -1:
            child.total_reward -= 1
        if child.parent != None:
            self._set_values(child.parent, result)
        return


    def _select_move(self, node: MCNode) -> MCNode:
        """
        choose based on UCT selection formula:
        w/n + c * sqrt(ln(N)/n)

        w: total reward with regard to wins and loses
        n: number of times selected node has been visited
        N: number of times parent node has been visited
        c: exploration parameter (hier = 1)
        """
        c: float = 2.0

        best_child = None
        best_value = -np.inf

        for child in node.children:
            n = child.visits
            w = child.total_reward
            N = node.visits

            temp = w / n + c * np.sqrt(np.log(N) / n)

            if temp > best_value:
                best_value = temp
                best_child = child

        return best_child


class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
        