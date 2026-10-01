from heuristics import Heuristic, SimpleHeuristic
from players import PlayerController, HumanPlayer, MinMaxPlayer, AlphaBetaPlayer, MonteCarloPlayer
from board import Board
from typing import List
from utils import winning

def start_game(game_n: int, board: Board, players: List[PlayerController]) -> int:
    """Starting a game and handling the game logic

    Args:
        game_n (int): n in a row required to win
        board (Board): board to play on
        players (List[PlayerController]): players of the game

    Returns:
        int: id of the winning player, or -1 if the game ends in a draw
    """
    print('Start game!')
    current_player_index: int = 0 # index of the current player in the players list
    winner: int = 0

    # Main game loop
    while winner == 0:
        print(board.get_board_state())
        print(board)
        current_player: PlayerController = players[current_player_index]
        move: int = current_player.make_move(board)

        while not board.play(move, current_player.player_id):
            move = current_player.make_move(board)

        current_player_index = 1 - current_player_index
        winner = winning(board.get_board_state(), game_n)

    # Printing out winner, final board and number of evaluations after the game 
    print(board)

    if winner < 0:
        print('Game is a draw!')
    else:
        print(f'Player {current_player} won!')

    for p in players:
        print(f'Player {p} evaluated a boardstate {p.get_eval_count()} times!')

    return winner


def _ask_player(number: int) -> int:
    try:
        choice: int = int(input(
            f"Choose Player {number}: \n"
            "1 - Human\n"
            "2 - MinMax\n"
            "3 - AlphaBeta\n"
            "4 - MonteCarlo\n"
        ))

        assert 1 <= choice <= 4
        print("You chose", choice)
        return choice

    except ValueError:
        print("Please enter a valid number.\n")
        return _ask_player(number)

    except AssertionError:
        print("Please enter 1,2,3 or 4.\n")
        return _ask_player(number)

def get_players(game_n: int) -> List[PlayerController]:
    """Gets the two players

    Args:
        game_n (int): n in a row required to win

    Raises:
        AssertionError: if the players are incorrectly initialised

    Returns:
        List[PlayerController]: list with two players
    """
    heuristic1: Heuristic = SimpleHeuristic(game_n)
    heuristic2: Heuristic = SimpleHeuristic(game_n)

    human1: PlayerController = HumanPlayer(1, game_n, heuristic1)
    human2: PlayerController = HumanPlayer(2, game_n, heuristic2)

    # DONE: Implement other PlayerControllers (MinMaxPlayer and AlphaBetaPlayer)
    depth = 6
    # MinMaxPlayer
    
    minmaxplayer1: PlayerController = MinMaxPlayer(1, game_n, depth, heuristic1)
    minmaxplayer2: PlayerController = MinMaxPlayer(2, game_n, depth, heuristic2)

    # AlphaBetaPlayer

    alphabetaplayer1: PlayerController = AlphaBetaPlayer(1, game_n, depth, heuristic1)
    alphabetaplayer2: PlayerController = AlphaBetaPlayer(2, game_n, depth, heuristic2)

    # MCPlayer

    max_simulations = 1000

    mcplayer1: PlayerController = MonteCarloPlayer(1, game_n, max_simulations, heuristic1)
    mcplayer2: PlayerController = MonteCarloPlayer(2, game_n, max_simulations, heuristic2)

    player1value = _ask_player(1)
    player2value = _ask_player(2)

    match player1value:
        case 1:
            player1 = human1
        case 2:
            player1 = minmaxplayer1
        case 3: 
            player1 = alphabetaplayer1
        case 4: 
            player1 = mcplayer1
        case _:
            print("Invalid player") 

    match player2value:
        case 1:
            player2 = human2
        case 2:
            player2 = minmaxplayer2
        case 3: 
            player2 = alphabetaplayer2
        case 4: 
            player2 = mcplayer2
        case _:
            print("Invalid player") 

    players: List[PlayerController] = [player1, player2]

    assert players[0].player_id in {1, 2}, 'The player_id of the first player must be either 1 or 2'
    assert players[1].player_id in {1, 2}, 'The player_id of the second player must be either 1 or 2'
    assert players[0].player_id != players[1].player_id, 'The players must have an unique player_id'
    assert players[0].heuristic is not players[1].heuristic, 'The players must have an unique heuristic'
    assert len(players) == 2, 'Not the correct amount of players'

    return players


if __name__ == '__main__':
    game_n: int = 4 # n in a row required to win
    width: int = 7  # width of the board
    height: int = 6 # height of the board

    # Check whether the game_n is possible
    assert 1 < game_n <= min(width, height), 'game_n is not possible'

    board: Board = Board(width, height)
    start_game(game_n, board, get_players(game_n))
    