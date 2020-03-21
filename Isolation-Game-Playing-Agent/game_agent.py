"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # getting opponent
    opponent = game.get_opponent(player)
    # obtaining locations
    playerMoves  = game.get_legal_moves(player)
    opponentMoves = game.get_legal_moves(opponent)

    # returning heuristic
    return float(len(playerMoves) - len(opponentMoves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Calculating center position of the game board
    mid_w, mid_h = game.height // 2 + 1, game.width // 2 + 1
    center_location = (mid_w, mid_h)

    # getting players location
    player_location  = game.get_player_location(player)
    # checking if player is the center location # returning heuristic1 with incentive
    if center_location == player_location:
        return custom_score(game, player)+100
    else: # returning heuristic1
        return custom_score(game, player)


def proximity(location1, location2):
    """
    Function return extra score as function of proximity between two positions.

    Parameters
    ----------
    location1, location2: tuple
        two tuples of integers (i,j) correspond two different positions on the board

    Returns
    ----------
    float
        The heuristic value of 100 for center of the board position and zero otherwise
    """
    return abs(location1[0]-location2[0])+abs(location1[1]-location2[1])


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    opponent            = game.get_opponent(player)
    player_location     = game.get_player_location(player)
    opponent_location   = game.get_player_location(opponent)
    playerMoves         = game.get_legal_moves(player)
    opponentMoves       = game.get_legal_moves(opponent)
    blank_spaces        = game.get_blank_spaces()

    board_size = game.width * game.height

    # size of local area
    localArea = (game.width + game.height)/4

    # condition that corresponds to later stages of the game
    if board_size - len(blank_spaces) > float(0.3 * board_size):
        # filtering out moves that are within local area
        playerMoves = [move for move in playerMoves if proximity(player_location, move) <= localArea]
        opponentMoves = [move for move in opponentMoves if proximity(opponent_location, move) <= localArea]

    return float(len(playerMoves) - len(opponentMoves))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation-Game-Playing-Agent agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # INVESTIGATE IF WE CAN USE A BETTER APPROACH -S Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def _timer(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def _end_of_game(self, game, depth):
        """
        helper method to see if we reached end of game or last depth
        """
        self._timer()
        legal_moves = game.get_legal_moves()
        if depth > 0 and len(legal_moves) != 0:
            return False
        return True

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation-Game-Playing-Agent game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        best_score = float('-inf')
        self.best_score = legal_moves[0]
        for move in legal_moves:
            self._timer()
            forecast = game.forecast_move(move)
            next_score = self._minimum_value(forecast, depth - 1)

            if next_score > best_score:
                best_score = next_score
                self.best_score = move
        return self.best_score

    def _minimum_value(self, game, depth):
        """
        function MIN-VALUE(state) returns a utility value
         if TERMINAL-TEST(state) then return UTILITY(state)
         v ← ∞
         for each a in ACTIONS(state) do
           v ← MIN(v, MAX-VALUE(RESULT(state, a)))
         return v
        """
        self._timer()
        if depth == 0:
            return self.score(game, self)
        min_value = float("inf")
        for move in game.get_legal_moves():
            self._timer()
            min_value_score = self._maximum_value(game.forecast_move(move), depth - 1)
            if min_value_score < min_value:
                min_value = min_value_score
        return min_value

    def _maximum_value(self, game, depth):
        """
        function MAX-VALUE(state) returns a utility value
         if TERMINAL-TEST(state) then return UTILITY(state)
         v ← −∞
         for each a in ACTIONS(state) do
           v ← MAX(v, MIN-VALUE(RESULT(state, a)))
         return v
        """
        self._timer()
        if depth == 0:
            return self.score(game, self)
        max_value = float("-inf")
        for move in game.get_legal_moves():
            self._timer()
            max_value_score = self._minimum_value(game.forecast_move(move), depth - 1)
            if max_value_score > max_value:
                max_value = max_value_score
        return max_value


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    def _timer(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # 5: return something in case of timeout
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        self.best_move = (-1, -1)
        depth = 1
        try:
            while (True):
                move = self.alphabeta(game, depth)
                if move is not (-1, -1):
                    # if move is not the best move return the move
                    self.best_move = move
                depth += 1

                if self.time_left() < self.TIMER_THRESHOLD:
                    return self.best_move
        except SearchTimeout:
            # return best move with best possible depth
            return self.best_move

    def end_of_game(self, game, depth):
        """
        helper method to see if we reached end of game or last depth
        """
        self._timer()
        legal_moves = game.get_legal_moves()
        if depth > 0 and len(legal_moves) != 0:
            return False
        return True

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation-Game-Playing-Agent game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        """
        function ALPHA-BETA-SEARCH(state) returns an action
         v ← MAX-VALUE(state, −∞, +∞)
         return the action in ACTIONS(state) with value v
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        self.best_current_move = legal_moves[0]
        for move in legal_moves:
            self._timer()
            forecast = self._min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if forecast > alpha:
                alpha = forecast
                self.best_current_move = move
        return self.best_current_move

    def _min_value(self, game, depth, alpha, beta):
        """
        function MIN-VALUE(state, α, β) returns a utility value
         if TERMINAL-TEST(state) then return UTILITY(state)
         v ← +∞
         for each a in ACTIONS(state) do
           v ← MIN(v, MAX-VALUE(RESULT(state, a), α, β))
           if v ≤ α then return v
           β ← MIN(β, v)
         return v
        """
        if depth == 0:
            return self.score(game, self)
        for move in game.get_legal_moves():
            self._timer()
            forecast_value = self._max_value(game.forecast_move(move), depth - 1, alpha, beta)
            if forecast_value < beta:
                beta = forecast_value
                if beta <= alpha:
                    break
        return beta

    def _max_value(self, game, depth, alpha, beta):
        """
            function MAX-VALUE(state, α, β) returns a utility value
             if TERMINAL-TEST(state) then return UTILITY(state)
             v ← −∞
             for each a in ACTIONS(state) do
               v ← MAX(v, MIN-VALUE(RESULT(state, a), α, β))
               if v ≥ β then return v
               α ← MAX(α, v)
             return v
        """
        if depth == 0:
            return self.score(game, self)
        for move in game.get_legal_moves():
            self._timer()
            forecast_value = self._min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if forecast_value > alpha:
                alpha = forecast_value
                if alpha >= beta:
                    break
        return alpha






