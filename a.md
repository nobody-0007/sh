#### A*
```
def astar_algorithm(start_node, goal_node):
    open_set = {start_node}
    closed_set = set()
    cost = {start_node: 0}
    parent = {start_node: None}

    while open_set:
        # Find node in open_set with the lowest f(n) = g(n) + h(n)
        curr = min(open_set, key=lambda node: cost.get(node, float('inf')) + heuristic(node))

        if curr == goal_node:
            return reconstruct_path(parent, start_node, goal_node, cost)

        open_set.remove(curr)
        closed_set.add(curr)

        neighbors = get_neighbors(curr)
        if neighbors is None:
            continue

        for neighbor, weight in neighbors:
            if neighbor in closed_set:
                continue

            total_cost = cost[curr] + weight

            if neighbor not in open_set:
                open_set.add(neighbor)

            if total_cost < cost.get(neighbor, float('inf')):
                parent[neighbor] = curr
                cost[neighbor] = total_cost

    print("Path does not exist!")
    return None

def reconstruct_path(parent, start, goal, cost):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    print(f"Path found: {path}")
    print(f"Total total_cost: {cost[goal]}")
    return path

def get_neighbors(node):
    return Graph_nodes.get(node, None)

def heuristic(node):
    H_dist = {
        'A': 11,
        'B': 6,
        'C': 99,
        'D': 1,
        'E': 5,
        'G': 0,
    }
    return H_dist.get(node, float('inf'))

Graph_nodes = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1), ('G', 9)],
    'C': None,
    'E': [('D', 6)],
    'D': [('G', 1)],
}

astar_algorithm('A', 'G')
```

#### GBFS
```
H={'a':12,'b':4,'c':7,'d':3,'e':8,'f':2,'g':0,'h':4,'i':9,'s':13}
graph={
    's':[(3,'a'),(2,'b')],
    'a':[(4,'c'),(1,'d')],
    'b':[(3,'e'),(1,'f')],
    'c':[],
    'd':[],
    'e':[(5,'h')],
    'f':[(2,'i'),(3,'g')],
    'g':[],
    'h':[],
    'i':[]
}
initial=input("Enter the initial state: ")
goal=input("Enter the goal state: ")

def gbfs(initial,H,goal):
    frontier=[]
    explored=[]
    frontier.append(initial)
    while frontier:
        node=frontier.pop(0)
        explored.append(node)
        if node==goal:
            return explored
        adj=graph[node]
        for cost, child in adj:  # Unpack the tuple into cost and child
            if child not in frontier and child not in explored:
                frontier.append(child)  # Append only the child (node)
        frontier.sort(key=lambda child:H[child])  # Sort by heuristic value
ans=gbfs(initial,H,goal)
print("The optimal path is: ",ans)
```

#### TSP
```
import itertools

# Define the cities and distances
cities = ['a', 'b', 'c', 'd']
distances = {
    ('a', 'b'): 20,
    ('a', 'c'): 42,
    ('a', 'd'): 35,
    ('b', 'c'): 30,
    ('b', 'd'): 34,
    ('c', 'd'): 12
}

# Function to calculate the total cost of a route
def calculate_cost(route):
    total_cost = 0
    n = len(route)
    for i in range(n):
        current_city = route[i]
        next_city = route[(i + 1) % n]  # Wrap around to the start of the route
        # Look up the distance in both directions
        if (current_city, next_city) in distances:
            total_cost += distances[(current_city, next_city)]
        else:
            total_cost += distances[(next_city, current_city)]
    return total_cost

# Generate all permutations of the cities
all_permutations = itertools.permutations(cities)

# Initialize variables to track the minimum cost and corresponding route
min_cost = float('inf')
optimal_route = None

# Iterate over all permutations and calculate costs
for perm in all_permutations:
    cost = calculate_cost(perm)
    if cost < min_cost:
        min_cost = cost
        optimal_route = perm + ('a',)

# Print the optimal route and its cost
print(f"Optimal Route: {optimal_route}")
print(f"Total Cost: {min_cost}")
```

#### BFS & DFS
```
graph={ 'A':['B','C','D'],
        'B':['D'],
        'C':['D','L'],
        'D':['L'],
        'E':['A','B','F'],
        'F':['L'],
        'L':['M'],
        'M':[]
}
initial=input("Enter the initial state: ")  
goal=input("Enter the goal state: ") 
def BFS(initial,goal):
    frontier=[]
    explored=[]
    frontier.append(initial)
    while frontier:
        node=frontier.pop(0)
        explored.append(node)
        if node==goal:
            return explored
        
        adj=graph[node]
        for child in adj:
            if child not in frontier and child not in explored:
                frontier.append(child)
ans=BFS(initial,goal)
print("The BFS is: ",ans)       

def DFS(initial,goal):
    frontier=[]
    explored=[]
    frontier.append(initial)
    while frontier:
        node=frontier.pop(0)
        explored.append(node)
        if node==goal:
            return explored
        adj=graph[node]
        for child in adj:
            if child not in frontier and child not in explored:
                frontier.append(child)
                frontier.reverse()
ans=DFS(initial,goal)
print("The Depth first traversal is: ",ans)
```

#### N-QUEENS
```
def is_safe(board, row, col):
    # Check if no other queen can attack this position
    for i in range(col):
        if board[i] == row or \
           board[i] - i == row - col or \
           board[i] + i == row + col:
            return False
    return True

def solve_n_queens(n):
    def backtrack(col):
        if col == n:
            sol.append(list(board[:]))
        else:
            for row in range(n):
                if is_safe(board, row, col):
                    board[col] = row
                    backtrack(col + 1)

    board = [-1] * n
    sol = []
    backtrack(0)
    return sol

def visualize_solution(sol):
    n = len(sol)
    chessboard = [["." for _ in range(n)] for _ in range(n)]

    for col, row in enumerate(sol):
        chessboard[row][col] = "Q"

    for row in chessboard:
        print(" ".join(row))

if __name__ == "__main__":
    n = 4 # No of queens
    sol = solve_n_queens(n)
    
    if sol:
        print(f"Found {len(sol)} Solutions:")
        for i, solution in enumerate(sol):
            print(f"\nSolution {i + 1}:")
            visualize_solution(solution)
            print("Queens placed in each column:", [row + 1 for row in solution])
    else:
        print("No Solution found.")
```

#### TIC-TAC-TOE
```
% Start the game
play :-
    Board = [empty, empty, empty,
             empty, empty, empty,
             empty, empty, empty],
    display_board(Board),
    play_turn(Board, x).

% Display the board
display_board(Board) :-
    nl,
    display_row(Board, 0),
    write('---+---+---'), nl,
    display_row(Board, 3),
    write('---+---+---'), nl,
    display_row(Board, 6),
    nl.

display_row(Board, Index) :-
    nth0(Index, Board, C1),
    I2 is Index + 1, nth0(I2, Board, C2),
    I3 is Index + 2, nth0(I3, Board, C3),
    display_cell(C1), write('|'),
    display_cell(C2), write('|'),
    display_cell(C3), nl.

display_cell(empty) :- write('   ').
display_cell(x) :- write(' X ').
display_cell(o) :- write(' O ').

% Play a turn
play_turn(Board, Player) :-
    choose_move(Board, Move),
    valid_move(Board, Move),
    move(Board, Move, Player, NewBoard),
    display_board(NewBoard),
    ( win(Player, NewBoard) ->
        write(Player), write(' wins!'), nl;
      draw(NewBoard) ->
        write('It\'s a draw!'), nl;
      switch_player(Player, NextPlayer),
      play_turn(NewBoard, NextPlayer)
    ).


% Choose move
choose_move(Board, Move) :-
    write('Enter your move (1-9): '),
    read(Input),
    Move is Input - 1.

% Validate move
valid_move(Board, Move) :-
    nth0(Move, Board, empty).

% Make move
move(Board, Move, Player, NewBoard) :-
    nth0(Move, Board, empty, Rest),
    nth0(Move, NewBoard, Player, Rest).

% Switch players
switch_player(x, o).
switch_player(o, x).

% Check draw
draw(Board) :-
    \+ member(empty, Board).

% Check win
win(Player, Board) :-
    win_pos(Pos),
    check_line(Pos, Board, Player).

win_pos([0,1,2]).
win_pos([3,4,5]).
win_pos([6,7,8]).
win_pos([0,3,6]).
win_pos([1,4,7]).
win_pos([2,5,8]).
win_pos([0,4,8]).
win_pos([2,4,6]).

check_line([A,B,C], Board, Player) :-
    nth0(A, Board, Player),
    nth0(B, Board, Player),
    nth0(C, Board, Player).
```

#### Monkey-Banana Problem
```
% initial state: Monkey is at door,
%                Monkey is on floor,
%                Box is at window,
%                Monkey doesn't have banana.
%

% prolog structure: structName(val1, val2, ... )

% state(Monkey location in the room, Monkey onbox/onfloor, box location, has/hasnot banana)


% legal actions
do( state(middle, onbox, middle, hasnot),   % grab banana
    grab,
    state(middle, onbox, middle, has) ).

do( state(L, onfloor, L, Banana),           % climb box
    climb,
    state(L, onbox, L, Banana) ).

do( state(L1, onfloor, L1, Banana),         % push box from L1 to L2
    push(L1, L2),
    state(L2, onfloor, L2, Banana) ).

do( state(L1, onfloor, Box, Banana),        % walk from L1 to L2
    walk(L1, L2),
    state(L2, onfloor, Box, Banana) ).


% canget(State): monkey can get banana in State
canget(state(_, _, _, has)).                % Monkey already has it, goal state

canget(State1) :-                           % not goal state, do some work to get it
      do(State1, Action, State2),           % do something (grab, climb, push, walk)
      canget(State2).                       % canget from State2

% get plan = list of actions
canget(state(_, _, _, has), []).            % Monkey already has it, goal state

canget(State1, Plan) :-                     % not goal state, do some work to get it
      do(State1, Action, State2),           % do something (grab, climb, push, walk)
      canget(State2, PartialPlan),          % canget from State2
      add(Action, PartialPlan, Plan).       % add action to Plan

add(X,L,[X|L]).

%-------------------OutPut Query------------------------>
% ?- canget(state(atdoor, onfloor, atwindow, hasnot), Plan).
% Plan = [walk(atdoor, atwindow), push(atwindow, middle), climb, grasp]
% Yes

% ?- canget(state(atwindow, onbox, atwindow, hasnot), Plan ).
% No

% ?- canget(state(Monkey, onfloor, atwindow, hasnot), Plan).
% Monkey = atwindow
% Plan = [push(atwindow, middle), climb, grasp]
% Yes
```

#### C to F
```
c_to_f(C,F) :-
    F is C*9/5+32.
freezing(F) :-
    F=<32.
```

#### FACTORIAL & FIBIONACCI
```
factorial(0, 1).

factorial(N, Res) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, Res1),
    Res is N * Res1.

fib(0, 0).  
fib(1, 1).  

fib(N, F) :-
    N > 1,  
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),  
    fib(N2, F2),  
    F is F1 + F2.  
```

