
def partition(number):
    answer = set()
    answer.add((number, ))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add(tuple(sorted((x, ) + y)))
    return answer


def partitions(number):
    tmp = list(map(lambda y: (y[0], 0) if len(y) == 1 else y, list(filter(lambda x: len(x) <= 2, partition(int(number))))))
    result = []

    for t1, t2 in tmp:
        result.append((t1,t2))
        if t1 != t2:
            result.append((t2,t1))

    return result


class Cell:

    def __init__(self, val, part):
        self.val = round(val, 0)
        self.part = part

    def __str__(self):
        return "(" + str(self.val) + ", " + str(self.part) + ")"


def print_matrix(rows, cols, matrix):
    for r in range(rows):
        for c in range(cols):
            print(matrix[r][c].part, end='')
            print("--", end='')
        print("\n")


def get_optimized_reward(rewards_matrix, budgets):
    return combinatorial_optimization(rewards_matrix, budgets)[1]


def get_optimized_arms(rewards_matrix, budgets):
    return combinatorial_optimization(rewards_matrix, budgets)[0]


def combinatorial_optimization(_input, budgets):
    """
    In according to the sampled values solve a combinatorial problem
    that finds one arm per sub-campaign such that maximize the rewards
    and such that satisfies the given budget
    :param _input: NxM matrix, N number of sub-campaigns, M budgets, each value is the sampled value
    :param budgets: discretization of the budgets, list of values
    :return: list of arm idx (1 per sub-campaign) founded by the combinatorial algorithm
    """

    rows = len(_input)
    cols = len(_input[0])
    opt_matrix = [[Cell(0, (0, 0)) for _ in range(cols)] for _ in range(rows)]

    # initialize the first row
    for i, b in enumerate(budgets):
        opt_matrix[0][i] = Cell(_input[0][i], (0, int(b)))

    for campaign_idx in range(1, rows):
        for col, b in enumerate(budgets):
            # b -> current budget
            values = [(opt_matrix[campaign_idx-1][budgets.index(part[0])].val + _input[campaign_idx][part[1]], part) for part in partitions(b)]
            _max = values[0][0]
            _part = values[0][1]
            # select the partition that has the highest reward
            for v, p in values:
                if v >= _max:
                    _max, _part = v, p
            opt_matrix[campaign_idx][col] = Cell(_max, _part)

    # compute the best reward using all sub-campaigns
    best = 0
    best_part = opt_matrix[rows-1][0].part

    #print_matrix(rows, cols, opt_matrix)

    for cell in opt_matrix[rows-1]:
        v, p = cell.val, cell.part
        if v >= best:
            best, best_part = v, p

    arms_values = [best_part[1]]
    for i in range(1, rows-1)[::-1]:
        cur = opt_matrix[i][best_part[0]]
        _, p = cur.val, cur.part
        arms_values.append(p[1])
        best_part = p

    arms_values.append(best_part[0])
    return [budgets.index(val) for val in arms_values[::-1]], best
