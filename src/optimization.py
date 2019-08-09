import numpy as np


def partition(number):
    answer = set()
    answer.add((number, ))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add(tuple(sorted((x, ) + y)))
    return answer


def partitions(number):
    tmp = list(map(lambda y: (y[0], 0) if len(y)==1 else y,list(filter(lambda x: len(x) <= 2, partition(number)))))
    result = []

    for t1, t2 in tmp:
        result.append((t1,t2))
        if t1 != t2:
            result.append((t2,t1))

    return result


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
    opt_matrix = np.zeros(shape=(rows, cols))

    # initialize the first row
    opt_matrix[0] = np.array((v,) for v in _input[0])

    for campaign_idx in range(1, rows):
        for col, b in enumerate(budgets):
            # b -> current budget
            values = [(opt_matrix[campaign_idx-1, budgets.index(part[0])] + _input[campaign_idx][part[1]], part) for part in partitions(b)]
            _max = 0
            _part = ()
            # select the partition that has the highest reward
            for v, p in values:
                if v > _max:
                    _max, _part = v, p
            opt_matrix[campaign_idx, col] = (_max, _part)

    # compute the best reward using all sub-campaigns
    best = 0
    best_part = ()
    for v, p in opt_matrix[rows-1]:
        if v > best:
            best, best_part = v, p

    arm_idx = [best_part[1]]
    for i in range(0, rows-1)[::-1]:
        _, p = opt_matrix[i, best_part[0]]
        arm_idx.append(p[1])
        best_part = p

    return [budgets.index(arm) for arm in arm_idx[::-1]]
