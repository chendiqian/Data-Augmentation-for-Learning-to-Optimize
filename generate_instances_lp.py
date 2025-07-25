from itertools import combinations

import numpy as np
import scipy
import scipy.sparse


def generate_setcover(nrows, ncols, nnzrs, rng):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    rng: numpy.random.RandomState
        Random number generator
    """

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows)  # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i + n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows],
                                          assume_unique=True)
            indices[nrows:i + n] = rng.choice(remaining_rows, size=i + n - nrows,
                                              replace=False)

        i += n
        indptr.append(i)

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
        (np.ones(len(indices), dtype=float), indices, indptr),
        shape=(nrows, ncols)).toarray().T

    # objective coefficients
    c = rng.rand(A.shape[1]).astype(np.float32)

    A = -A
    b = np.ones(A.shape[0], dtype=np.float32) * -1

    return A, b, c


class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability, random):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        all_edges = np.vstack(np.triu_indices(number_of_nodes, k=1))
        selected_edges = random.binomial(1, np.ones(all_edges.shape[1]) * edge_probability, size=None)
        selected_edges = np.where(selected_edges)[0]
        selected_edges = all_edges[:, selected_edges].T

        degrees = np.zeros(number_of_nodes, dtype=int)
        node_id, cnts = np.unique(selected_edges, return_counts=True)
        degrees[node_id] = cnts

        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
        for n1, n2 in selected_edges:
            edges.add((n1, n2))
            neighbors[n1].add(n2)
            neighbors[n2].add(n1)

        # for edge in combinations(np.arange(number_of_nodes), 2):
        #     if random.uniform() < edge_probability:
        #         edges.add(edge)
        #         degrees[edge[0]] += 1
        #         degrees[edge[1]] += 1
        #         neighbors[edge[0]].add(edge[1])
        #         neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def generate_indset(graph, nnodes):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    """
    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(nnodes):
        if node not in used_nodes:
            inequalities.add((node,))

    c = -np.random.rand(len(graph))
    A, b = np.zeros((len(inequalities), len(graph))), np.ones(len(inequalities))
    for ineq, group in enumerate(inequalities):
        A[ineq, sorted(group)] = 1.

    return A, b, c


def generate_cauctions(n_items, n_bids, rng, min_value=1, max_value=100,
                    value_deviation=0.5, add_item_prob=0.7, max_n_sub_bids=5,
                    additivity=0.2, budget_factor=1.5, resale_factor=0.5,
                    integers=False, warnings=False):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    rng : numpy.random.RandomState
        A random number generator.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, add_item_prob, rng):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return rng.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * rng.rand(n_items)

    # item compatibilities
    compats = np.triu(rng.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = rng.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = rng.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while rng.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, rng)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, rng)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    bids_per_item = [[] for item in range(n_items + n_dummy_items)]

    c = np.zeros(len(bids))
    for i, bid in enumerate(bids):
        bundle, price = bid
        c[i] = -price
        for item in bundle:
            bids_per_item[item].append(i)

    A = []
    for item_bids in bids_per_item:
        if item_bids:
            row = np.zeros(len(c))
            row[item_bids] = 1
            A.append(row)
    A = np.array(A)
    b = np.ones(A.shape[0])

    return A, b, c


def generate_capacited_facility_location(n_customers, n_facilities, ratio, rng):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    rng : numpy.random.RandomState
        A rng number generator.
    """
    demands = rng.randn(n_customers)
    demands[demands < 0] = 0.
    capacities = rng.rand(n_facilities)

    total_demand = demands.sum()
    total_capacity = capacities.sum()
    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    # capacities = capacities.astype(int)

    c = (rng.rand(n_facilities * n_customers + n_facilities) + 1) / 2

    A, b = [], []
    # sum_j x_ij = 1
    # fac j provides to customers i
    for i in range(n_customers):
        row = np.zeros((n_customers, n_facilities))
        row[i, :] = 1
        row = np.concatenate([row.flatten(), np.zeros(n_facilities)])
        A.append(row)
        b.append(1)

    # A_eq = np.array(A)
    # b_eq = np.array(b)
    A.append(np.concatenate([np.zeros(n_customers * n_facilities), capacities]))
    b.append(total_demand)

    # A, b = [], []
    # sum_i demand_i * x_ij - volume_j * y_j <= 0
    for j in range(n_facilities):
        row = np.zeros((n_customers, n_facilities))
        row[:, j] = -demands
        row = np.concatenate([row.flatten(), capacities])
        A.append(row)
        b.append(0)

    ## sum_j cap_j >= sum_i demand_i
    # A.append(np.concatenate([np.zeros(n_customers * n_facilities), capacities]))
    # b.append(total_demand)

    # x_ij < y_j
    for i in range(n_customers):
        for j in range(n_facilities):
            row1 = np.zeros((n_customers, n_facilities))
            row1[i, j] = -1
            row2 = np.zeros(n_facilities)
            row2[j] = 1
            row = np.concatenate([row1.flatten(), row2])
            A.append(row)
            b.append(0)

    A_ub, b_ub = -np.array(A), -np.array(b)
    return A_ub, b_ub, c
