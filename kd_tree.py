
#=======================================================================
# Imports
import sys, operator, heapq

#=======================================================================
# Constants
DEFAULT_BUCKET_SIZE = 1

#=======================================================================
class KDTree(object):
    """A k-dimensional tree for providing approximate nearest neighbor searches.

    The tree is built out of KDTree.Node objects. Data is stored only
    at the leaf nodes. There is a bucket_size parameter to the
    constructor for specifying the maximum number of data entries to
    store at each leaf. Each of these entries is a KDTree.Entry object
    which can store any type of user data.

    --------------------------------------------------------------------
    Set up some helper functions.

    >>> import math, random

    >>> def flatten(tree):  # Also sorts the resulting list.
    ...     return sorted(tree.flatten())

    >>> def rand():
    ...     return random.randint(-sys.maxint - 1, sys.maxint)

    >>> def rand_point(num_dims):
    ...     return [rand() for i in xrange(num_dims)]

    >>> def find_random(tree, num, num_dims, delete):
    ...     return [tree.find_nearest_neighbor(rand_point(num_dims), delete).flatten() for i in xrange(num)]

    Define this in case we are using a Python version before 2.5:
    >>> def all(iterable):
    ...     for element in iterable:
    ...         if not element:
    ...             return False
    ...     return True
    
    --------------------------------------------------------------------
    Create an empty tree and call all of the methods on it.

    >>> tree = KDTree([])
    >>> print tree.find_nearest_neighbor(rand_point(3))
    None

    >>> print tree.find_nearest_neighbor(rand_point(10), delete=True)
    None

    >>> print tree
    []

    --------------------------------------------------------------------
    Create a tree with one entry (of five dimensions) and call all of
    the methods on it.

    >>> tree = KDTree.create( [ ((7, 8, 9, 1000000000000L, 11.5), 12) ] )
    >>> print tree.find_nearest_neighbor((1, 2, 3))
    ((7, 8, 9, 1000000000000L, 11.5), 12)

    >>> print tree
    [((7, 8, 9, 1000000000000L, 11.5), 12)]

    >>> print tree.find_nearest_neighbor((5, 6, 7), delete=True)
    ((7, 8, 9, 1000000000000L, 11.5), 12)

    >>> print tree.find_nearest_neighbor((7, 8, 9), delete=True)
    None

    >>> print tree
    []

    --------------------------------------------------------------------
    Create a tree with 100000 entries (of three dimensions) with a
    bucket size of 32.

    >>> num_dims = 3
    >>> data = [(rand_point(num_dims), rand()) for i in xrange(10000)]
    >>> tree = KDTree.create(data, bucket_size=32)
    >>> data.sort()

    Make sure that all of the entries were added.
    >>> flatten(tree) == data
    True

    Do a find_nearest_neighbor search of a random point once for each
    entry that is in the tree.  There is nothing to check for this
    since we don't delete the entries, and we can't easily predict
    what values will be returned.
    >>> results = find_random(tree, len(data), num_dims, delete=False)

    The tree should still be the same.
    >>> flatten(tree) == data
    True

    Do a find_nearest_neighbor search with delete of a random point
    once for each entry in the tree.
    >>> sorted(find_random(tree, len(data), num_dims, delete=True)) == data
    True
    
    Since we deleted all entries, the tree should now be empty.
    >>> print tree
    []

    --------------------------------------------------------------------
    Create a tree with increasing data, so that it is easy to test
    whether find_nearest_neighbor is working.

    >>> num_dims = 3
    >>> data = [((i, i, i), i) for i in xrange(1000, 11000, 1000)]

    Shuffle the data first to make sure that the order that we use to
    create the tree doesn't matter.
    >>> shuffled_data = list(data)
    >>> random.shuffle(shuffled_data)
    >>> tree = KDTree.create(shuffled_data)

    Do a search for a point close to each of the points in the
    tree. Since the search is an approximate nearest neighbor search,
    we are not guaranteed to always find the point closest to the one
    that searched for, but we check that the distance is close enough.
    >>> all(map(lambda x, y: math.sqrt(euclidean_distance(x, y)) / 1000 < 2, \
            [tree.find_nearest_neighbor((entry[0][0] - 999, entry[0][1] - 999, entry[0][2] - 999), delete=False).point for entry in data], \
            [e[0] for e in data]))
    True

    Do a search with delete from the point (0, 0, 0) once for each
    entry in the tree. The resulting values should move through each
    of the points in the data list in order.
    >>> [tree.find_nearest_neighbor((0, 0, 0), delete=True).flatten() for i in xrange(len(data))] == data
    True

    Since we deleted all entries, the tree should now be empty.
    >>> print tree
    []

    Add tests with different bucket sizes and error thresholds.
    Add tests with higher numbers of dimensions.
    """

    #-------------------------------------------------------------------
    class Entry(object):
        """Represents a point and data pair.

        The bucket field in the Node is a list of these.
        """
        def __init__(self, point, data):
            """Create the Entry object.

            point must be a tuple or list of numbers
            data can be any type
            """
            self.point = point
            self.data = data
            self.is_deleted = False

        def delete(self):
            self.is_deleted = True

        def flatten(self):
            return (self.point, self.data)

        def __str__(self):
            return str(self.flatten())

    #-------------------------------------------------------------------
    class Node(object):
        """Represents a node in the kd-tree

        We don't store any data at the non-leaf nodes, so bucket is
        always empty for those nodes.
        """
        def __init__(self, point, left_child, right_child, entries=[]):
            self.point = point
            self.left_child = left_child
            self.right_child = right_child
            self.bucket = entries

    #-------------------------------------------------------------------
    def __init__(self, point_list, bucket_size=DEFAULT_BUCKET_SIZE):
        """Construct a kd-tree.

        point_list must be a list or tuple of KDTree Entry objects. It
        may be an empty list or empty tuple which creates an empty tree.

        bucket_size is used to specify the maximum number of entries
        stored at each leaf node.

        We determine the number of dimensions by looking at the first
        point in the list. All of the points must have the same number
        of dimensions, but we don't check to make sure of that.
        """

        if point_list:
            num_dimensions = len(point_list[0].point)
        else:
            num_dimensions = 0  # This value doesn't matter.

        def _create_kdtree(point_list, bucket_size):
            # Handle the case when the list is empty.
            if not point_list:
                return None

            # If the points can all fit in one bucket, then we are
            # done.
            if len(point_list) <= bucket_size:
                return KDTree.Node(point_list[0].point, None, None, list(point_list))

            # We split on the dimension of the hyper-cube with the
            # longest edge. This is the dimension that has the
            # greatest difference between the maximum and minimum
            # values in it.
            #
            # 1. Create a generator for the list of pairs of
            #    (min-value, max-value) for each dimension.
            # 2. Create a generator for the list containing the
            #    differences between the max and min values for each
            #    dimension.
            # 3. Find the index of the maximum entry in that list, and
            #    use that as the pivot dimension.
            min_max_generator = \
                (find_min_max((entry.point[dim] for entry in point_list))[0]
                 for dim in xrange(num_dimensions))
            difference_generator = \
                (max - min for min, max in min_max_generator)
            dimension = find_min_max(difference_generator)[1][1]

            # It is actually faster to sort the list to find the
            # median than to find the median and then partition the
            # list ourselves even though the sort is O(n log2 n) and
            # the median finding and partitioning is expected O(n),
            # probably because the sort is implemented in C. This was
            # determined by experimenting with our typical data sets.
            point_list.sort(key=lambda x: x.point[dimension])
            median_index = len(point_list) // 2  # Choose median

            # Create node and construct subtrees. We include the
            # median value in the right child.
            return KDTree.Node(point_list[median_index].point,
                               _create_kdtree(point_list[:median_index],
                                              bucket_size),
                               _create_kdtree(point_list[median_index:],
                                              bucket_size))

        self.top = _create_kdtree(point_list, bucket_size)

    #-------------------------------------------------------------------
    @staticmethod
    def create(data_list, bucket_size=DEFAULT_BUCKET_SIZE):
        "Static method for creating a tree from a list of (point, data) pairs"
        return KDTree([KDTree.Entry(*entry) for entry in data_list])

    #-----------------------------------------------------------------------
    def find_nearest_neighbor(self, target, delete=False, error_threshold=1):
        """Returns the approximate nearest entry to the target point.

        A value of True for the delete parameter is used to specify
        that the element found should then be deleted from the
        tree. This can be used to perform nearest neighbor searches
        without repeats.

        The error_threshold parameter is used to speed up the search
        at the expense of getting less accurate values.

        Returns the entry object for the point found or None if the
        tree is empty.
        """
        nearest_index, nearest_node = \
            self._find_nearest_neighbor_aux(self.top, target, error_threshold)

        # Handle the case where the tree is empty.
        if not nearest_node:
            return None

        nearest_entry = nearest_node.bucket[nearest_index]

        if delete:
            # Adjust the list so that we delete the element at the end
            # of it instead of an element in the middle for
            # efficiency.
            nearest_node.bucket[nearest_index] = nearest_node.bucket[-1]
            del nearest_node.bucket[-1]

        return nearest_entry

    #-----------------------------------------------------------------------
    @staticmethod
    def _find_nearest_neighbor_aux(top_node, target, error_threshold):
        """
        This is a helper method that does most of the work for the
        find_nearest_neighbor method.

        Retuns (nearest_index, nearest_node) or (None, None) if the
        tree is empty.

        This is the way that deletes are handled. When an entry is
        deleted, it is just marked as deleted, but it isn't removed
        from the tree at that point. The next time that the entry is
        encountered during a nearest neighbor search, it is removed
        from the bucket list in the Node that contains it. If the
        bucket list becomes empty at that point, that node will be
        deleted the next time that it is encountered during a nearest
        neighbor search. Whenever a node is encountered that has no
        children and an empty bucket list during a nearest neighbor
        search, that node is deleted (but its parent will not be
        deleted until the next search even if it has no children
        and an empty bucket list at that point).
        """
        if not top_node:
            return (None, None)

        heap = []
        curr_element = (euclidean_distance(top_node.point, target), top_node)

        # None for the distance means infinity.
        nearest_node = nearest_index = nearest_distance = None

        while True:
            if not curr_element:
                if not heap:
                    # This case is needed only when we delete entries,
                    # and we have only one entry in the tree because
                    # there is nothing to pop in that case.
                    return (nearest_index, nearest_node)
                curr_element = heapq.heappop(heap)
            curr_distance = curr_element[0]
            curr_node = curr_element[1]
            if (nearest_distance is not None and
                (1 + error_threshold) * curr_distance >= nearest_distance):
                return (nearest_index, nearest_node)

            # If the left child has no children of its own, and its
            # bucket is empty, then set it to None to mark it as
            # deleted. Then do the same check for the right child.
            if (curr_node.left_child and
                not curr_node.left_child.bucket and
                not curr_node.left_child.left_child and
                not curr_node.left_child.right_child):
                curr_node.left_child = None
            if (curr_node.right_child and
                not curr_node.right_child.bucket and
                not curr_node.right_child.left_child and
                not curr_node.right_child.right_child):
                curr_node.right_child = None

            if curr_node.left_child and curr_node.right_child:
                # This is not a leaf node and both children are present.
                left_distance = \
                    euclidean_distance(curr_node.left_child.point, target)
                right_distance = \
                    euclidean_distance(curr_node.right_child.point, target)
                if left_distance <= right_distance:
                    nearer_node = curr_node.left_child
                    farther_node = curr_node.right_child
                    nearer_distance = left_distance
                    farther_distance = right_distance
                else:
                    nearer_node = curr_node.right_child
                    farther_node = curr_node.left_child
                    nearer_distance = right_distance
                    farther_distance = left_distance
                curr_element = (nearer_distance, nearer_node)
                heapq.heappush(heap, (farther_distance, farther_node))

            elif curr_node.left_child:
                # This is not a leaf, but only the left node is present.
                curr_element = \
                    (euclidean_distance(curr_node.left_child.point, target),
                     curr_node.left_child)

            elif curr_node.right_child:
                # This is not a leaf, but only the right node is present.
                curr_element = \
                    (euclidean_distance(curr_node.right_child.point, target),
                     curr_node.right_child)

            else:
                # This is a leaf node. Iterate over the bucket to see
                # if any of of those are closer than the current best
                # value. Then reset curr_element to None to signal
                # that we should pop off the closest node in the heap
                # since we reached the leaf.
                for i in xrange(len(curr_node.bucket) - 1, -1, -1):
                    # We iterate backwards since we delete some entries.
                    entry = curr_node.bucket[i]
                    if entry.is_deleted:
                        del curr_node.bucket[i]
                        continue
                    entry_distance = euclidean_distance(entry.point, target)
                    if (nearest_distance is None or
                        entry_distance < nearest_distance):
                        nearest_distance = entry_distance
                        nearest_index = i
                        nearest_node = curr_node

                curr_element = None

    #-----------------------------------------------------------------------
    def traverse(self):
        "Creates an iterator for traversing the tree."
        if not self.top:  # Special case for empty tree.
            return []

        def traverse_aux(node):  # Node must not be None.

            # Check if this is a leaf node.
            if not node.left_child and not node.right_child:
                for entry in node.bucket:
                    if not entry.is_deleted:
                        yield entry
                return  # Since there are no more children.

            # If we are here, then this is not a leaf-node, and since
            # we store data only at the leaf nodes, we just traverse
            # the left and right children.
            if node.left_child:
                for entry in traverse_aux(node.left_child):
                    yield entry
            if node.right_child:
                for entry in traverse_aux(node.right_child):
                    yield entry

        return traverse_aux(self.top)

    #-----------------------------------------------------------------------
    def flatten(self):
        """
        Returns a list of (point, data) pairs resulting from a traversal
        of the tree.
        """
        return [e.flatten() for e in self.traverse()]

    #-----------------------------------------------------------------------
    def __str__(self):
        return str(self.flatten())


#=======================================================================
# Free helper functions

#-----------------------------------------------------------------------
def euclidean_distance(x, y):
    """ Returns the square of the euclidean distance between points x and y.

    x and y must both be iterables of number numerical types. If
    either one is not an iterable, then a TypeError exception will be
    thrown. They can contain any number of dimensions. If x and y
    contain a different number of dimensions, then the smaller of the
    two will be used.
    """
    # Note that it is slightly faster to use a list comprehension here
    # than a generator with our current data sets.
    return sum([(i-j)**2 for (i,j) in zip(x, y)])

#-----------------------------------------------------------------------
def find_min_max(lst):
    """
    Returns the (min_value, max_value), (min_index, max_index) of lst.

    lst can be any iterable. It doesn't have to be list or tuple.
    If lst is empty, then (None, None), (None, None) is returned.
    """
    if not lst:
        return (None, None), (None, None)

    min_index = max_index = 0
    min_value = max_value = None

    for index, entry in enumerate(lst):
        if max_value is None or entry > max_value:
            max_value = entry
            max_index = index
        if min_value is None or entry < min_value:
            min_value = entry
            min_index = index

    return (min_value, max_value), (min_index, max_index)

#-----------------------------------------------------------------------
def select(data, n, key=lambda x: x):
    """Find the nth rank ordered element (the least value has rank 0).

    This was taken from the Python Cookbook.
    """
    # make a new list, deal with <0 indices, check for valid index
    data = list(data)
    if n<0:
        n += len(data)
    if not 0 <= n < len(data):
        raise ValueError, "can't get rank %d out of %d" % (n, len(data))
    # main loop, quicksort-like but with no need for recursion
    while True:
        pivot = random.choice(data)
        pivot_key = key(pivot)
        pcount = 0
        under, over = [], []
        uappend, oappend = under.append, over.append
        for elem in data:
            elem_key = key(elem)
            if elem_key < pivot_key:
                uappend(elem)
            elif elem_key > pivot_key:
                oappend(elem)
            else:
                pcount += 1
        numunder = len(under)
        if n < numunder:
            data = under
        elif n < numunder + pcount:
            return pivot
        else:
            data = over
            n -= numunder + pcount

#-----------------------------------------------------------------------
if __name__ == "__main__":
    import doctest
    doctest.testmod()
