from point import makePointList, Point

class Cluster:
    """A class representing a cluster of points.

    Attributes:
      center: A Point object representing the exact center of the cluster.
      points: A set of Point objects that constitute our cluster.
    """

    def __init__(self, center=Point([0, 0])):
        """Inits a Cluster with a specific center (defaults to [0,0])."""
        self.center = center
        self.points = set()

    @property
    def coords(self):
        return self.center.coords

    @property
    def dim(self):
        return self.center.dim

    def addPoint(self, p):
        self.points.add(p)

    def removePoint(self, p):
        self.points.remove(p)

    @property
    def avgDistance(self):
        """Calculates the average distance of points in the cluster to the center.

        Returns:
          A float representing the average distance from all points in self.points
          to self.center.
        """
        if not self.points:
            return 0.0  # Avoid division by zero
        total_distance = 0.0
        for p in self.points:
            total_distance += p.distFrom(self.center)
        return total_distance / len(self.points)

    def updateCenter(self):
        """Updates self.center to be the average of all points in the cluster.

        If no points are in the cluster, then self.center should be unchanged.

        Returns:
          The coords of self.center.
        """
        if not self.points:
            return self.center.coords
        points_num = len(self.points)
        new_coords = [0.0] * self.dim

        for p in self.points:
            for i in range(self.dim):
                new_coords[i] += p.coords[i]

        new_coords = [x / points_num for x in new_coords]
        
        self.center = Point(new_coords)

        return self.center.coords
        
        # Hint: make sure self.center is a Point object after this function runs.

    def printAllPoints(self):
        print(str(self))
        for p in self.points:
            print("   {}".format(p))

    def __str__(self):
        return "Cluster: {} points and center = {}".format(
            len(self.points), self.center
        )

    def __repr__(self):
        return self.__str__()


def createClusters(data):
    """Creates clusters with centers from a k-by-d numpy array.

    Args:
      data: A k-by-d numpy array representing k d-dimensional points.

    Returns:
      A list of Clusters with each cluster centered at a d-dimensional
      point from each row of data.
    """
    centers = makePointList(data)
    return [Cluster(c) for c in centers]


if __name__ == "__main__":

    p1 = Point([2.5, 4.0])
    p2 = Point([3.0, 5.0])
    p3 = Point([1.0, 3.0])
    c = Cluster(Point([2.0, 2.0]))
    print(c)

    c.addPoint(p1)
    c.addPoint(p2)
    c.addPoint(p3)
    print("Updated", c)
    print("Average distance:", c.avgDistance)
    c.updateCenter()
    print("Updated", c)
    print("Updated average distance:", c.avgDistance)
    assert isinstance(
        c.center, Point
    ), "After updateCenter, the center must remain a Point object."
