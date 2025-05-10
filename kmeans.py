import numpy as np
from cluster import createClusters
from point import makePointList


def kmeans(point_data, cluster_data):
    """Performs k-means clustering on points.

    Args:
      point_data: a p-by-d numpy array used for creating a list of Points.
      cluster_data: A k-by-d numpy array used for creating a list of Clusters.

    Returns:
      A list of clusters (with update centers) after peforming k-means
      clustering on the points initialized from point_data
    """
    # Fill in

    # 1. Make list of points using makePointList and point_data
    points_list = makePointList(point_data)

    # 2. Make list of clusters using createClusters and cluster_data
    clusters_list = createClusters(cluster_data)

    # 3. For as long as points keep moving between clusters:
    moving = True
    while moving:
      moving = False

    #   A. Move every point to its closest cluster (use Point.closest and
    #     Point.moveToCluster)
    #     Hint: keep track here whether any point changed clusters by
    #           seeing if any moveToCluster call returns "True"
    #   for point in points_list:
    #      closest_cluster = point.closest(clusters_list)
    #      #closest_cluster = min(clusters_list, key=lambda c: point.distFrom(c.center))

    #      moved = point.moveToCluster(closest_cluster)
    #      if moved:
    #         moving = True
            
    for point in points_list:
        centers = [Cluster.center for Cluster in clusters_list]
        closest_center = point.closest(centers)
        for Cluster in clusters_list:
            if Cluster.center == closest_center:
                if point.moveToCluster(Cluster):
                    changed = True
                break
    # #   B. Update the centers for each cluster (use Cluster.updateCenter)
    #   for Cluster in clusters_list:
    #     Cluster.updateCenter()
        for Cluster in clusters_list:
          Cluster.updateCenter()
          Cluster.center.coords = np.array(Cluster.center.coords, dtype=float)
    # 4. Return the list of clusters, with the centers in their final positions
    return clusters_list


if __name__ == "__main__":
    data = np.array(
        [
            [12.1, -7.1], [0.5, 2.8], [1.2, 5.3], [10.3, -4.8], [-1.1, 3.9],
            [8.9, -3.6], [11.5, -6.2], [7.4, -2.5], [10.8, -5.5], [9.4, -4.3]
        ],
        dtype=float,
    )
    centers = np.array([[0, 0], [1, 1]], dtype=float)

    clusters = kmeans(data, centers)
    for c in clusters:
        c.printAllPoints()

# # k=3
# np.random.seed(0)
# init_center = student_data[np.random.choice(student_data.shape[0], size=k, replace=False)]
# points = makePointList(student_data)
# clusters = createClusters(init_center)
# clusters_final = kmeans(student_data, init_center)
# results

# # Get points for each cluster

# colors = ['red', 'green', 'blue', 'purple']
# for idx, c in enumerate(clusters_final):
#     points = np.array([p.coords for p in c.points])
#     plt.scatter(points[:, 0], points[:, 2], color=colors[idx], label=f'Cluster {idx}', alpha=0.5)

# plt.xlabel('fracSpent')
# plt.ylabel('fracPaused')
# plt.title('Student Clusters based on Video-Watching Behavior')
# plt.legend()
# plt.grid(True)
# plt.show()
# for idx, c in enumerate(clusters_final):
#     avg_behavior = np.mean([p.coords for p in c.points], axis=0)
#     print(f"Cluster {idx}: {len(c.points)} students")
#     print(f"  Average behavior: {np.round(avg_behavior, 4)}")
#     if avg_behavior[0] > 50:
#         print("  => Heavy watchers\n")
#     elif avg_behavior[2] > 100:
#         print("  => Heavy interaction group (lots of pauses)\n")
#     else:
#         print("  => Light watchers\n")